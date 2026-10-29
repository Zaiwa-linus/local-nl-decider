"""パブコメの結果の PDF から、寄せられた意見の要旨を 1 件ずつ取り出す。

- PDF の文字を pypdf で取り出し、数ページずつ DeepSeek に渡して、意見の要旨だけを原文のまま抜き出させる
  （省庁の考え方・回答、前置き、見出しは除く）
- 抜き出した文が PDF の本文に本当にあるかを、空白と改行を除いて比べて確かめる。ないもの（書き換えられたもの）は捨てる
- 結果は 02_data/pubcom/opinions.jsonl に 1 行 1 件で保存する。途中で止まっても、続きから再開できる
  各行は {"opinion_id", "case_id", "seq_no", "ministry", "title", "text"}

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/10_extract_opinions.py
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "pubcom"
API_URL = "https://api.deepseek.com/chat/completions"
CHUNK_CHARS = 4000
MIN_CHARS = 10

SYSTEM_PROMPT = (
    "これからパブリックコメント（意見募集）の結果をまとめた文書の一部を渡します。"
    "文書には、寄せられた意見の要旨と、それに対する省庁の考え方（回答）が並んでいます。\n"
    "寄せられた意見の要旨だけを、1 件ずつ分けて抜き出してください。\n"
    "- 文字は書き換えず、原文のまま写してください（行の途中の改行や余分な空白だけは除いてかまいません）\n"
    "- 省庁の考え方・回答、前置き、見出し、番号、表の列名は含めないでください\n"
    "- 意見が見つからなければ空の配列にしてください\n"
    '答えは JSON で {"opinions": ["意見1", "意見2", ...]} の形だけを返してください。'
)


def normalize(text):
    return re.sub(r"\s+", "", text)


def chunks_of(pdf_path):
    """PDF の文字を、ページの区切りで CHUNK_CHARS 程度ずつにまとめる。"""
    pages = [page.extract_text() or "" for page in PdfReader(pdf_path).pages]
    chunks, current = [], ""
    for page in pages:
        if current and len(current) + len(page) > CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current += page + "\n"
    if current.strip():
        chunks.append(current)
    return chunks


def ask_llm(text, model, api_key):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 8000,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("DeepSeek への問い合わせに 5 回失敗しました")


def extract(case, file, model, api_key):
    """1 つの PDF から意見を取り出し、PDF にあると確かめられたものだけを返す。"""
    try:
        chunks = chunks_of(ROOT / file["path"])
    except Exception as error:  # 壊れた PDF などは飛ばす
        return case, file, [], {"error": str(error)}
    opinions, stats = [], {"candidates": 0, "verified": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for chunk in chunks:
        result = ask_llm(chunk, model, api_key)
        stats["prompt_tokens"] += result["usage"]["prompt_tokens"]
        stats["completion_tokens"] += result["usage"]["completion_tokens"]
        try:
            candidates = json.loads(result["choices"][0]["message"]["content"]).get("opinions", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        source = normalize(chunk)
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            stats["candidates"] += 1
            text = re.sub(r"^[・\-\s]+", "", candidate).strip()
            if len(normalize(text)) >= MIN_CHARS and normalize(text) in source:
                stats["verified"] += 1
                opinions.append(text)
    return case, file, opinions, stats


parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--model", default="deepseek-flash")
args = parser.parse_args()
api_key = os.environ["DEEPSEEK_API_KEY"]

with (DATA_DIR / "cases.jsonl").open(encoding="utf-8") as f:
    cases = [json.loads(line) for line in f]
out_path = DATA_DIR / "opinions.jsonl"
done = set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as f:
        done = {(r["case_id"], r["seq_no"]) for r in map(json.loads, f)}
# 意見が 0 件だった PDF も済みとして記録する
done_path = DATA_DIR / "extracted_files.jsonl"
if done_path.exists():
    with done_path.open(encoding="utf-8") as f:
        done |= {(r["case_id"], r["seq_no"]) for r in map(json.loads, f)}

todo = [(c, f) for c in cases for f in c["files"] if (c["id"], f["seq_no"]) not in done]
print(f"PDF {sum(len(c['files']) for c in cases)} 件のうち、残り {len(todo)} 件")

start = time.time()
totals = {"candidates": 0, "verified": 0, "prompt_tokens": 0, "completion_tokens": 0, "opinions": 0}
with out_path.open("a", encoding="utf-8") as out, done_path.open("a", encoding="utf-8") as done_out, ThreadPoolExecutor(
    max_workers=args.workers
) as pool:
    futures = [pool.submit(extract, c, f, args.model, api_key) for c, f in todo]
    for count, future in enumerate(as_completed(futures), 1):
        case, file, opinions, stats = future.result()
        for k, text in enumerate(opinions):
            record = {
                "opinion_id": f"{case['id']}_{file['seq_no']}_{k}",
                "case_id": case["id"],
                "seq_no": file["seq_no"],
                "ministry": case["ministry"],
                "title": case["title"],
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        done_out.write(json.dumps({"case_id": case["id"], "seq_no": file["seq_no"], **stats}, ensure_ascii=False) + "\n")
        out.flush()
        done_out.flush()
        for key in ["candidates", "verified", "prompt_tokens", "completion_tokens"]:
            totals[key] += stats.get(key, 0)
        totals["opinions"] += len(opinions)
        if count % 20 == 0 or count == len(todo):
            print(f"{count}/{len(todo)} 件  意見 {totals['opinions']} 件  経過 {time.time() - start:.0f} 秒")

if todo:
    rate = totals["verified"] / max(totals["candidates"], 1)
    print(
        f"完了: {time.time() - start:.0f} 秒  意見 {totals['opinions']} 件（抜き出し {totals['candidates']} 件中、原文と一致 {rate:.1%}）  "
        f"入力 {totals['prompt_tokens']:,} トークン  出力 {totals['completion_tokens']:,} トークン"
    )
