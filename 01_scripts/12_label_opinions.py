"""パブコメの意見に、DeepSeek でカテゴリーのラベルを付ける。このラベルを正解として判定機を学習する（蒸留）。

カテゴリーは 11_design_categories.py で決めたものを使う。--scheme type（既定）なら categories.json、topic なら categories_topic.json。
結果は 02_data/pubcom/labels.jsonl（topic なら labels_topic.jsonl。--out で変えられる）に 1 行 1 件で追記する。途中で止まっても、続きから再開できる。
各行は {"opinion_id", "label", "raw", "prompt_tokens", "cache_hit_tokens", "completion_tokens", "seconds"}。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/12_label_opinions.py
"""

import argparse
import http.client
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "pubcom"
API_URL = "https://api.deepseek.com/chat/completions"

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", choices=["type", "topic"], default="type")
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--model", default="deepseek-flash")
parser.add_argument("--out", help="02_data/pubcom/ の下の出力ファイル名（既定は labels.jsonl、topic なら labels_topic.jsonl）")
parser.add_argument("--sample", type=int, default=0, help="0 なら全件。正の数ならその件数だけ無作為に選ぶ（再現性の確認用）")
args = parser.parse_args()
suffix = "" if args.scheme == "type" else "_topic"

categories = json.loads((DATA_DIR / f"categories{suffix}.json").read_text(encoding="utf-8"))
NAMES = {c["name"] for c in categories}
SYSTEM_PROMPT = (
    f"これからパブリックコメント（意見募集）に寄せられた意見を 1 件渡します。{'意見の内容の種類' if args.scheme == 'type' else '意見の話題'}として、次のカテゴリーからいちばん近いものを 1 つ選んでください。\n\n"
    + "\n".join(f"- {c['name']}: {c['description']}" for c in categories)
    + '\n\n答えは JSON で {"category": "<カテゴリー名>"} の形だけを返してください。'
)


def ask_llm(text, model, api_key):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 30,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            start = time.time()
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            return result, time.time() - start
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("DeepSeek への問い合わせに 5 回失敗しました")


def parse_label(content):
    try:
        label = json.loads(content).get("category")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in NAMES else None


api_key = os.environ["DEEPSEEK_API_KEY"]

with (DATA_DIR / "opinions.jsonl").open(encoding="utf-8") as f:
    opinions = [json.loads(line) for line in f]
if args.sample:
    opinions = random.Random(0).sample(opinions, args.sample)
out_path = DATA_DIR / (args.out or f"labels{suffix}.jsonl")
done = set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as f:
        done = {json.loads(line)["opinion_id"] for line in f}
todo = [o for o in opinions if o["opinion_id"] not in done]
print(f"意見 {len(opinions)} 件のうち、残り {len(todo)} 件")

start = time.time()
totals = {"prompt": 0, "cache_hit": 0, "completion": 0, "invalid": 0}
with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(ask_llm, o["text"], args.model, api_key): o for o in todo}
    for count, future in enumerate(as_completed(futures), 1):
        opinion = futures[future]
        result, seconds = future.result()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        record = {
            "opinion_id": opinion["opinion_id"],
            "label": parse_label(content),
            "raw": content,
            "prompt_tokens": usage["prompt_tokens"],
            "cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
            "completion_tokens": usage["completion_tokens"],
            "seconds": round(seconds, 3),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        out.flush()
        totals["prompt"] += record["prompt_tokens"]
        totals["cache_hit"] += record["cache_hit_tokens"]
        totals["completion"] += record["completion_tokens"]
        totals["invalid"] += record["label"] is None
        if count % 500 == 0 or count == len(todo):
            print(f"{count}/{len(todo)} 件  経過 {time.time() - start:.1f} 秒")

elapsed = time.time() - start
if todo:
    print(
        f"完了: {len(todo)} 件  {elapsed:.1f} 秒（{len(todo) / elapsed:.1f} 件/秒）  "
        f"入力 {totals['prompt']:,} トークン（うちキャッシュ {totals['cache_hit']:,}）  出力 {totals['completion']:,} トークン  "
        f"形式不正 {totals['invalid']} 件"
    )
