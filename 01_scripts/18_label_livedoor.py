"""livedoor ニュースの記事に、DeepSeek でカテゴリー（9 種類）のラベルを付ける（蒸留用）。

正解のラベルは DeepSeek に渡さない。連載名ではなく、各カテゴリーの内容の説明だけを渡して、JSON で 1 つ選ばせる。
学習用（train）と評価用（test）の全件に付ける。評価用のラベルは、先生（DeepSeek）自体の正解率を測るのに使う。
--shots N を付けると、検証用（validation）からカテゴリーごとに N 件ずつ選んだ記事（先頭 200 文字）と正解を、例として一緒に渡す（few-shot）。
検証用は学習にも評価にも使わないので、例が評価に混ざらない。例に使った記事の ID は shots<N>.json に保存する。
--rules NAME を付けると、20_feedback_rules_livedoor.py で採用した境目のルール（03_results/04_livedoor/<NAME>.json）も渡す。
--only A,B を付けると、ルールなしのラベル（labels_shots<N>.jsonl）が A か B の記事だけを付け直し、ほかの記事はそのラベルを写す
（"copied": true）。ルールが関わる境目の記事だけに問い合わせて、料金を抑えるため。選ぶのに使うのは DeepSeek の判定だけで、正解は見ない。
プロンプトの組み立てと問い合わせは livedoor_llm.py にある。

結果は 02_data/livedoor/labels.jsonl（--shots N のときは labels_shots<N>.jsonl、--rules も付けると labels_shots<N>_rules.jsonl）に
1 行 1 件で追記する。途中で止まっても、続きから再開できる。
各行は {"split", "id", "label", "raw", "prompt_tokens", "cache_hit_tokens", "completion_tokens", "seconds"}。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/18_label_livedoor.py
    uv run --env-file .env python 01_scripts/18_label_livedoor.py --shots 5
    uv run --env-file .env python 01_scripts/18_label_livedoor.py --shots 5 \
        --rules 04_feedback_rules_dokujo --only dokujo-tsushin,peachy
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from livedoor_llm import DATA_DIR, ROOT, build_prompt, classify, parse_label

parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=48)
parser.add_argument("--model", default="deepseek-flash")
parser.add_argument("--shots", type=int, default=0)
parser.add_argument("--rules")
parser.add_argument("--only")
args = parser.parse_args()
api_key = os.environ["DEEPSEEK_API_KEY"]

articles = []
for split in ["train", "test"]:
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        articles += [{**json.loads(line), "split": split} for line in f]
rules = []
if args.rules:
    rules_path = ROOT / "03_results" / "04_livedoor" / f"{args.rules}.json"
    rules = [r["rule"] for r in json.loads(rules_path.read_text(encoding="utf-8"))["pairs"] if r["adopted"]]
system_prompt = build_prompt(args.shots, rules)
suffix = (f"_shots{args.shots}" if args.shots else "") + ("_rules" if args.rules else "")
out_path = DATA_DIR / f"labels{suffix}.jsonl"
done = set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as f:
        done = {json.loads(line)["id"] for line in f}
todo = [a for a in articles if a["id"] not in done]
if args.only:
    with (DATA_DIR / f"labels_shots{args.shots}.jsonl" if args.shots else DATA_DIR / "labels.jsonl").open(encoding="utf-8") as f:
        base = {r["id"]: r for r in map(json.loads, f)}
    only = set(args.only.split(","))
    with out_path.open("a", encoding="utf-8") as out:
        for article in [a for a in todo if base[a["id"]]["label"] not in only]:
            out.write(json.dumps({**base[article["id"]], "copied": True}, ensure_ascii=False) + "\n")
    todo = [a for a in todo if base[a["id"]]["label"] in only]
print(f"記事 {len(articles)} 件のうち、問い合わせるのは {len(todo)} 件（ルール {len(rules)} 個）")

start = time.time()
totals = {"prompt": 0, "cache_hit": 0, "completion": 0, "invalid": 0}
with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(classify, a["text"], system_prompt, api_key, args.model): a for a in todo}
    for count, future in enumerate(as_completed(futures), 1):
        article = futures[future]
        result, seconds = future.result()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        record = {
            "split": article["split"],
            "id": article["id"],
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
        if count % 1000 == 0 or count == len(todo):
            print(f"{count}/{len(todo)} 件  経過 {time.time() - start:.1f} 秒")

elapsed = time.time() - start
if todo:
    print(
        f"完了: {len(todo)} 件  {elapsed:.1f} 秒（{len(todo) / elapsed:.1f} 件/秒）  "
        f"入力 {totals['prompt']:,} トークン（うちキャッシュ {totals['cache_hit']:,}）  出力 {totals['completion']:,} トークン  "
        f"形式不正 {totals['invalid']} 件"
    )
