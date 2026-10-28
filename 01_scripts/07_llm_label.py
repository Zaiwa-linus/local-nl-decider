"""MASSIVE の train から一部を取り出し、DeepSeek にシナリオ（18 種類）のラベルを付けさせる（蒸留用の学習データ作り）。

正解のラベルは DeepSeek に渡さない。グループの名前と説明だけを渡して、JSON で 1 つ選ばせる。
DeepSeek の規約（Open Platform Terms of Service §4.2(3)）で、出力を別のモデルの学習（蒸留）に使うことが認められている。
結果は 02_data/deepseek/massive_scenario_labels.jsonl に 1 行 1 件で追記する。途中で止まっても、続きから再開できる。
各行は {"rank"（抽出の順番）, "index"（train の行番号）, "id", "label", "raw", "prompt_tokens", "cache_hit_tokens", "completion_tokens", "seconds"}。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/07_llm_label.py --n 2000
"""

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "02_data" / "massive" / "train.jsonl"
OUT_PATH = ROOT / "02_data" / "deepseek" / "massive_scenario_labels.jsonl"
API_URL = "https://api.deepseek.com/chat/completions"

CRITERIA = {
    "alarm": "アラームの設定・確認・解除",
    "audio": "音量の調整、ミュート",
    "calendar": "予定やリマインダーの登録・確認・削除",
    "cooking": "料理のレシピや作り方",
    "datetime": "現在の日付や時刻、時差",
    "email": "メールの送信・確認、連絡先",
    "general": "雑談、冗談、あいさつ、その他どれにも当てはまらないもの",
    "iot": "照明・家電・掃除機などスマート家電の操作",
    "lists": "買い物リストや ToDo リストの作成・追加・確認",
    "music": "音楽の好みや曲の情報（再生以外）",
    "news": "ニュースの確認",
    "play": "音楽・ラジオ・ポッドキャスト・オーディオブック・ゲームの再生",
    "qa": "事実についての質問、定義、計算、通貨換算、株価など",
    "recommendation": "お店・イベント・映画などのおすすめ",
    "social": "SNS への投稿、企業への苦情や問い合わせ",
    "takeaway": "料理の注文や配達",
    "transport": "電車・タクシー・交通情報・チケット",
    "weather": "天気や気温",
}
SYSTEM_PROMPT = (
    "これからスマートスピーカーへの話しかけを 1 件渡します。何をしてほしいという依頼か、次のグループからいちばん近いものを 1 つ選んでください。\n\n"
    + "\n".join(f"- {name}: {description}" for name, description in CRITERIA.items())
    + '\n\n答えは JSON で {"scenario": "<グループ名>"} の形だけを返してください。'
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
        except (urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(f"DeepSeek への問い合わせに 5 回失敗しました: {text}")


def parse_label(content):
    try:
        label = json.loads(content).get("scenario")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in CRITERIA else None


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000, help="ラベルを付ける件数")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--workers", type=int, default=16, help="同時に送るリクエスト数")
parser.add_argument("--model", default="deepseek-flash")
args = parser.parse_args()
api_key = os.environ["DEEPSEEK_API_KEY"]

with TRAIN_PATH.open(encoding="utf-8") as f:
    train = [json.loads(line) for line in f]
# 抽出の順番（rank）を残しておき、先頭から 500 件・1,000 件…と部分集合を取れるようにする
sampled = random.Random(args.seed).sample(range(len(train)), args.n)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
done = set()
if OUT_PATH.exists():
    with OUT_PATH.open(encoding="utf-8") as f:
        done = {json.loads(line)["index"] for line in f}
todo = [(rank, i) for rank, i in enumerate(sampled) if i not in done]
print(f"対象 {len(sampled)} 件のうち、済み {len(sampled) - len(todo)} 件、残り {len(todo)} 件")

start = time.time()
totals = {"prompt": 0, "cache_hit": 0, "completion": 0, "invalid": 0}
with OUT_PATH.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(ask_llm, train[i]["text"], args.model, api_key): (rank, i) for rank, i in todo}
    for count, future in enumerate(as_completed(futures), 1):
        rank, i = futures[future]
        result, seconds = future.result()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]
        record = {
            "rank": rank,
            "index": i,
            "id": train[i]["id"],
            "label": parse_label(content),
            "raw": content,
            "model": result["model"],
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
        if count % 200 == 0 or count == len(todo):
            print(f"{count}/{len(todo)} 件  経過 {time.time() - start:.1f} 秒")

elapsed = time.time() - start
if todo:
    print(
        f"完了: {len(todo)} 件  {elapsed:.1f} 秒（{len(todo) / elapsed:.1f} 件/秒）  "
        f"入力 {totals['prompt']:,} トークン（うちキャッシュ {totals['cache_hit']:,}）  出力 {totals['completion']:,} トークン  "
        f"形式不正 {totals['invalid']} 件"
    )
