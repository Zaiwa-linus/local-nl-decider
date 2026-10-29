"""livedoor ニュースの記事に、DeepSeek でカテゴリー（9 種類）のラベルを付ける（蒸留用）。

正解のラベルは DeepSeek に渡さない。連載名ではなく、各カテゴリーの内容の説明だけを渡して、JSON で 1 つ選ばせる。
学習用（train）と評価用（test）の全件に付ける。評価用のラベルは、先生（DeepSeek）自体の正解率を測るのに使う。
--shots N を付けると、検証用（validation）からカテゴリーごとに N 件ずつ選んだ記事（先頭 200 文字）と正解を、例として一緒に渡す（few-shot）。
検証用は学習にも評価にも使わないので、例が評価に混ざらない。例に使った記事の ID は shots<N>.json に保存する。

結果は 02_data/livedoor/labels.jsonl（--shots N のときは labels_shots<N>.jsonl）に 1 行 1 件で追記する。途中で止まっても、続きから再開できる。
各行は {"split", "id", "label", "raw", "prompt_tokens", "cache_hit_tokens", "completion_tokens", "seconds"}。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/18_label_livedoor.py
    uv run --env-file .env python 01_scripts/18_label_livedoor.py --shots 5
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

SHOT_CHARS = 200
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "livedoor"
API_URL = "https://api.deepseek.com/chat/completions"

CRITERIA = {
    "topic-news": "社会・事件・芸能・ネットで話題になったことなど、一般のニュース",
    "sports-watch": "スポーツ（野球、サッカー、格闘技など）の試合や選手の話題",
    "it-life-hack": "パソコン・ソフト・ネットサービス・周辺機器など、IT の製品や使い方",
    "kaden-channel": "家電・デジタル機器・ガジェットの製品や話題",
    "movie-enter": "映画（新作、俳優・監督、公開・試写会、興行）",
    "dokujo-tsushin": "独身で働く女性の恋愛・結婚・仕事・生活についてのコラム",
    "smax": "スマートフォン・携帯電話・タブレット、そのアプリや通信会社",
    "livedoor-homme": "男性向けの仕事・キャリア・お金・ライフスタイル",
    "peachy": "女性向けの美容・ファッション・恋愛・グルメ・エンタメ・ライフスタイル",
}


def build_prompt(shots):
    prompt = (
        "これからニュースサイトの記事（タイトルと本文の先頭）を 1 件渡します。次のカテゴリーからいちばん近いものを 1 つ選んでください。\n\n"
        + "\n".join(f"- {name}: {description}" for name, description in CRITERIA.items())
    )
    if shots:
        with (DATA_DIR / "validation.jsonl").open(encoding="utf-8") as f:
            validation = [json.loads(line) for line in f]
        rng = random.Random(0)
        used = []
        prompt += "\n\nカテゴリーは記事が載った連載で決まっています。各カテゴリーの記事の例（先頭だけ）を示します。\n"
        for name in CRITERIA:
            for article in rng.sample([a for a in validation if a["category"] == name], shots):
                used.append(article["id"])
                example = article["text"][:SHOT_CHARS].replace("\n", " ")
                prompt += f"\n[{name}] {example}"
        (DATA_DIR / f"shots{shots}.json").write_text(json.dumps(used) + "\n", encoding="utf-8")
    return prompt + '\n\n答えは JSON で {"category": "<カテゴリー名>"} の形だけを返してください。'


def ask_llm(text, model, api_key, system_prompt):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
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
    return label if label in CRITERIA else None


parser = argparse.ArgumentParser()
parser.add_argument("--workers", type=int, default=48)
parser.add_argument("--model", default="deepseek-flash")
parser.add_argument("--shots", type=int, default=0)
args = parser.parse_args()
api_key = os.environ["DEEPSEEK_API_KEY"]

articles = []
for split in ["train", "test"]:
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        articles += [{**json.loads(line), "split": split} for line in f]
system_prompt = build_prompt(args.shots)
out_path = DATA_DIR / (f"labels_shots{args.shots}.jsonl" if args.shots else "labels.jsonl")
done = set()
if out_path.exists():
    with out_path.open(encoding="utf-8") as f:
        done = {json.loads(line)["id"] for line in f}
todo = [a for a in articles if a["id"] not in done]
print(f"記事 {len(articles)} 件のうち、残り {len(todo)} 件")

start = time.time()
totals = {"prompt": 0, "cache_hit": 0, "completion": 0, "invalid": 0}
with out_path.open("a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(ask_llm, a["text"], args.model, api_key, system_prompt): a for a in todo}
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
