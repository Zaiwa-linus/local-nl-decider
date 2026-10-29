"""パブコメの意見をランダムに取り出して DeepSeek に見せ、意見をグループ分けするカテゴリーを決めさせる。

--scheme で分け方を選ぶ。
- type（既定）: どの省庁・どの政策分野の意見でも使えるように、「意見の内容の種類」で分ける。
  結果は 02_data/pubcom/categories.json と 03_results/02_pubcom_type/01_categories.md
- topic: 意見の「話題（政策分野）」で分ける。
  結果は 02_data/pubcom/categories_topic.json と 03_results/03_pubcom_topic/01_categories.md
思考ありで 1 回だけ問い合わせる（type は deepseek-v4-pro、topic は deepseek-flash で作った）。
保存するのはカテゴリーの名前と説明だけで、意見の本文は含めない。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/11_design_categories.py --scheme topic --model deepseek-flash
"""

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "pubcom"
RESULTS_DIR = ROOT / "03_results"
API_URL = "https://api.deepseek.com/chat/completions"

TYPE_RULE = """- 政策分野（医療、環境など）ではなく、「意見の内容の種類」で分けてください。どの省庁・どの政策分野の意見にも同じカテゴリーが使えるようにするためです"""
TOPIC_RULE = """- 意見の種類（賛成・反対・質問など）ではなく、「何についての意見か」という話題（政策分野）で分けてください
- 特定の 1 つの案件だけに当てはまる細かすぎる話題ではなく、ほかの省庁の意見募集でも使える広さにしてください"""

PROMPT = """以下は、日本の省庁が行ったパブリックコメント（意見募集）に寄せられた意見の要旨です。さまざまな省庁・政策分野のものが混ざっています。

これらの意見を自動でグループ分けする判定機を作ります。そのためのカテゴリーを設計してください。
{rule}
- カテゴリーは 6〜12 個にしてください
- 1 件の意見がどれか 1 つにはっきり入るように、カテゴリー同士の境目をはっきりさせてください
- どれにも当てはまらない意見のために「その他」を 1 つ入れてください
- 名前は英小文字とアンダースコアの短い ID、説明は日本語で、何が入って何が入らないかが分かるように書いてください

答えは JSON で {"categories": [{"name": "...", "description": "..."}, ...]} の形だけを返してください。

意見:
"""

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", choices=["type", "topic"], default="type")
parser.add_argument("--n", type=int, default=300, help="見せる意見の件数")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default="deepseek-v4-pro")
args = parser.parse_args()

with (DATA_DIR / "opinions.jsonl").open(encoding="utf-8") as f:
    opinions = [json.loads(line) for line in f]
sample = random.Random(args.seed).sample(opinions, min(args.n, len(opinions)))
suffix = "" if args.scheme == "type" else "_topic"
content = PROMPT.replace("{rule}", TYPE_RULE if args.scheme == "type" else TOPIC_RULE) + "\n".join(f"- {o['text'][:300]}" for o in sample)

body = {
    "model": args.model,
    "messages": [{"role": "user", "content": content}],
    "response_format": {"type": "json_object"},
    "thinking": {"type": "enabled"},
    "max_tokens": 16000,
}
request = urllib.request.Request(
    API_URL,
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)

categories = json.loads(result["choices"][0]["message"]["content"])["categories"]
(DATA_DIR / f"categories{suffix}.json").write_text(json.dumps(categories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

results_dir = RESULTS_DIR / ("02_pubcom_type" if args.scheme == "type" else "03_pubcom_topic")
results_dir.mkdir(parents=True, exist_ok=True)
lines = [
    f"# パブコメの意見のカテゴリー（{'意見の種類' if args.scheme == 'type' else '話題'}、DeepSeek が設計）",
    "",
    f"意見 {len(sample)} 件をランダムに見せて、{args.model}（思考あり）に設計させた。",
    "",
    "| 名前 | ID | 説明 |",
    "| --- | --- | --- |",
]
lines += [f"| {ja(c['name'])} | `{c['name']}` | {c['description']} |" for c in categories]
(results_dir / "01_categories.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

usage = result["usage"]
print(f"カテゴリー {len(categories)} 個  入力 {usage['prompt_tokens']:,} トークン  出力 {usage['completion_tokens']:,} トークン")
for c in categories:
    print(f"- {c['name']}: {c['description']}")
