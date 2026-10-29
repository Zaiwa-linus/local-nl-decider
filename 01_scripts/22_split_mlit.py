"""02_data/mlit/defects_<年>.jsonl を、学習・検証・評価用に分けて 02_data/mlit/<split>.jsonl に保存する。

出典: 国土交通省「自動車不具合情報ホットライン」の不具合情報（https://renrakuda.mlit.go.jp/ ）を加工して作成。PDL1.0

- 判定に使う文は「申告内容の要約」だけ。車名や走行距離は使わない
- 文と装置名は NFKC で正規化する（全角の英数字と半角のカナをそろえる）
- 【】で囲んだ印（「【ＡＳＶ検索対象】」「【エンジン故障】」など）は、検索用の印や答えになるので取り除く
- 件数がごく少ない「装置その他」「非装置」「タイヤ」は「その他」にまとめる
- 装置名ごとの割合を保って、学習用 70%・検証用 15%・評価用 15% に分ける（乱数の種は 0）
各行は {"id", "category", "text"}。id は管理番号。

このリポジトリの環境で実行する:
    uv run python 01_scripts/22_split_mlit.py
"""

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "02_data" / "mlit"
MERGE_INTO_OTHER = {"装置その他", "非装置", "タイヤ"}

reports = []
for path in sorted(DATA_DIR.glob("defects_*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = unicodedata.normalize("NFKC", record["text"])
            text = re.sub(r"\s+", " ", re.sub(r"【[^】]*】", "", text)).strip()
            category = unicodedata.normalize("NFKC", record["device"])
            if category in MERGE_INTO_OTHER:
                category = "その他"
            if text:
                reports.append({"id": record["control_no"], "category": category, "text": text})

by_category = defaultdict(list)
for report in reports:
    by_category[report["category"]].append(report)
splits = {"train": [], "validation": [], "test": []}
rng = random.Random(0)
for category in sorted(by_category):
    items = by_category[category]
    rng.shuffle(items)
    n_validation = n_test = round(len(items) * 0.15)
    splits["validation"] += items[:n_validation]
    splits["test"] += items[n_validation : n_validation + n_test]
    splits["train"] += items[n_validation + n_test :]

for split, items in splits.items():
    with (DATA_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{split}: {len(items)} 件")
print(dict(Counter(r["category"] for r in reports).most_common()))
