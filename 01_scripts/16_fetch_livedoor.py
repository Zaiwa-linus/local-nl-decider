"""livedoor ニュースコーパスを取得し、学習・検証・評価用に分けて 02_data/livedoor/<split>.jsonl に保存する。

配布元: https://www.rondhuit.com/download.html#ldcc （株式会社ロンウイット）
ライセンス: 各記事に CC BY-ND 2.1 JP（表示 - 改変禁止）。手元での学習・評価にだけ使い、記事も加工したものも配布しない。

- 判定に使う文は「タイトル + 本文」の先頭 500 文字
- 文中の連載名（「【Sports Watch】」「S-MAX」「独女」など）は、そのままだとカテゴリーの答えになるので取り除く
- カテゴリーごとの割合を保って、学習用 70%・検証用 15%・評価用 15% に分ける（乱数の種は 0）
各行は {"id", "category", "text"}。

このリポジトリの環境で実行する:
    uv run python 01_scripts/16_fetch_livedoor.py
"""

import json
import random
import re
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "livedoor"
URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
MAX_CHARS = 500
# 連載名・サイト名。長いものから順に取り除く
SOURCE_NAMES = [
    "【Sports Watch】", "Sports Watch", "ITライフハック", "家電チャンネル", "MOVIE ENTER", "エスマックス", "S-MAX",
    "livedoor HOMME", "HOMME", "Peachy", "トピックニュース", "livedoor ニュース", "livedoor", "独女通信", "独女",
]

archive = DATA_DIR / "ldcc-20140209.tar.gz"
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    urllib.request.urlretrieve(URL, archive)

articles = []
with tarfile.open(archive) as tar:
    for member in sorted(tar.getmembers(), key=lambda m: m.name):
        parts = member.name.split("/")
        if len(parts) != 3 or not parts[2].endswith(".txt") or parts[2] == "LICENSE.txt":
            continue
        lines = tar.extractfile(member).read().decode("utf-8").split("\n")
        text = lines[2] + "\n" + "\n".join(lines[3:])
        for name in SOURCE_NAMES:
            text = text.replace(name, "")
        text = re.sub(r"[ \t　]+", " ", re.sub(r"\n\s*\n+", "\n", text)).strip()
        articles.append({"id": parts[2][:-4], "category": parts[1], "text": text[:MAX_CHARS]})

by_category = defaultdict(list)
for article in articles:
    by_category[article["category"]].append(article)
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
print(dict(Counter(a["category"] for a in articles).most_common()))
