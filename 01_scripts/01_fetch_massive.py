"""MASSIVE（日本語）を JMTEB から取得し、02_data/massive/ に JSONL で保存する。

各行は {"id", "text", "scenario", "intent"}。シナリオ（18 種類）と意図（60 種類）のラベルを id で結合している。
ライセンス: CC BY 4.0（https://github.com/alexa/massive）

このリポジトリの環境で実行する:
    uv run python 01_scripts/01_fetch_massive.py
"""

import io
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

BASE_URL = "https://huggingface.co/datasets/sbintuitions/JMTEB/resolve/main/data"
OUT_DIR = Path(__file__).resolve().parent.parent / "02_data" / "massive"


def fetch_table(task, split):
    with urllib.request.urlopen(f"{BASE_URL}/{task}/{split}.parquet") as response:
        return pq.read_table(io.BytesIO(response.read())).to_pylist()


OUT_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "validation", "test"]:
    scenarios = fetch_table("massive_scenario_classification", split)
    intents = {row["id"]: row["label_text"] for row in fetch_table("massive_intent_classification", split)}

    out_path = OUT_DIR / f"{split}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in scenarios:
            record = {"id": row["id"], "text": row["text"], "scenario": row["label_text"], "intent": intents[row["id"]]}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{split}: {len(scenarios)} 件 → {out_path}")
