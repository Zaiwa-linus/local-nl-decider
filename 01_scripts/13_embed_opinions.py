"""パブコメの意見（02_data/pubcom/opinions.jsonl）を EmbeddingGemma 2 で埋め込み、02_data/pubcom/opinions.npy に保存する。

行の順番は opinions.jsonl と同じ。768 次元のまま保存する。

埋め込み用の環境で実行する:
    HF_HUB_OFFLINE=1 uv run --project ~/development/tools/embeddinggemma2-image python 01_scripts/13_embed_opinions.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "02_data" / "pubcom"

model = SentenceTransformer(
    "google/embeddinggemma-2",
    device="mps",
    model_kwargs={"torch_dtype": torch.float32},  # float16 は NaN になるので使わない
    config_kwargs={"vision_config": None, "audio_config": None},  # テキストだけ使う
)

with (DATA_DIR / "opinions.jsonl").open(encoding="utf-8") as f:
    texts = [json.loads(line)["text"] for line in f]

start = time.time()
embeddings = model.encode(texts, prompt_name="Classification", batch_size=32)
elapsed = time.time() - start

np.save(DATA_DIR / "opinions.npy", embeddings.astype(np.float32))
lengths = sorted(len(t) for t in texts)
print(f"{embeddings.shape}  {elapsed:.1f} 秒（{len(texts) / elapsed:.0f} 件/秒）  文字数の中央値 {lengths[len(lengths) // 2]}")
