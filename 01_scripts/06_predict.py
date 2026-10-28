"""テキストを判定して、カテゴリーと確信度を返す。

05_train_final.py で保存した重み・温度・しきい値（02_data/models/massive_<label>.npz）を使う。
確信度は温度で補正した確率。--target で指定した正解率のしきい値を下回ったものは「要確認」と表示する。
テキストは引数で渡す。引数がなければ標準入力から 1 行 1 件で読む。

埋め込み用の環境で実行する:
    HF_HUB_OFFLINE=1 uv run --project ~/development/tools/embeddinggemma2-image python 01_scripts/06_predict.py "明日七時に起こして" "東京の天気は"
    HF_HUB_OFFLINE=1 uv run --project ~/development/tools/embeddinggemma2-image python 01_scripts/06_predict.py --label intent < texts.txt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent.parent / "02_data" / "models"

parser = argparse.ArgumentParser()
parser.add_argument("texts", nargs="*")
parser.add_argument("--label", choices=["scenario", "intent"], default="scenario")
parser.add_argument("--target", type=float, default=0.97, help="目標の正解率（05_train_final.py で決めたもの: 0.95 / 0.97）")
args = parser.parse_args()
texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]

weights = np.load(MODEL_DIR / f"massive_{args.label}.npz")
threshold = float(weights["thresholds"][list(weights["target_accuracies"]).index(args.target)])
embedder = SentenceTransformer(
    "google/embeddinggemma-2",
    device="mps",
    model_kwargs={"torch_dtype": torch.float32},  # float16 は NaN になるので使わない
    config_kwargs={"vision_config": None, "audio_config": None},  # テキストだけ使う
)

start = time.time()
embeddings = embedder.encode(texts, prompt_name="Classification", batch_size=64)
embed_seconds = time.time() - start

# ロジスティック回帰の確率（点数を温度で割ってから softmax）
start = time.time()
logits = (embeddings @ weights["coef"].T + weights["intercept"]) / weights["temperature"]
logits -= logits.max(axis=1, keepdims=True)
probabilities = np.exp(logits)
probabilities /= probabilities.sum(axis=1, keepdims=True)
classify_seconds = time.time() - start

for text, p in zip(texts, probabilities):
    best = p.argmax()
    status = "OK" if p[best] >= threshold else "要確認"
    print(f"{status}\t{weights['classes'][best]}\t{p[best]:.3f}\t{text}")
print(
    f"{len(texts)} 件  埋め込み {embed_seconds:.2f} 秒 / 判定 {classify_seconds * 1000:.2f} ミリ秒",
    file=sys.stderr,
)
