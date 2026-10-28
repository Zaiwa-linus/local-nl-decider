"""EmbeddingGemma 2 で口コミを埋め込むサンプル。

埋め込み用の環境で実行する:
    HF_HUB_OFFLINE=1 uv run --project ~/development/tools/embeddinggemma2-image python 01_scripts/embed_sample.py
"""

import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "google/embeddinggemma-2",
    device="mps",
    model_kwargs={"torch_dtype": torch.float32},  # float16 は NaN になるので使わない
    config_kwargs={"vision_config": None, "audio_config": None},  # テキストだけ使う
)

reviews = ["注文から3日で届いたが、箱がつぶれていた。", "味はいいけど量が少ない。"]

embeddings = model.encode(reviews, prompt_name="Classification", batch_size=64)
print("768 次元:", embeddings.shape)

# MRL で 256 次元に切り詰める（正規化し直す）
embeddings_256 = model.encode(reviews, prompt_name="Classification", truncate_dim=256, normalize_embeddings=True)
print("256 次元:", embeddings_256.shape)
