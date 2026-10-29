"""埋め込みを特徴量にして判定機を学習し、テストデータで評価する。

- ラベル: シナリオ（18 種類）と意図（60 種類）
- 次元: 768 と 256（256 は先頭を切り詰めて正規化し直す）
- 判定機: ロジスティック回帰、k-NN
- 学習は train、評価は test を使う
結果は 03_results/01_massive/01_baseline.json と 01_baseline.md に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/03_train_eval.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "massive"
RESULTS_DIR = ROOT / "03_results" / "01_massive"


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, np.load(DATA_DIR / f"{split}.npy")


def truncate(embeddings, dim):
    truncated = embeddings[:, :dim]
    return truncated / np.linalg.norm(truncated, axis=1, keepdims=True)


CLASSIFIERS = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=10.0),
    "knn": lambda: KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance"),
}

train_records, train_embeddings = load("train")
test_records, test_embeddings = load("test")

results = []
for label in ["scenario", "intent"]:
    y_train = [r[label] for r in train_records]
    y_test = [r[label] for r in test_records]
    for dim in [768, 256]:
        X_train = truncate(train_embeddings, dim)
        X_test = truncate(test_embeddings, dim)
        for name, make in CLASSIFIERS.items():
            classifier = make()
            start = time.time()
            classifier.fit(X_train, y_train)
            train_seconds = time.time() - start

            start = time.time()
            predicted = classifier.predict(X_test)
            predict_seconds = time.time() - start

            result = {
                "label": label,
                "classes": len(set(y_train)),
                "dim": dim,
                "classifier": name,
                "accuracy": round(accuracy_score(y_test, predicted), 4),
                "macro_f1": round(f1_score(y_test, predicted, average="macro"), 4),
                "train_seconds": round(train_seconds, 2),
                "predict_per_second": round(len(y_test) / predict_seconds),
            }
            results.append(result)
            print(result)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "01_baseline.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# MASSIVE（日本語）の評価結果",
    "",
    f"学習 {len(train_records):,} 件、テスト {len(test_records):,} 件。埋め込みは EmbeddingGemma 2（`Classification` プレフィックス）。",
    "",
    "| ラベル | クラス数 | 次元 | 判定機 | Accuracy | Macro-F1 | 学習時間（秒） | 判定の速さ（件/秒） |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for r in results:
    lines.append(
        f"| {ja(r['label'])} | {r['classes']} | {r['dim']} | {r['classifier']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
        f"| {r['train_seconds']} | {r['predict_per_second']:,} |"
    )
(RESULTS_DIR / "01_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
