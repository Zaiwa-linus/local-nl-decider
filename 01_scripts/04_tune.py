"""判定機のパラメーターを validation で選び、test で評価する。間違えやすいグループの組み合わせも出す。

- ラベル: シナリオ（18 種類）と意図（60 種類）
- 次元: 768
- 判定機: ロジスティック回帰（C）、k-NN（k）、MLP（隠れ層の大きさ）
結果は 03_results/massive_tuned.json と 03_results/massive_tuned.md に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/04_tune.py
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "massive"
RESULTS_DIR = ROOT / "03_results"


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, np.load(DATA_DIR / f"{split}.npy")


CANDIDATES = {
    "logreg": [({"C": c}, lambda c=c: LogisticRegression(max_iter=3000, C=c)) for c in [1, 3, 10, 30, 100]],
    "knn": [
        ({"k": k}, lambda k=k: KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance"))
        for k in [3, 5, 10, 20, 40]
    ],
    "mlp": [
        (
            {"hidden": h},
            lambda h=h: MLPClassifier(hidden_layer_sizes=(h,), alpha=1e-3, early_stopping=True, max_iter=500, random_state=0),
        )
        for h in [256, 512, 1024]
    ],
}

train_records, X_train = load("train")
validation_records, X_validation = load("validation")
test_records, X_test = load("test")

results = []
confusions = {}
for label in ["scenario", "intent"]:
    y_train = [r[label] for r in train_records]
    y_validation = [r[label] for r in validation_records]
    y_test = np.array([r[label] for r in test_records])

    for name, candidates in CANDIDATES.items():
        # validation の Macro-F1 がいちばん高いパラメーターを選ぶ
        best = None
        for params, make in candidates:
            classifier = make().fit(X_train, y_train)
            score = f1_score(y_validation, classifier.predict(X_validation), average="macro")
            print(f"{label} {name} {params} validation macro_f1={score:.4f}")
            if best is None or score > best[0]:
                best = (score, params, make)

        validation_f1, params, make = best
        classifier = make()
        start = time.time()
        classifier.fit(X_train, y_train)
        train_seconds = time.time() - start

        start = time.time()
        predicted = classifier.predict(X_test)
        predict_seconds = time.time() - start

        result = {
            "label": label,
            "classifier": name,
            "params": params,
            "validation_macro_f1": round(validation_f1, 4),
            "accuracy": round(accuracy_score(y_test, predicted), 4),
            "macro_f1": round(f1_score(y_test, predicted, average="macro"), 4),
            "train_seconds": round(train_seconds, 2),
            "predict_per_second": round(len(y_test) / predict_seconds),
        }
        results.append(result)
        print(result)

        if name == "logreg":
            wrong = Counter((t, p) for t, p in zip(y_test, predicted) if t != p)
            confusions[label] = [{"true": t, "predicted": p, "count": c} for (t, p), c in wrong.most_common(10)]

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "massive_tuned.json").write_text(
    json.dumps({"results": results, "confusions": confusions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# MASSIVE（日本語）の評価結果（パラメーター調整後）",
    "",
    "768 次元。パラメーターは validation の Macro-F1 で選び、test で評価した。",
    "",
    "| ラベル | 判定機 | パラメーター | validation Macro-F1 | test Accuracy | test Macro-F1 | 学習時間（秒） | 判定の速さ（件/秒） |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for r in results:
    params = ", ".join(f"{k}={v}" for k, v in r["params"].items())
    lines.append(
        f"| {r['label']} | {r['classifier']} | {params} | {r['validation_macro_f1']:.3f} | {r['accuracy']:.3f} "
        f"| {r['macro_f1']:.3f} | {r['train_seconds']} | {r['predict_per_second']:,} |"
    )
for label, pairs in confusions.items():
    lines += ["", f"## 間違えやすい組み合わせ（{label}、ロジスティック回帰、上位 10）", "", "| 正解 | 予測 | 件数 |", "| --- | --- | --- |"]
    lines += [f"| {p['true']} | {p['predicted']} | {p['count']} |" for p in pairs]
(RESULTS_DIR / "massive_tuned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
