"""国交省の自動車不具合情報で、正解の装置名をそのまま学習した判定機の精度を測る（蒸留と比べるための基準）。

- 判定機（ロジスティック回帰）: 学習用（train）の正解で学習し、評価用（test）で評価する
  C は、学習用の中の交差検証（5 分割）で選ぶ
- 学習件数を 500 / 1,000 / 2,000 / 全件に変えた場合と、確信度のしきい値ごとの回答率・正解率、取り違えやすい組み合わせも出す
結果は 03_results/05_mlit/01_baseline.md と 01_baseline.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/24_baseline_mlit.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "mlit"
RESULTS_DIR = ROOT / "03_results" / "05_mlit"
C_CANDIDATES = [1, 3, 10, 30, 100]
SIZES = [500, 1000, 2000]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return np.array([r["category"] for r in records]), np.load(DATA_DIR / f"{split}.npy")


def choose_c(X, y):
    scores = {}
    for c in C_CANDIDATES:
        fold_scores = []
        for fit_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
            classifier = LogisticRegression(max_iter=3000, C=c).fit(X[fit_idx], y[fit_idx])
            fold_scores.append(f1_score(y[val_idx], classifier.predict(X[val_idx]), average="macro"))
        scores[c] = float(np.mean(fold_scores))
    return max(scores, key=scores.get), scores


y_train, X_train = load("train")
y_test, X_test = load("test")
best_c, c_scores = choose_c(X_train, y_train)

classifier = LogisticRegression(max_iter=3000, C=best_c).fit(X_train, y_train)
proba = classifier.predict_proba(X_test)
predicted = classifier.classes_[proba.argmax(axis=1)]
confidence = proba.max(axis=1)
accuracy = float((predicted == y_test).mean())
macro_f1 = float(f1_score(y_test, predicted, average="macro"))

by_size = {}
rng = np.random.default_rng(0)
for size in SIZES:
    idx = rng.choice(len(y_train), size, replace=False)
    small = LogisticRegression(max_iter=3000, C=best_c).fit(X_train[idx], y_train[idx])
    by_size[size] = float((small.predict(X_test) == y_test).mean())
by_size[len(y_train)] = accuracy

by_threshold = {}
for threshold in THRESHOLDS:
    answered = confidence >= threshold
    by_threshold[threshold] = {
        "coverage": float(answered.mean()),
        "accuracy": float((predicted[answered] == y_test[answered]).mean()) if answered.any() else None,
    }

confusions = Counter((t, p) for t, p in zip(y_test, predicted) if t != p).most_common(10)
report = classification_report(y_test, predicted, output_dict=True, zero_division=0)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "train": len(y_train),
    "test": len(y_test),
    "C": best_c,
    "cv_macro_f1": c_scores,
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "by_size": by_size,
    "by_threshold": by_threshold,
    "confusions": [{"true": t, "predicted": p, "count": n} for (t, p), n in confusions],
    "report": report,
}
(RESULTS_DIR / "01_baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 国交省の自動車不具合情報: 正解の装置名で学習した判定機（基準）",
    "",
    f"- 学習用 {len(y_train):,} 件の正解で学習し、評価用 {len(y_test):,} 件で評価した（ロジスティック回帰、C={best_c}）",
    f"- 正解率 {accuracy:.3f}、Macro-F1 {macro_f1:.3f}",
    "",
    "## 学習件数ごとの正解率",
    "",
    "| 学習件数 | 正解率 |",
    "| --- | --- |",
    *[f"| {size:,} | {acc:.3f} |" for size, acc in by_size.items()],
    "",
    "## 確信度のしきい値ごとの回答率と正解率",
    "",
    "| しきい値 | 回答率 | 正解率 |",
    "| --- | --- | --- |",
    *[f"| {t} | {v['coverage']:.3f} | {v['accuracy']:.3f} |" for t, v in by_threshold.items()],
    "",
    "## 装置名ごとの精度",
    "",
    "| 装置名 | 件数 | 適合率 | 再現率 | F1 |",
    "| --- | --- | --- | --- | --- |",
    *[
        f"| {name} | {v['support']:.0f} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} |"
        for name, v in sorted(report.items(), key=lambda item: -item[1]["support"] if isinstance(item[1], dict) else 0)
        if name in set(y_test)
    ],
    "",
    "## 取り違えやすい組み合わせ（正解 → 判定）",
    "",
    "| 正解 | 判定 | 件数 |",
    "| --- | --- | --- |",
    *[f"| {t} | {p} | {n} |" for (t, p), n in confusions],
    "",
]
(RESULTS_DIR / "01_baseline.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
