"""LLM（DeepSeek）が付けたラベルで判定機を学習し、正解のラベルで学習したものと比べる（蒸留の評価）。

- LLM のラベルと正解のラベルがどれくらい一致するか
- 学習に使う件数（500 / 1,000 / 2,000 件）ごとに、LLM のラベルで学習した判定機と、同じ文を正解のラベルで学習した判定機を test で評価する
- 参考として、train 全件（11,514 件）を正解のラベルで学習した判定機も評価する
- 確信度（いちばん高い確率）のしきい値ごとに、回答率と回答したものの正解率を test で出す（2,000 件で学習したものと train 全件で学習したもの）
  本番では正解のラベルがないので、確信度は温度で補正しない
判定機はロジスティック回帰（C=30、05_train_final.py と同じ）。評価には test の正解ラベルを使う。
結果は 03_results/01_massive/04_distill.md と 04_distill.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/08_distill_eval.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "massive"
LABELS_PATH = ROOT / "02_data" / "deepseek" / "massive_scenario_labels.jsonl"
RESULTS_DIR = ROOT / "03_results" / "01_massive"
SIZES = [500, 1000, 2000]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, np.load(DATA_DIR / f"{split}.npy")


def evaluate(X, y, X_test, y_test):
    predicted = LogisticRegression(max_iter=3000, C=30).fit(X, y).predict(X_test)
    return round(accuracy_score(y_test, predicted), 4), round(f1_score(y_test, predicted, average="macro"), 4)


def by_confidence(X, y, X_test, y_test):
    """しきい値ごとの回答率と、回答したものの正解率。"""
    classifier = LogisticRegression(max_iter=3000, C=30).fit(X, y)
    probabilities = classifier.predict_proba(X_test)
    confidence = probabilities.max(axis=1)
    correct = classifier.classes_[probabilities.argmax(axis=1)] == np.array(y_test)
    return [
        {"threshold": t, "coverage": round(float((confidence >= t).mean()), 3), "accuracy": round(float(correct[confidence >= t].mean()), 3)}
        for t in THRESHOLDS
    ]


train_records, X_train = load("train")
test_records, X_test = load("test")
y_test = [r["scenario"] for r in test_records]

with LABELS_PATH.open(encoding="utf-8") as f:
    labeled = sorted((json.loads(line) for line in f), key=lambda r: r["rank"])
labeled = [r for r in labeled if r["label"] is not None]

llm_labels = [r["label"] for r in labeled]
true_labels = [train_records[r["index"]]["scenario"] for r in labeled]
agreement = float(np.mean([a == b for a, b in zip(llm_labels, true_labels)]))
disagreements = Counter((t, p) for t, p in zip(true_labels, llm_labels) if t != p).most_common(10)
print(f"LLM のラベルと正解の一致率: {agreement:.3f}（{len(labeled)} 件）")

rows = []
for size in SIZES:
    subset = labeled[:size]
    indices = [r["index"] for r in subset]
    X = X_train[indices]
    llm_accuracy, llm_f1 = evaluate(X, [r["label"] for r in subset], X_test, y_test)
    true_accuracy, true_f1 = evaluate(X, [train_records[i]["scenario"] for i in indices], X_test, y_test)
    rows.append({"size": size, "llm_accuracy": llm_accuracy, "llm_macro_f1": llm_f1, "true_accuracy": true_accuracy, "true_macro_f1": true_f1})
    print(rows[-1])

full_accuracy, full_f1 = evaluate(X_train, [r["scenario"] for r in train_records], X_test, y_test)
print(f"train 全件・正解のラベル: accuracy={full_accuracy} macro_f1={full_f1}")

largest = labeled[: SIZES[-1]]
largest_X = X_train[[r["index"] for r in largest]]
confidence_rows = {
    f"DeepSeek のラベル {len(largest):,} 件": by_confidence(largest_X, [r["label"] for r in largest], X_test, y_test),
    f"正解のラベル {len(largest):,} 件": by_confidence(largest_X, [train_records[r["index"]]["scenario"] for r in largest], X_test, y_test),
    f"正解のラベル {len(train_records):,} 件（train 全件）": by_confidence(X_train, [r["scenario"] for r in train_records], X_test, y_test),
}
for name, values in confidence_rows.items():
    print(name, values)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "04_distill.json").write_text(
    json.dumps(
        {
            "agreement": round(agreement, 4),
            "labeled": len(labeled),
            "rows": rows,
            "full_train": {"size": len(train_records), "accuracy": full_accuracy, "macro_f1": full_f1},
            "disagreements": [{"true": t, "llm": p, "count": c} for (t, p), c in disagreements],
            "by_confidence": confidence_rows,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

lines = [
    "# 蒸留の評価（MASSIVE シナリオ、DeepSeek のラベル）",
    "",
    f"DeepSeek（deepseek-flash、思考なし）が train の {len(labeled):,} 件に付けたラベルで判定機（ロジスティック回帰）を学習し、test（{len(y_test):,} 件）の正解ラベルで評価した。",
    "",
    f"- LLM のラベルと正解のラベルの一致率: **{agreement:.3f}**",
    "",
    "| 学習件数 | LLM のラベル: Accuracy | LLM のラベル: Macro-F1 | 正解のラベル: Accuracy | 正解のラベル: Macro-F1 |",
    "| --- | --- | --- | --- | --- |",
]
lines += [
    f"| {r['size']:,} | {r['llm_accuracy']:.3f} | {r['llm_macro_f1']:.3f} | {r['true_accuracy']:.3f} | {r['true_macro_f1']:.3f} |"
    for r in rows
]
lines.append(f"| {len(train_records):,}（train 全件） | – | – | {full_accuracy:.3f} | {full_f1:.3f} |")
names = list(confidence_rows)
lines += [
    "",
    "## 確信度のしきい値ごとの回答率と正解率（test）",
    "",
    "確信度（いちばん高い確率、温度の補正なし）がしきい値未満のものは「答えない」とした場合。各セルは「回答率 / 回答したものの正解率」。",
    "",
    "| しきい値 | " + " | ".join(names) + " |",
    "| --- | " + " | ".join("---" for _ in names) + " |",
]
for k, threshold in enumerate(THRESHOLDS):
    cells = [f"{confidence_rows[n][k]['coverage']:.3f} / {confidence_rows[n][k]['accuracy']:.3f}" for n in names]
    lines.append(f"| {threshold} | " + " | ".join(cells) + " |")
lines += ["", "## LLM と正解が食い違った組み合わせ（上位 10）", "", "| 正解 | LLM | 件数 |", "| --- | --- | --- |"]
lines += [f"| {ja(t)} | {ja(p)} | {c} |" for (t, p), c in disagreements]
(RESULTS_DIR / "04_distill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
