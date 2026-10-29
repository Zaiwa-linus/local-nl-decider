"""DeepSeek のラベルを正解として、パブコメの意見の判定機を学習・評価する。

- 案件単位で学習用（80%）と評価用（20%）に分ける（同じ PDF の意見が両方に入らないようにする）
- 判定機はロジスティック回帰。C は学習用の中で案件単位の交差検証をして選ぶ
- 評価用で、正解率・Macro-F1・カテゴリーごとの F1・確信度別の回答率と正解率を出す
- 学習件数を減らした場合（25% / 50% / 100%）の正解率も出す
結果は 03_results/02_pubcom_type/02_classifier.{md,json} に保存する。
--scheme topic なら、話題のラベル（labels_topic.jsonl）を使い、03_results/03_pubcom_topic/02_classifier.{md,json} に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/14_train_eval_pubcom.py [--scheme topic]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "pubcom"
RESULTS_DIR = ROOT / "03_results"
C_CANDIDATES = [1, 3, 10, 30]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]
FRACTIONS = [0.25, 0.5, 1.0]

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", choices=["type", "topic"], default="type")
args = parser.parse_args()
suffix = "" if args.scheme == "type" else "_topic"
results_dir = RESULTS_DIR / ("02_pubcom_type" if args.scheme == "type" else "03_pubcom_topic")

with (DATA_DIR / "opinions.jsonl").open(encoding="utf-8") as f:
    opinions = [json.loads(line) for line in f]
embeddings = np.load(DATA_DIR / "opinions.npy")
with (DATA_DIR / f"labels{suffix}.jsonl").open(encoding="utf-8") as f:
    labels = {r["opinion_id"]: r["label"] for r in map(json.loads, f)}

keep = [i for i, o in enumerate(opinions) if labels.get(o["opinion_id"])]
X = embeddings[keep]
y = np.array([labels[opinions[i]["opinion_id"]] for i in keep])
groups = np.array([opinions[i]["case_id"] for i in keep])
print(f"意見 {len(keep)} 件、案件 {len(set(groups))} 件、カテゴリー {len(set(y))} 個")

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# C を学習用の中の交差検証（案件単位、5 分割）で選ぶ
scores = {}
for c in C_CANDIDATES:
    fold_scores = []
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(X_train, y_train, g_train):
        classifier = LogisticRegression(max_iter=3000, C=c).fit(X_train[fit_idx], y_train[fit_idx])
        fold_scores.append(f1_score(y_train[val_idx], classifier.predict(X_train[val_idx]), average="macro"))
    scores[c] = float(np.mean(fold_scores))
    print(f"C={c}  交差検証 Macro-F1={scores[c]:.4f}")
best_c = max(scores, key=scores.get)

classifier = LogisticRegression(max_iter=3000, C=best_c).fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)
predicted = classifier.classes_[probabilities.argmax(axis=1)]
confidence = probabilities.max(axis=1)
correct = predicted == y_test

accuracy = float(correct.mean())
macro_f1 = float(f1_score(y_test, predicted, average="macro"))
report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
by_confidence = [
    {"threshold": t, "coverage": round(float((confidence >= t).mean()), 3), "accuracy": round(float(correct[confidence >= t].mean()), 3)}
    for t in THRESHOLDS
]

rng = np.random.default_rng(0)
train_cases = np.array(sorted(set(g_train)))
learning_curve = []
for fraction in FRACTIONS:
    chosen = set(rng.choice(train_cases, size=max(1, int(len(train_cases) * fraction)), replace=False))
    mask = np.array([g in chosen for g in g_train])
    partial = LogisticRegression(max_iter=3000, C=best_c).fit(X_train[mask], y_train[mask]).predict(X_test)
    learning_curve.append({"fraction": fraction, "size": int(mask.sum()), "accuracy": round(float(accuracy_score(y_test, partial)), 4)})

confusions = Counter((t, p) for t, p in zip(y_test, predicted) if t != p).most_common(10)
summary = {
    "opinions": len(keep),
    "cases": len(set(groups)),
    "train": len(train_idx),
    "test": len(test_idx),
    "label_distribution": dict(Counter(y).most_common()),
    "cv_macro_f1": scores,
    "C": best_c,
    "accuracy": round(accuracy, 4),
    "macro_f1": round(macro_f1, 4),
    "per_class": {k: {m: round(v[m], 3) for m in ["precision", "recall", "f1-score", "support"]} for k, v in report.items() if k in set(y)},
    "by_confidence": by_confidence,
    "learning_curve": learning_curve,
    "confusions": [{"deepseek": t, "classifier": p, "count": c} for (t, p), c in confusions],
}
print(json.dumps({k: v for k, v in summary.items() if k not in ["per_class"]}, ensure_ascii=False))

results_dir.mkdir(parents=True, exist_ok=True)
(results_dir / "02_classifier.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    f"# パブコメの意見の判定機（{'意見の種類' if args.scheme == 'type' else '話題'}、DeepSeek のラベルを正解として学習・評価）",
    "",
    f"意見 {len(keep):,} 件（案件 {len(set(groups)):,} 件）。案件単位で学習用 {len(train_idx):,} 件・評価用 {len(test_idx):,} 件に分けた。",
    f"埋め込みは EmbeddingGemma 2（`Classification` プレフィックス）、判定機はロジスティック回帰（C={best_c}、学習用の中の交差検証で選んだ）。",
    "",
    f"- 評価用での DeepSeek のラベルとの一致率（正解率）: **{accuracy:.3f}**",
    f"- Macro-F1: {macro_f1:.3f}",
    "",
    "## カテゴリーごとの結果（評価用）",
    "",
    "| カテゴリー | 全体の件数 | 評価用の件数 | 適合率 | 再現率 | F1 |",
    "| --- | --- | --- | --- | --- | --- |",
]
for name, count in Counter(y).most_common():
    r = summary["per_class"].get(name, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
    lines.append(f"| {ja(name)} | {count} | {int(r['support'])} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} |")
lines += ["", "## 確信度のしきい値ごとの回答率と正解率（評価用、補正なし）", "", "| しきい値 | 回答率 | 正解率 |", "| --- | --- | --- |"]
lines += [f"| {r['threshold']} | {r['coverage']:.3f} | {r['accuracy']:.3f} |" for r in by_confidence]
lines += ["", "## 学習件数と正解率（評価用）", "", "| 学習用の案件の割合 | 学習件数 | 正解率 |", "| --- | --- | --- |"]
lines += [f"| {int(r['fraction'] * 100)}% | {r['size']:,} | {r['accuracy']:.3f} |" for r in learning_curve]
lines += ["", "## 間違えやすい組み合わせ（評価用、上位 10）", "", "| DeepSeek | 判定機 | 件数 |", "| --- | --- | --- |"]
lines += [f"| {ja(t)} | {ja(p)} | {c} |" for (t, p), c in confusions]
(results_dir / "02_classifier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
