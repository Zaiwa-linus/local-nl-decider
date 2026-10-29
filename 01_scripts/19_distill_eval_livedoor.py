"""livedoor ニュースで、DeepSeek のラベルで学習した判定機と、正解のラベルで学習した判定機を比べる。

- 先生（DeepSeek）の正解率: 評価用（test）で、DeepSeek のラベルと正解を比べる
- 判定機（ロジスティック回帰）: 学習用（train）の正解のラベル / DeepSeek のラベルで学習し、評価用の正解で評価する
  C は、それぞれのラベルを使った学習用の中の交差検証（5 分割）で選ぶ
- 学習件数を 500 / 1,000 / 2,000 / 全件に変えた場合と、確信度のしきい値ごとの回答率・正解率も出す
結果は 03_results/04_livedoor/01_distill.md と 01_distill.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/19_distill_eval_livedoor.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "livedoor"
RESULTS_DIR = ROOT / "03_results" / "04_livedoor"
C_CANDIDATES = [1, 3, 10, 30, 100]
SIZES = [500, 1000, 2000]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, np.load(DATA_DIR / f"{split}.npy")


def choose_c(X, y):
    scores = {}
    for c in C_CANDIDATES:
        fold_scores = []
        for fit_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y):
            classifier = LogisticRegression(max_iter=3000, C=c).fit(X[fit_idx], y[fit_idx])
            fold_scores.append(f1_score(y[val_idx], classifier.predict(X[val_idx]), average="macro"))
        scores[c] = float(np.mean(fold_scores))
    return max(scores, key=scores.get), scores


train_records, X_train = load("train")
test_records, X_test = load("test")
with (DATA_DIR / "labels.jsonl").open(encoding="utf-8") as f:
    llm = {r["id"]: r["label"] for r in map(json.loads, f)}

gold_train = np.array([r["category"] for r in train_records])
llm_train = np.array([llm.get(r["id"]) or "" for r in train_records])
gold_test = np.array([r["category"] for r in test_records])
llm_test = np.array([llm.get(r["id"]) or "" for r in test_records])

teacher_accuracy = float((llm_test == gold_test).mean())
teacher_train_agreement = float((llm_train == gold_train).mean())
teacher_report = classification_report(gold_test, llm_test, output_dict=True, zero_division=0)
teacher_confusions = Counter((t, p) for t, p in zip(gold_test, llm_test) if t != p).most_common(10)
print(f"先生の正解率（test）: {teacher_accuracy:.4f}  train での一致率: {teacher_train_agreement:.4f}")

valid = llm_train != ""
order = np.random.default_rng(0).permutation(len(train_records))
order = order[valid[order]]
results = {}
for name, y_train in [("gold", gold_train), ("llm", llm_train)]:
    best_c, cv_scores = choose_c(X_train[valid], y_train[valid])
    print(f"{name}: C={best_c}  {cv_scores}")
    rows = []
    for size in SIZES + [len(order)]:
        idx = order[:size]
        classifier = LogisticRegression(max_iter=3000, C=best_c).fit(X_train[idx], y_train[idx])
        probabilities = classifier.predict_proba(X_test)
        predicted = classifier.classes_[probabilities.argmax(axis=1)]
        rows.append({
            "size": int(size),
            "accuracy": round(float((predicted == gold_test).mean()), 4),
            "macro_f1": round(float(f1_score(gold_test, predicted, average="macro")), 4),
            "agreement_with_llm_test": round(float((predicted == llm_test).mean()), 4),
        })
    confidence = probabilities.max(axis=1)
    correct = predicted == gold_test
    results[name] = {
        "C": best_c,
        "cv_macro_f1": cv_scores,
        "rows": rows,
        "per_class_f1": {k: round(v["f1-score"], 3) for k, v in classification_report(gold_test, predicted, output_dict=True).items() if k in set(gold_test)},
        "by_confidence": [
            {"threshold": t, "coverage": round(float((confidence >= t).mean()), 3), "accuracy": round(float(correct[confidence >= t].mean()), 3)}
            for t in THRESHOLDS
        ],
        "confusions": [{"true": t, "predicted": p, "count": c} for (t, p), c in Counter((t, p) for t, p in zip(gold_test, predicted) if t != p).most_common(10)],
    }
    print(name, rows)

summary = {
    "train": len(train_records),
    "test": len(test_records),
    "teacher_accuracy": round(teacher_accuracy, 4),
    "teacher_train_agreement": round(teacher_train_agreement, 4),
    "teacher_per_class": {k: {m: round(v[m], 3) for m in ["precision", "recall", "f1-score"]} for k, v in teacher_report.items() if k in set(gold_test)},
    "teacher_confusions": [{"true": t, "llm": p, "count": c} for (t, p), c in teacher_confusions],
    "classifiers": results,
}
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "01_distill.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

gold, distilled = results["gold"], results["llm"]
lines = [
    "# 蒸留の評価（livedoor ニュース、DeepSeek のラベル）",
    "",
    f"学習用 {len(train_records):,} 件・評価用 {len(test_records):,} 件。文はタイトル＋本文の先頭 500 文字（連載名は取り除いた）。",
    "埋め込みは EmbeddingGemma 2（`Classification` プレフィックス）、判定機はロジスティック回帰。評価はすべて評価用の正解のラベルで行った。",
    "",
    f"- 先生（deepseek-flash、思考なし）の正解率（評価用）: **{teacher_accuracy:.3f}**（学習用での正解との一致率 {teacher_train_agreement:.3f}）",
    f"- C は学習用の中の交差検証で選んだ（正解のラベル: C={gold['C']}、DeepSeek のラベル: C={distilled['C']}）",
    "",
    "## 学習件数と正解率（評価用）",
    "",
    "| 学習件数 | DeepSeek のラベル: 正解率 | DeepSeek のラベル: Macro-F1 | 正解のラベル: 正解率 | 正解のラベル: Macro-F1 |",
    "| --- | --- | --- | --- | --- |",
]
for d, g in zip(distilled["rows"], gold["rows"]):
    lines.append(f"| {d['size']:,} | {d['accuracy']:.3f} | {d['macro_f1']:.3f} | {g['accuracy']:.3f} | {g['macro_f1']:.3f} |")
lines += [
    "",
    "## 確信度のしきい値ごとの回答率と正解率（評価用、全件で学習、補正なし）",
    "",
    "各セルは「回答率 / 回答したものの正解率」。",
    "",
    "| しきい値 | DeepSeek のラベルで学習 | 正解のラベルで学習 |",
    "| --- | --- | --- |",
]
for d, g in zip(distilled["by_confidence"], gold["by_confidence"]):
    lines.append(f"| {d['threshold']} | {d['coverage']:.3f} / {d['accuracy']:.3f} | {g['coverage']:.3f} / {g['accuracy']:.3f} |")
lines += [
    "",
    "## カテゴリーごとの F1（評価用）",
    "",
    "| カテゴリー | 先生（DeepSeek） | DeepSeek のラベルで学習 | 正解のラベルで学習 |",
    "| --- | --- | --- | --- |",
]
for name in sorted(set(gold_test), key=lambda n: -summary["teacher_per_class"][n]["f1-score"]):
    lines.append(f"| {ja(name)} | {summary['teacher_per_class'][name]['f1-score']:.3f} | {distilled['per_class_f1'][name]:.3f} | {gold['per_class_f1'][name]:.3f} |")
lines += ["", "## 先生（DeepSeek）が間違えた組み合わせ（評価用、上位 10）", "", "| 正解 | DeepSeek | 件数 |", "| --- | --- | --- |"]
lines += [f"| {ja(t)} | {ja(p)} | {c} |" for (t, p), c in teacher_confusions]
lines += ["", "## 正解のラベルで学習した判定機が間違えた組み合わせ（評価用、上位 10）", "", "| 正解 | 判定機 | 件数 |", "| --- | --- | --- |"]
lines += [f"| {ja(r['true'])} | {ja(r['predicted'])} | {r['count']} |" for r in gold["confusions"]]
(RESULTS_DIR / "01_distill.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
