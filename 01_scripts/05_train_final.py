"""判定に使うロジスティック回帰を学習し、確信度を補正して、しきい値を決める。

1. train で学習する
2. validation で温度（temperature scaling）を決める。点数を温度で割ってから softmax にすると、確信度が実際の正解率に近づく
3. validation で、目標の正解率を満たすいちばん低いしきい値を決める
4. test で、確信度のずれ（ECE）と、しきい値を使ったときの回答率・正解率を確かめる（test では何も決めない）
5. 重み・温度・しきい値を 02_data/models/massive_<label>.npz に保存する

温度としきい値は train だけで学習した判定機に合わせて決めているので、validation を足して学習し直すことはしない。
結果は 03_results/01_massive/03_confidence.md と 03_confidence.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/05_train_final.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "massive"
MODEL_DIR = ROOT / "02_data" / "models"
RESULTS_DIR = ROOT / "03_results" / "01_massive"

# 04_tune.py で validation から選んだ値
C_BY_LABEL = {"scenario": 30, "intent": 100}
TARGET_ACCURACIES = [0.95, 0.97]
BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
TEMPERATURES = np.linspace(0.3, 3.0, 271)


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records, np.load(DATA_DIR / f"{split}.npy")


def softmax(logits, temperature=1.0):
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def reliability(confidence, correct):
    """確信度の区間ごとの件数・平均確信度・正解率と、ECE（確信度と正解率の差の加重平均）。"""
    rows, ece = [], 0.0
    for low, high in zip(BINS[:-1], BINS[1:]):
        in_bin = (confidence >= low) & ((confidence < high) if high < 1.0 else (confidence <= high))
        if not in_bin.any():
            continue
        mean_confidence, accuracy = confidence[in_bin].mean(), correct[in_bin].mean()
        ece += in_bin.mean() * abs(mean_confidence - accuracy)
        rows.append({"bin": f"{low}–{high}", "count": int(in_bin.sum()), "confidence": round(float(mean_confidence), 3), "accuracy": round(float(accuracy), 3)})
    return rows, round(float(ece), 4)


def choose_threshold(confidence, correct, target):
    """回答したものの正解率が target 以上になる、いちばん低いしきい値。"""
    for threshold in np.round(np.arange(0.0, 1.0, 0.01), 2):
        answered = confidence >= threshold
        if answered.any() and correct[answered].mean() >= target:
            return float(threshold)
    return 1.0


train_records, X_train = load("train")
validation_records, X_validation = load("validation")
test_records, X_test = load("test")

summary = {}
lines = [
    "# 確信度の補正としきい値（MASSIVE、ロジスティック回帰）",
    "",
    "train で学習し、温度としきい値は validation で決め、test で確かめた。",
    "ECE は「確信度と実際の正解率の差」の平均（0 に近いほど、確信度がそのまま当たる確率として使える）。",
]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for label, c in C_BY_LABEL.items():
    y_train = [r[label] for r in train_records]
    y_validation = np.array([r[label] for r in validation_records])
    y_test = np.array([r[label] for r in test_records])

    classifier = LogisticRegression(max_iter=3000, C=c).fit(X_train, y_train)
    classes = classifier.classes_
    validation_logits = classifier.decision_function(X_validation)
    test_logits = classifier.decision_function(X_test)

    # validation の対数尤度がいちばん高くなる温度を選ぶ
    label_index = {name: i for i, name in enumerate(classes)}
    validation_targets = np.array([label_index[y] for y in y_validation])
    nll = [-np.log(softmax(validation_logits, t)[np.arange(len(validation_targets)), validation_targets] + 1e-12).mean() for t in TEMPERATURES]
    temperature = float(TEMPERATURES[int(np.argmin(nll))])

    test_predicted = classes[test_logits.argmax(axis=1)]
    test_correct = test_predicted == y_test
    validation_correct = classes[validation_logits.argmax(axis=1)] == y_validation

    before_rows, before_ece = reliability(softmax(test_logits).max(axis=1), test_correct)
    after_rows, after_ece = reliability(softmax(test_logits, temperature).max(axis=1), test_correct)

    validation_confidence = softmax(validation_logits, temperature).max(axis=1)
    test_confidence = softmax(test_logits, temperature).max(axis=1)
    thresholds = []
    for target in TARGET_ACCURACIES:
        threshold = choose_threshold(validation_confidence, validation_correct, target)
        answered = test_confidence >= threshold
        thresholds.append({
            "target": target,
            "threshold": threshold,
            "validation_coverage": round(float((validation_confidence >= threshold).mean()), 3),
            "test_coverage": round(float(answered.mean()), 3),
            "test_accuracy": round(float(test_correct[answered].mean()), 3),
        })

    summary[label] = {
        "temperature": temperature,
        "test_accuracy": round(float(test_correct.mean()), 4),
        "ece_before": before_ece,
        "ece_after": after_ece,
        "reliability_before": before_rows,
        "reliability_after": after_rows,
        "thresholds": thresholds,
    }
    print(label, json.dumps({k: v for k, v in summary[label].items() if not k.startswith("reliability")}, ensure_ascii=False))

    np.savez(
        MODEL_DIR / f"massive_{label}.npz",
        coef=classifier.coef_,
        intercept=classifier.intercept_,
        classes=classes,
        temperature=temperature,
        thresholds=np.array([t["threshold"] for t in thresholds]),
        target_accuracies=np.array(TARGET_ACCURACIES),
    )

    lines += [
        "",
        f"## {ja(label)}",
        "",
        f"- 温度: {temperature:.2f}（1 より大きいと、補正前は自信過剰だったということ）",
        f"- test の正解率（全件）: {test_correct.mean():.3f}",
        f"- ECE: 補正前 {before_ece:.4f} → 補正後 {after_ece:.4f}",
        "",
        "### 確信度の区間ごとの正解率（test）",
        "",
        "| 区間 | 補正前: 件数 | 補正前: 平均確信度 | 補正前: 正解率 | 補正後: 件数 | 補正後: 平均確信度 | 補正後: 正解率 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    before_by_bin = {r["bin"]: r for r in before_rows}
    after_by_bin = {r["bin"]: r for r in after_rows}
    for low, high in zip(BINS[:-1], BINS[1:]):
        key = f"{low}–{high}"
        b, a = before_by_bin.get(key), after_by_bin.get(key)
        cells = [f"{r['count']} | {r['confidence']:.3f} | {r['accuracy']:.3f}" if r else "0 | – | –" for r in (b, a)]
        lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "### しきい値（validation で決めて、test で確かめた）",
        "",
        "| 目標の正解率 | しきい値 | validation の回答率 | test の回答率 | test の正解率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {t['target']} | {t['threshold']:.2f} | {t['validation_coverage']:.3f} | {t['test_coverage']:.3f} | {t['test_accuracy']:.3f} |"
        for t in thresholds
    ]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "03_confidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(RESULTS_DIR / "03_confidence.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
