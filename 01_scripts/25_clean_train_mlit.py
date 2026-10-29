"""国交省の自動車不具合情報で、教師データから怪しい申告を埋め込みで見つけて除き、判定機を学習し直す。

- 学習用（train）の申告を、その申告を含まない判定機（5 分割の交差検証）で判定し、2 種類の怪しさを測る
  - 付け間違いの疑い: 判定がほかの装置名になった申告を、「ほかの装置名の確率 − 付いている装置名の確率」が大きい順に並べる
  - 境目: すべての申告を、いちばん高い確率（確信度）が低い順に並べる
- それぞれ学習用の 5% / 10% / 20% を除いて学習し直す。比べるために、同じ件数をランダムに除いた場合（乱数の種 3 つの平均）も出す
- 検証用（validation）と評価用（test）は全件のまま評価する。どちらも除かない
  - 正解率、Macro-F1、正解率 90% / 95% を保てる回答率（確信度の高い順に答えたとき、正解率が目標を下回らない最大の割合）
  - 除く前の判定機で確信度が低かった（0.5 未満）申告と、高かった（0.8 以上）申告に分けた正解率と平均の確信度
判定機はロジスティック回帰（C=30、24_baseline_mlit.py で選んだ値）。
結果は 03_results/05_mlit/02_clean_train.md と 02_clean_train.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/25_clean_train_mlit.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "mlit"
RESULTS_DIR = ROOT / "03_results" / "05_mlit"
C = 30
RATES = [0.05, 0.10, 0.20]
RANDOM_SEEDS = [0, 1, 2]
TARGETS = [0.90, 0.95]


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        labels = np.array([json.loads(line)["category"] for line in f])
    return labels, np.load(DATA_DIR / f"{split}.npy")


def fit(X, y):
    return LogisticRegression(max_iter=3000, C=C).fit(X, y)


def coverage_at(correct, confidence, target):
    """確信度の高い順に答えたとき、正解率が target を下回らない最大の回答率。"""
    order = np.argsort(-confidence)
    cumulative = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cumulative >= target)[0]
    return float((ok.max() + 1) / len(order)) if len(ok) else 0.0


def evaluate(classifier, X, y, base_confidence):
    proba = classifier.predict_proba(X)
    predicted = classifier.classes_[proba.argmax(axis=1)]
    confidence = proba.max(axis=1)
    correct = predicted == y
    low, high = base_confidence < 0.5, base_confidence >= 0.8
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        **{f"coverage@{t}": coverage_at(correct, confidence, t) for t in TARGETS},
        "low_accuracy": float(correct[low].mean()),
        "low_confidence": float(confidence[low].mean()),
        "high_accuracy": float(correct[high].mean()),
        "high_confidence": float(confidence[high].mean()),
    }


def average(results):
    return {key: float(np.mean([r[key] for r in results])) for key in results[0]}


y_train, X_train = load("train")
eval_sets = {split: load(split) for split in ["validation", "test"]}

# 学習用の申告を、その申告を含まない判定機で判定する
oof = np.zeros((len(y_train), len(np.unique(y_train))))
for fit_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_train, y_train):
    classifier = fit(X_train[fit_idx], y_train[fit_idx])
    oof[val_idx] = classifier.predict_proba(X_train[val_idx])
classes = np.unique(y_train)
given = oof[np.arange(len(y_train)), np.searchsorted(classes, y_train)]
oof_predicted = classes[oof.argmax(axis=1)]
other = np.where(oof_predicted != y_train, oof.max(axis=1), 0.0)

mislabel_order = [i for i in np.argsort(-(other - given)) if oof_predicted[i] != y_train[i]]
ambiguous_order = list(np.argsort(oof.max(axis=1)))

base = fit(X_train, y_train)
base_confidence = {split: base.predict_proba(X).max(axis=1) for split, (y, X) in eval_sets.items()}
results = {split: {"なし（全件）": evaluate(base, X, y, base_confidence[split])} for split, (y, X) in eval_sets.items()}
removed_pairs = {}

for rate in RATES:
    n_remove = round(len(y_train) * rate)
    for name, order in [("付け間違いの疑い", mislabel_order), ("境目", ambiguous_order)]:
        removed = np.array(order[:n_remove])
        keep = np.setdiff1d(np.arange(len(y_train)), removed)
        classifier = fit(X_train[keep], y_train[keep])
        for split, (y, X) in eval_sets.items():
            results[split][f"{name} {rate:.0%}"] = evaluate(classifier, X, y, base_confidence[split])
        removed_pairs[f"{name} {rate:.0%}"] = Counter(
            f"{y_train[i]} → {oof_predicted[i]}" for i in removed
        ).most_common(8)

    random_runs = {split: [] for split in eval_sets}
    for seed in RANDOM_SEEDS:
        keep = np.setdiff1d(np.arange(len(y_train)), np.random.default_rng(seed).choice(len(y_train), n_remove, replace=False))
        classifier = fit(X_train[keep], y_train[keep])
        for split, (y, X) in eval_sets.items():
            random_runs[split].append(evaluate(classifier, X, y, base_confidence[split]))
    for split in eval_sets:
        results[split][f"ランダム {rate:.0%}"] = average(random_runs[split])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "train": len(y_train),
    "oof_accuracy": float((oof_predicted == y_train).mean()),
    "mislabel_candidates": len(mislabel_order),
    "results": results,
    "removed_pairs": removed_pairs,
}
(RESULTS_DIR / "02_clean_train.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

order_names = ["なし（全件）"] + [f"{name} {rate:.0%}" for rate in RATES for name in ["付け間違いの疑い", "境目", "ランダム"]]
lines = [
    "# 国交省の自動車不具合情報: 教師データから怪しい申告を除いて学習し直す",
    "",
    f"- 学習用 {len(y_train):,} 件を交差検証（5 分割）で判定すると、付いている装置名と合ったのは {summary['oof_accuracy']:.1%}。合わなかった {len(mislabel_order):,} 件が付け間違いの疑いの候補",
    "- 付け間違いの疑い: 判定がほかの装置名になった申告を、「ほかの装置名の確率 − 付いている装置名の確率」が大きい順に除く",
    "- 境目: 確信度（いちばん高い確率）が低い順に除く",
    "- ランダム: 同じ件数をランダムに除く（乱数の種 3 つの平均）",
    "- 評価用・検証用は全件のまま。「低い」「高い」は、除く前の判定機での確信度（0.5 未満 / 0.8 以上）で分けた申告",
    "",
]
for split in eval_sets:
    n_low = int((base_confidence[split] < 0.5).sum())
    n_high = int((base_confidence[split] >= 0.8).sum())
    lines += [
        f"## {'評価用（test）' if split == 'test' else '検証用（validation）'} {len(eval_sets[split][0]):,} 件（低い {n_low} 件、高い {n_high} 件）",
        "",
        "| 除いたもの | 正解率 | Macro-F1 | 正解率 90% の回答率 | 正解率 95% の回答率 | 低い: 正解率 | 低い: 平均確信度 | 高い: 正解率 | 高い: 平均確信度 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in order_names:
        r = results[split][name]
        lines.append(
            f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['coverage@0.9']:.3f} | {r['coverage@0.95']:.3f} "
            f"| {r['low_accuracy']:.3f} | {r['low_confidence']:.3f} | {r['high_accuracy']:.3f} | {r['high_confidence']:.3f} |"
        )
    lines.append("")
lines += ["## 除いた申告の内訳（付いている装置名 → 交差検証の判定、多い順）", ""]
for name, pairs in removed_pairs.items():
    lines.append(f"- {name}: " + "、".join(f"{pair} {count}" for pair, count in pairs))
lines.append("")
(RESULTS_DIR / "02_clean_train.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
