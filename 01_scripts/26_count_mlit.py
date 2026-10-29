"""国交省の自動車不具合情報で、装置名ごとの件数を数える方法を比べる。

- 1 つに決める: 各申告を、いちばん確率の高い装置名 1 件として数える
- 確率で按分: 各申告の確率を、装置名ごとに足し合わせる（「エンジン 0.45 / 動力伝達 0.40」なら 0.45 件と 0.40 件）
- 按分（補正後）: 確信度を温度で補正してから按分する。温度は検証用（validation）の対数尤度が最大になるように決める
- 確信度 0.8 未満だけ按分: 補正後の確信度が 0.8 以上なら 1 つに決め、それ未満は按分する
評価用（test）の正解の件数と比べ、ずれの合計（装置名ごとの差の絶対値の和）を出す。
判定機はロジスティック回帰（C=30、24_baseline_mlit.py で選んだ値）で、学習用（train）の正解で学習する。
結果は 03_results/05_mlit/03_count.md と 03_count.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/26_count_mlit.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "mlit"
RESULTS_DIR = ROOT / "03_results" / "05_mlit"
C = 30
MIX_THRESHOLD = 0.8


def load(split):
    with (DATA_DIR / f"{split}.jsonl").open(encoding="utf-8") as f:
        labels = np.array([json.loads(line)["category"] for line in f])
    return labels, np.load(DATA_DIR / f"{split}.npy")


def softmax(logits, temperature):
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


y_train, X_train = load("train")
y_val, X_val = load("validation")
y_test, X_test = load("test")
classifier = LogisticRegression(max_iter=3000, C=C).fit(X_train, y_train)
classes = classifier.classes_

val_logits = classifier.decision_function(X_val)
val_idx = np.searchsorted(classes, y_val)
temperature = minimize_scalar(
    lambda t: -np.log(softmax(val_logits, t)[np.arange(len(y_val)), val_idx] + 1e-12).mean(),
    bounds=(0.05, 10),
    method="bounded",
).x

test_logits = classifier.decision_function(X_test)
proba = softmax(test_logits, 1.0)
calibrated = softmax(test_logits, temperature)
confident = calibrated.max(axis=1) >= MIX_THRESHOLD


def argmax_counts(p):
    return np.bincount(p.argmax(axis=1), minlength=len(classes)).astype(float)


counts = {
    "正解": np.array([(y_test == c).sum() for c in classes], dtype=float),
    "1 つに決める": argmax_counts(proba),
    "確率で按分": proba.sum(axis=0),
    "按分（補正後）": calibrated.sum(axis=0),
    f"確信度 {MIX_THRESHOLD} 未満だけ按分": argmax_counts(calibrated[confident]) + calibrated[~confident].sum(axis=0),
}
truth = counts["正解"]
errors = {name: float(np.abs(v - truth).sum()) for name, v in counts.items() if name != "正解"}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary = {
    "test": len(y_test),
    "temperature": float(temperature),
    "confident_share": float(confident.mean()),
    "counts": {name: dict(zip(classes, v.tolist())) for name, v in counts.items()},
    "total_abs_error": errors,
}
(RESULTS_DIR / "03_count.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

names = list(counts)
lines = [
    "# 国交省の自動車不具合情報: 装置名ごとの件数の数え方",
    "",
    f"- 評価用 {len(y_test):,} 件の装置名ごとの件数を、判定機（学習用の正解で学習したロジスティック回帰）の出力から数え、正解の件数と比べた",
    f"- 温度（検証用で決めた）: {temperature:.2f}。補正後の確信度が {MIX_THRESHOLD} 以上の申告は {confident.mean():.1%}",
    "",
    "## ずれの合計（装置名ごとの差の絶対値の和）",
    "",
    "| 数え方 | ずれの合計 | 全件に対する割合 |",
    "| --- | --- | --- |",
    *[f"| {name} | {err:.0f} | {err / len(y_test):.1%} |" for name, err in errors.items()],
    "",
    "## 装置名ごとの件数",
    "",
    "| 装置名 | " + " | ".join(names) + " |",
    "| --- | " + " | ".join("---" for _ in names) + " |",
    *[
        f"| {c} | " + " | ".join(f"{counts[name][i]:.0f}" for name in names) + " |"
        for i, c in sorted(enumerate(classes), key=lambda item: -truth[item[0]])
    ],
    "",
]
(RESULTS_DIR / "03_count.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
