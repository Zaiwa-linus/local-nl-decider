"""DeepSeek のラベルの再現性を確かめる。

12_label_opinions.py で、同じ 1,000 件（--sample 1000）に付け直したラベルを、最初のラベル（labels.jsonl）と比べる。
- labels_rerun1.jsonl / labels_rerun2.jsonl: 同じ設定（deepseek-flash、temperature 0）でもう 2 回
- labels_pro.jsonl: 上位モデル（deepseek-v4-pro、思考なし）で 1 回
あわせて、評価用（14_train_eval_pubcom.py と同じ分け方）に入った意見で、判定機がラベルの揺れていない意見をどれだけ当てるかも出す。
結果は 03_results/02_pubcom_type/03_label_stability.md と 03_label_stability.json に保存する。

このリポジトリの環境で実行する:
    uv run python 01_scripts/15_label_stability.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

from label_names import ja

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "pubcom"
RESULTS_DIR = ROOT / "03_results" / "02_pubcom_type"
RUNS = {"flash_1": "labels.jsonl", "flash_2": "labels_rerun1.jsonl", "flash_3": "labels_rerun2.jsonl", "pro": "labels_pro.jsonl"}


def load_labels(name):
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return {r["opinion_id"]: r["label"] for r in map(json.loads, f)}


runs = {k: load_labels(v) for k, v in RUNS.items()}
ids = sorted(set(runs["flash_2"]) & set(runs["flash_3"]) & set(runs["pro"]))
ids = [i for i in ids if all(runs[k].get(i) for k in runs)]
labels = {k: np.array([runs[k][i] for i in ids]) for k in runs}
print(f"比べる意見 {len(ids)} 件")

pairs = [("flash_1", "flash_2"), ("flash_1", "flash_3"), ("flash_2", "flash_3"), ("flash_1", "pro"), ("flash_2", "pro"), ("flash_3", "pro")]
agreement = [
    {"a": a, "b": b, "agreement": round(float((labels[a] == labels[b]).mean()), 4), "kappa": round(float(cohen_kappa_score(labels[a], labels[b])), 4)}
    for a, b in pairs
]
flash_stable = (labels["flash_1"] == labels["flash_2"]) & (labels["flash_1"] == labels["flash_3"])
all_stable = flash_stable & (labels["flash_1"] == labels["pro"])

per_class = []
for name, count in Counter(labels["flash_1"]).most_common():
    mask = labels["flash_1"] == name
    per_class.append({
        "category": name,
        "count": int(count),
        "flash_stable": round(float(flash_stable[mask].mean()), 3),
        "pro_agrees": round(float((labels["pro"][mask] == name).mean()), 3),
    })
changes = Counter((a, b) for a, b in zip(labels["flash_1"], labels["pro"]) if a != b).most_common(10)

# 14_train_eval_pubcom.py と同じ分け方・同じ C で判定機を作り、評価用に入った意見で比べる
with (DATA_DIR / "opinions.jsonl").open(encoding="utf-8") as f:
    opinions = [json.loads(line) for line in f]
embeddings = np.load(DATA_DIR / "opinions.npy")
keep = [i for i, o in enumerate(opinions) if runs["flash_1"].get(o["opinion_id"])]
X = embeddings[keep]
y = np.array([runs["flash_1"][opinions[i]["opinion_id"]] for i in keep])
groups = np.array([opinions[i]["case_id"] for i in keep])
opinion_ids = np.array([opinions[i]["opinion_id"] for i in keep])
C = json.loads((RESULTS_DIR / "02_classifier.json").read_text(encoding="utf-8"))["C"]
train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
classifier = LogisticRegression(max_iter=3000, C=C).fit(X[train_idx], y[train_idx])
predicted = dict(zip(opinion_ids[test_idx], classifier.predict(X[test_idx])))
position = {i: n for n, i in enumerate(ids)}
in_test = [i for i in ids if i in predicted]
pred = np.array([predicted[i] for i in in_test])
rows = np.array([position[i] for i in in_test])
classifier_check = {
    "test_opinions": len(in_test),
    "vs_flash_1": round(float((pred == labels["flash_1"][rows]).mean()), 4),
    "vs_pro": round(float((pred == labels["pro"][rows]).mean()), 4),
    "flash_stable_count": int(flash_stable[rows].sum()),
    "vs_flash_1_when_flash_stable": round(float((pred == labels["flash_1"][rows])[flash_stable[rows]].mean()), 4),
    "all_stable_count": int(all_stable[rows].sum()),
    "vs_flash_1_when_all_stable": round(float((pred == labels["flash_1"][rows])[all_stable[rows]].mean()), 4),
}

summary = {
    "opinions": len(ids),
    "agreement": agreement,
    "flash_all_three_agree": round(float(flash_stable.mean()), 4),
    "flash_and_pro_all_agree": round(float(all_stable.mean()), 4),
    "per_class": per_class,
    "flash_1_to_pro_changes": [{"flash": a, "pro": b, "count": c} for (a, b), c in changes],
    "classifier": classifier_check,
}
print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, ensure_ascii=False, indent=1))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "03_label_stability.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# DeepSeek のラベルの再現性（パブコメの意見）",
    "",
    f"無作為に選んだ意見 {len(ids):,} 件に、同じ設定（deepseek-flash、思考なし、temperature 0）で 3 回、上位モデル（deepseek-v4-pro、思考なし）で 1 回ラベルを付けて比べた。",
    "flash_1 は判定機の学習に使ったラベル。",
    "",
    "## 2 回のラベルの一致率",
    "",
    "| A | B | 一致率 | κ 係数 |",
    "| --- | --- | --- | --- |",
]
lines += [f"| {r['a']} | {r['b']} | {r['agreement']:.3f} | {r['kappa']:.3f} |" for r in agreement]
lines += [
    "",
    f"- flash の 3 回がすべて同じ: {summary['flash_all_three_agree']:.3f}",
    f"- flash の 3 回と pro がすべて同じ: {summary['flash_and_pro_all_agree']:.3f}",
    "",
    "## カテゴリーごと（flash_1 のラベルで分けた）",
    "",
    "| カテゴリー | 件数 | flash の 3 回が同じ | pro も同じカテゴリー |",
    "| --- | --- | --- | --- |",
]
lines += [f"| {ja(r['category'])} | {r['count']} | {r['flash_stable']:.3f} | {r['pro_agrees']:.3f} |" for r in per_class]
lines += ["", "## flash_1 と pro で食い違った組み合わせ（上位 10）", "", "| flash_1 | pro | 件数 |", "| --- | --- | --- |"]
lines += [f"| {ja(a)} | {ja(b)} | {c} |" for (a, b), c in changes]
c = classifier_check
lines += [
    "",
    "## 判定機との比較（評価用に入った意見）",
    "",
    f"14_train_eval_pubcom.py と同じ分け方・C={C} で学習した判定機。1,000 件のうち評価用に入った {c['test_opinions']} 件で比べた。",
    "",
    "| 比べる相手 | 件数 | 判定機の一致率 |",
    "| --- | --- | --- |",
    f"| flash_1（学習に使ったラベル） | {c['test_opinions']} | {c['vs_flash_1']:.3f} |",
    f"| pro | {c['test_opinions']} | {c['vs_pro']:.3f} |",
    f"| flash_1（flash の 3 回が同じ意見だけ） | {c['flash_stable_count']} | {c['vs_flash_1_when_flash_stable']:.3f} |",
    f"| flash_1（flash の 3 回と pro がすべて同じ意見だけ） | {c['all_stable_count']} | {c['vs_flash_1_when_all_stable']:.3f} |",
]
(RESULTS_DIR / "03_label_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
