"""人の異議を、埋め込みが近い記事と並べて言語化し、境目のルールにする（00_knowledge/05_architecture/ の流れを試す）。

人の異議は、正解のラベルで代わりにする。対象は既定では DeepSeek（few-shot、labels_shots5.jsonl）が最後まで取り違えた
家電・IT・スマートフォンの 3 つの組み合わせ。--pairs で組み合わせを、--name で結果のファイル名を変えられる。
学習用（train）だけを使い、評価用（test）には触れない。

組み合わせ (A, B) ごとに:
1. 異議: 正解も DeepSeek の判定も A か B で、両者が食い違う記事
2. 言語化: 異議から 10 件を選び、それぞれに埋め込みが近い記事（DeepSeek と正解が一致した A と B を 2 件ずつ）を並べて、
   DeepSeek（deepseek-flash、思考あり）に A と B を分けるルールを書かせる
3. 確かめ: 言語化に使っていない記事（正解も判定も A か B）から 200 件を選び、ルールを足したプロンプトで判定し直す。
   正解率が 2 ポイント以上上がれば「定義違い」としてルールを採用し、そうでなければ「ブレ」として捨てる
結果は 03_results/04_livedoor/03_feedback_rules.md と .json（--name を付けるとその名前）に保存する。
03_feedback_rules.json で採用したルールは 18_label_livedoor.py --rules で使う。

環境変数 DEEPSEEK_API_KEY が必要。.env に置いてある場合は次のように実行する:
    uv run --env-file .env python 01_scripts/20_feedback_rules_livedoor.py
    uv run --env-file .env python 01_scripts/20_feedback_rules_livedoor.py \
        --pairs dokujo-tsushin:movie-enter,dokujo-tsushin:peachy --name 04_feedback_rules_dokujo
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from label_names import ja
from livedoor_llm import CRITERIA, DATA_DIR, ROOT, ask, build_prompt, classify, parse_label

RESULTS_DIR = ROOT / "03_results" / "04_livedoor"
PAIRS = [("kaden-channel", "it-life-hack"), ("kaden-channel", "smax"), ("it-life-hack", "smax")]
SHOTS = 5
N_DISPUTES = 10
N_NEIGHBORS = 2
N_CHECK = 200
MIN_GAIN = 0.02
EXCERPT_CHARS = 200

ARTICULATE_PROMPT = """ニュースサイトの記事を、カテゴリー「{a}」と「{b}」に分けています。
- {a}: {a_desc}
- {b}: {b_desc}

AI の判定に対して、人が「この記事のカテゴリーは違う」と異を唱えました。異議の出た記事と、内容が近い記事を並べます。
近い記事は、AI と人の判定が一致したものです。

{cases}

人の判定を説明できる、{a} と {b} を分けるルールを書いてください。
- ほかの記事にも当てはめられる一般的なルールにする（記事の固有名詞や文をそのまま引用しない）
- 1〜3 文で、記事の内容（題材・切り口・書き方）だけから判断できるようにする
- 異議どうし、または異議と近い記事のあいだで人の判定が矛盾していて、一貫したルールが書けない場合は consistent を false にする

答えは JSON で {{"rule": "<ルール>", "consistent": true または false, "reason": "<そう判断した理由（1〜2 文）>"}} の形だけを返してください。"""


def excerpt(record):
    return record["text"][:EXCERPT_CHARS].replace("\n", " ")


parser = argparse.ArgumentParser()
parser.add_argument("--pairs", default=",".join(f"{a}:{b}" for a, b in PAIRS))
parser.add_argument("--name", default="03_feedback_rules")
args = parser.parse_args()
pairs = [tuple(pair.split(":")) for pair in args.pairs.split(",")]
api_key = os.environ["DEEPSEEK_API_KEY"]
with (DATA_DIR / "train.jsonl").open(encoding="utf-8") as f:
    train = [json.loads(line) for line in f]
X = np.load(DATA_DIR / "train.npy")
with (DATA_DIR / f"labels_shots{SHOTS}.jsonl").open(encoding="utf-8") as f:
    llm = {r["id"]: r["label"] for r in map(json.loads, f)}
gold = np.array([r["category"] for r in train])
predicted = np.array([llm.get(r["id"]) or "" for r in train])

results = []
for a, b in pairs:
    rng = random.Random(0)
    in_pair = np.isin(gold, [a, b]) & np.isin(predicted, [a, b])
    disputes = np.flatnonzero(in_pair & (gold != predicted))
    agreed = {name: np.flatnonzero(in_pair & (gold == predicted) & (gold == name)) for name in (a, b)}
    chosen = rng.sample(list(disputes), N_DISPUTES)
    used = set(chosen)

    cases = []
    for number, i in enumerate(chosen, 1):
        lines = [f"### 異議 {number}: 人の判定 {gold[i]}（AI の判定 {predicted[i]}）", excerpt(train[i]), "", "内容が近い記事:"]
        for name in (a, b):
            candidates = agreed[name][np.argsort(-(X[agreed[name]] @ X[i]))]
            for j in [j for j in candidates if j not in used][:N_NEIGHBORS]:
                used.add(j)
                lines.append(f"- [{name}] {excerpt(train[j])}")
        cases.append("\n".join(lines))
    prompt = ARTICULATE_PROMPT.format(a=a, b=b, a_desc=CRITERIA[a], b_desc=CRITERIA[b], cases="\n\n".join(cases))
    response, seconds = ask([{"role": "user", "content": prompt}], api_key, thinking=True, max_tokens=32000)
    choice = response["choices"][0]
    if choice["finish_reason"] != "stop" or not choice["message"].get("content"):
        raise RuntimeError(f"ルールが返ってきませんでした: {choice['finish_reason']} {response['usage']}")
    answer = json.loads(choice["message"]["content"])
    print(f"{ja(a)} / {ja(b)}: {answer}")

    check = rng.sample([i for i in np.flatnonzero(in_pair) if i not in used], N_CHECK)
    system_prompt = build_prompt(SHOTS, [answer["rule"]])
    with ThreadPoolExecutor(max_workers=48) as pool:
        responses = list(pool.map(lambda i: classify(train[i]["text"], system_prompt, api_key)[0], check))
    relabeled = np.array([parse_label(r["choices"][0]["message"]["content"]) or "" for r in responses])
    before = predicted[check] == gold[check]
    after = relabeled == gold[check]
    gain = float(after.mean() - before.mean())
    results.append({
        "a": a,
        "b": b,
        "disputes": int(len(disputes)),
        "rule": answer["rule"],
        "consistent": bool(answer.get("consistent")),
        "reason": answer.get("reason", ""),
        "articulation_seconds": round(seconds, 1),
        "articulation_tokens": {"prompt": response["usage"]["prompt_tokens"], "completion": response["usage"]["completion_tokens"]},
        "check": {
            "n": N_CHECK,
            "disputes_in_check": int((gold[check] != predicted[check]).sum()),
            "accuracy_before": round(float(before.mean()), 3),
            "accuracy_after": round(float(after.mean()), 3),
            "fixed": int((~before & after).sum()),
            "broken": int((before & ~after).sum()),
            "tokens": {
                "prompt": sum(r["usage"]["prompt_tokens"] for r in responses),
                "cache_hit": sum(r["usage"].get("prompt_cache_hit_tokens", 0) for r in responses),
                "completion": sum(r["usage"]["completion_tokens"] for r in responses),
            },
        },
        "adopted": bool(answer.get("consistent")) and gain >= MIN_GAIN,
    })
    print(results[-1]["check"], "採用" if results[-1]["adopted"] else "不採用")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / f"{args.name}.json").write_text(
    json.dumps({"shots": SHOTS, "min_gain": MIN_GAIN, "pairs": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

lines = [
    "# 人の異議を言語化した境目のルール（livedoor ニュース）",
    "",
    "人の異議は正解のラベルで代わりにした。学習用の記事だけを使い、評価用には触れていない。",
    f"異議 {N_DISPUTES} 件と、それぞれに埋め込みが近い記事（各カテゴリー {N_NEIGHBORS} 件）を並べ、DeepSeek（deepseek-flash、思考あり）にルールを書かせた。",
    f"言語化に使っていない {N_CHECK} 件で、ルールを足す前（{SHOTS}-shot）と足した後の正解率を比べ、{MIN_GAIN * 100:.0f} ポイント以上上がったら採用した。",
    "",
    "| 組み合わせ | 異議の件数 | ルールが書けたか | 足す前 | 足した後 | 直った / 壊れた | 判断 |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for r in results:
    c = r["check"]
    verdict = "定義違い（採用）" if r["adopted"] else "ブレ（不採用）"
    lines.append(
        f"| {ja(r['a'])} / {ja(r['b'])} | {r['disputes']} | {'はい' if r['consistent'] else 'いいえ'} | "
        f"{c['accuracy_before']:.3f} | {c['accuracy_after']:.3f} | {c['fixed']} / {c['broken']} | {verdict} |"
    )
lines += ["", "## ルール", ""]
for r in results:
    lines += [f"### {ja(r['a'])} / {ja(r['b'])}", "", f"- ルール: {r['rule']}", f"- 理由: {r['reason']}", ""]
(RESULTS_DIR / f"{args.name}.md").write_text("\n".join(lines), encoding="utf-8")
