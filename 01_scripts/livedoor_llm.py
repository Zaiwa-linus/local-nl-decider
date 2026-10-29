"""livedoor ニュースのラベル付けで使う、DeepSeek へのプロンプトと問い合わせ。

18_label_livedoor.py と 20_feedback_rules_livedoor.py から `import livedoor_llm` で使う（01_scripts/ から実行すると読み込める）。
"""

import http.client
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "02_data" / "livedoor"
API_URL = "https://api.deepseek.com/chat/completions"
SHOT_CHARS = 200

CRITERIA = {
    "topic-news": "社会・事件・芸能・ネットで話題になったことなど、一般のニュース",
    "sports-watch": "スポーツ（野球、サッカー、格闘技など）の試合や選手の話題",
    "it-life-hack": "パソコン・ソフト・ネットサービス・周辺機器など、IT の製品や使い方",
    "kaden-channel": "家電・デジタル機器・ガジェットの製品や話題",
    "movie-enter": "映画（新作、俳優・監督、公開・試写会、興行）",
    "dokujo-tsushin": "独身で働く女性の恋愛・結婚・仕事・生活についてのコラム",
    "smax": "スマートフォン・携帯電話・タブレット、そのアプリや通信会社",
    "livedoor-homme": "男性向けの仕事・キャリア・お金・ライフスタイル",
    "peachy": "女性向けの美容・ファッション・恋愛・グルメ・エンタメ・ライフスタイル",
}


def build_prompt(shots=0, rules=()):
    """カテゴリーの説明に、検証用からの例（shots 件ずつ）と、境目のルールを足したシステムプロンプトを作る。"""
    prompt = (
        "これからニュースサイトの記事（タイトルと本文の先頭）を 1 件渡します。次のカテゴリーからいちばん近いものを 1 つ選んでください。\n\n"
        + "\n".join(f"- {name}: {description}" for name, description in CRITERIA.items())
    )
    if shots:
        with (DATA_DIR / "validation.jsonl").open(encoding="utf-8") as f:
            validation = [json.loads(line) for line in f]
        rng = random.Random(0)
        used = []
        prompt += "\n\nカテゴリーは記事が載った連載で決まっています。各カテゴリーの記事の例（先頭だけ）を示します。\n"
        for name in CRITERIA:
            for article in rng.sample([a for a in validation if a["category"] == name], shots):
                used.append(article["id"])
                example = article["text"][:SHOT_CHARS].replace("\n", " ")
                prompt += f"\n[{name}] {example}"
        (DATA_DIR / f"shots{shots}.json").write_text(json.dumps(used) + "\n", encoding="utf-8")
    if rules:
        prompt += "\n\n迷いやすいカテゴリーの境目のルール:\n" + "\n".join(f"- {rule}" for rule in rules)
    return prompt + '\n\n答えは JSON で {"category": "<カテゴリー名>"} の形だけを返してください。'


def ask(messages, api_key, model="deepseek-flash", thinking=False, max_tokens=30):
    """DeepSeek に問い合わせて、(応答の JSON, かかった秒数) を返す。JSON で答えさせる。"""
    body = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "thinking": {"type": "enabled" if thinking else "disabled"},
        "max_tokens": max_tokens,
    }
    if not thinking:
        body["temperature"] = 0
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            start = time.time()
            with urllib.request.urlopen(request, timeout=300 if thinking else 60) as response:
                result = json.load(response)
            return result, time.time() - start
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500 and error.code != 429:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("DeepSeek への問い合わせに 5 回失敗しました")


def classify(text, system_prompt, api_key, model="deepseek-flash"):
    """記事 1 件を判定して、(応答の JSON, かかった秒数) を返す。"""
    return ask([{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], api_key, model)


def parse_label(content):
    try:
        label = json.loads(content).get("category")
    except (json.JSONDecodeError, AttributeError):
        return None
    return label if label in CRITERIA else None
