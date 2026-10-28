"""jev の choice 型の質問で、口コミのカテゴリーを判定するサンプル。

環境変数 TYPESAFE_API_KEY に API キーを入れてから実行する:
    uv run python 01_scripts/jev_sample.py
"""

import json
import os
import urllib.request

request_body = {
    "model": "jev-latest",
    "state": "注文から3日で届いたが、箱がつぶれていた。",
    "questions": {
        "category": {
            "type": "choice",
            "instructions": "この口コミは主に何について書かれていますか？",
            "criteria": {
                "delivery": "配送の速さ、梱包、届いたときの状態",
                "quality": "商品そのものの品質や性能",
                "price": "価格やコスパ",
                "support": "問い合わせ対応や返品などのサポート",
            },
        }
    },
}

request = urllib.request.Request(
    "https://api.typesafe.ai/v1/systemone",
    data=json.dumps(request_body).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request) as response:
    result = json.load(response)

# answers.category に choice / confidence / probabilities が入る
print(json.dumps(result, ensure_ascii=False, indent=2))
