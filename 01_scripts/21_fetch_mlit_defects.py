"""国土交通省の「自動車不具合情報ホットライン」から、ユーザーの申告をまとめた不具合情報（自動車）を取得する。

- 不具合情報検索（https://renrakuda.mlit.go.jp/renrakuda/opn.html）の画面が使う検索 API を、年ごとに --limit 件ずつたどる
- 本文は申告そのものではなく、国交省が要約した「申告内容の要約」。装置名（15 種類）が国交省によって付いている
- 都道府県と型式は使わないので保存しない
- サーバーに負担をかけないよう、リクエストの間に --interval 秒あける。取り終えた年は飛ばすので、途中で止まっても続きから再開できる

保存先:
- 02_data/mlit/defects_<年>.jsonl（control_no, date, route, device, maker, car, registered, mileage, emerged, text）

出典: 国土交通省「自動車不具合情報ホットライン」の不具合情報（https://renrakuda.mlit.go.jp/ ）。公共データ利用規約（第1.0版）（PDL1.0）

このリポジトリの環境で実行する:
    uv run python 01_scripts/21_fetch_mlit_defects.py --years 2024 2025
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "02_data" / "mlit"
API_URL = "https://renrakuda.mlit.go.jp/mt/mt-estraier.cgi"
HEADERS = {"User-Agent": "Mozilla/5.0 (local-nl-decider; research use)", "Accept": "application/json"}
PREFIX = "release_data_car_mlit_"
FIELDS = {
    "control_no": "control_no",
    "date": "reception_date",
    "route": "declaration_route",
    "device": "defective_device",
    "maker": "car_name",
    "car": "car_common_name",
    "registered": "initial_registration_date",
    "mileage": "total_mileage",
    "emerged": "emergence_time",
    "text": "report_content_summary",
}


def search(year, offset, limit):
    params = {
        "blog_id": 4,
        "limit": limit,
        "offset": offset,
        "order_by": PREFIX + "reception_date",
        "order_condition": "STRA",
        PREFIX + "reception_date": f"{year}-01-01 {year}-12-31",
        "class": "releasedatacar",
    }
    request = urllib.request.Request(f"{API_URL}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8")
    # 画面の JS と同じく、末尾の余分なカンマを取り除いてから読む
    return json.loads(re.sub(r",\s*(\]\s*\}\s*)$", r"\1", text))["data"] or []


parser = argparse.ArgumentParser()
parser.add_argument("--years", type=int, nargs="+", required=True, help="取得する受付年")
parser.add_argument("--limit", type=int, default=200, help="1 回のリクエストで取る件数")
parser.add_argument("--interval", type=float, default=1.0, help="リクエストの間隔（秒）")
args = parser.parse_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
for year in args.years:
    path = OUT_DIR / f"defects_{year}.jsonl"
    if path.exists():
        print(f"{year}: 取得済みなので飛ばす")
        continue

    records = []
    total = None
    while total is None or len(records) < total:
        rows = search(year, len(records), args.limit)
        time.sleep(args.interval)
        if not rows:
            break
        total = int(rows[0]["count"])
        records += [{key: row.get(PREFIX + name, "") for key, name in FIELDS.items()} for row in rows]
        print(f"{year}: {len(records)} / {total}", flush=True)

    # 同じ受付日の中は並びが決まらず、ページの境目で重なることがあるので、管理番号で重複を除く
    records = list({record["control_no"]: record for record in records}.values())
    if total is not None and len(records) < total:
        print(f"{year}: {total} 件のうち {len(records)} 件しか取れなかった")

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.rename(path)
