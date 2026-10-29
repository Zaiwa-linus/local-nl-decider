"""e-Gov パブリック・コメントの結果公示案件から、各省庁がまとめた「意見募集の結果」の PDF を取得する。

- 結果公示案件の一覧を新しい順に --pages ページ分（1 ページ 20 件）たどる
- 提出意見数が 1 件以上の案件について、結果の PDF（名前に「結果」「考え方」「概要」を含むもの）を保存する
- 市民が書いた意見の原文（「提出された意見」など）は著作権が提出者にあるので取らない。省庁がまとめた文書だけを使う
- サーバーに負担をかけないよう、リクエストの間に --interval 秒あける。途中で止まっても続きから再開できる

保存先:
- 02_data/pubcom/cases.jsonl（案件ごとに id, title, ministry, result_date, opinions, files）
- 02_data/pubcom/pdf/<案件番号>_<seqNo>.pdf

出典: e-Gov パブリック・コメント（https://public-comment.e-gov.go.jp/）。公共データ利用規約（第1.0版）（PDL1.0）

このリポジトリの環境で実行する:
    uv run python 01_scripts/09_fetch_pubcom.py --pages 30
"""

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "02_data" / "pubcom"
BASE_URL = "https://public-comment.e-gov.go.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (local-nl-decider; research use)"}
RESULT_WORDS = ["結果", "考え方", "概要"]
EXCLUDE_WORDS = ["提出された意見", "提出意見", "意見全文", "意見の全文"]


def fetch(url, data=None):
    body = urllib.parse.urlencode(data).encode() if data else None
    request = urllib.request.Request(url, data=body, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def text_of(fragment):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def field(page, name):
    """詳細ページの表から、見出しが name の行の値を取り出す。"""
    match = re.search(rf"<th>\s*{name}.*?</th>\s*<td>(.*?)</td>", page, flags=re.S)
    return text_of(match.group(1)) if match else ""


parser = argparse.ArgumentParser()
parser.add_argument("--pages", type=int, default=30, help="たどる一覧のページ数（1 ページ 20 件）")
parser.add_argument("--interval", type=float, default=1.0, help="リクエストの間隔（秒）")
args = parser.parse_args()

(OUT_DIR / "pdf").mkdir(parents=True, exist_ok=True)
cases_path = OUT_DIR / "cases.jsonl"
done = set()
if cases_path.exists():
    with cases_path.open(encoding="utf-8") as f:
        done = {json.loads(line)["id"] for line in f}

with cases_path.open("a", encoding="utf-8") as out:
    for page_number in range(1, args.pages + 1):
        listing = fetch(
            f"{BASE_URL}/pcm/list",
            {"CLASSNAME": "PCMMSTLIST", "Page": page_number, "Mode": 1, "Type": 0, "bMode": 1, "sortItem": 1, "sortButton": 1},
        ).decode("utf-8")
        time.sleep(args.interval)
        case_ids = list(dict.fromkeys(re.findall(r"PCM1040&(?:amp;)?id=(\d+)", listing)))

        for case_id in case_ids:
            if case_id in done:
                continue
            page = fetch(f"{BASE_URL}/servlet/Public?CLASSNAME=PCM1040&id={case_id}&Mode=1").decode("utf-8")
            time.sleep(args.interval)

            opinions = re.sub(r"\D", "", field(page, "提出意見数")) or "0"
            case = {
                "id": case_id,
                "title": field(page, "定めようとする命令などの題名") or field(page, "命令等の題名"),
                "ministry": field(page, "問合せ先"),
                "result_date": field(page, "結果の公示日"),
                "opinions": int(opinions),
                "files": [],
            }

            result_cell = re.search(r"<th>\s*結果概要.*?</th>\s*<td>(.*?)</td>", page, flags=re.S)
            if case["opinions"] > 0 and result_cell:
                for seq_no, name, file_type in re.findall(
                    r'href="/pcm/download\?seqNo=(\d+)"[^>]*>([^<]*)</a>\s*<span[^>]*>([^<]*)</span>', result_cell.group(1)
                ):
                    name = html.unescape(name).strip()
                    if file_type.strip() != "PDF" or not any(w in name for w in RESULT_WORDS) or any(w in name for w in EXCLUDE_WORDS):
                        continue
                    pdf_path = OUT_DIR / "pdf" / f"{case_id}_{seq_no}.pdf"
                    if not pdf_path.exists():
                        pdf_path.write_bytes(fetch(f"{BASE_URL}/pcm/download?seqNo={seq_no}"))
                        time.sleep(args.interval)
                    case["files"].append({"seq_no": seq_no, "name": name, "path": str(pdf_path.relative_to(ROOT))})

            out.write(json.dumps(case, ensure_ascii=False) + "\n")
            out.flush()
            done.add(case_id)
        print(f"ページ {page_number}/{args.pages} 完了（案件 {len(done)} 件）")
