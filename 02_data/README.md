# 02_data

スクリプトで取得したデータと埋め込みを置くディレクトリ。この README 以外は git に入れない。
中身は [01_scripts/](../01_scripts/) のスクリプトで作り直せる。

| パス | 内容 | 作るスクリプト |
| --- | --- | --- |
| `massive/<split>.jsonl` | MASSIVE（日本語）のテキストとラベル（シナリオ、意図） | `01_fetch_massive.py` |
| `massive/<split>.npy` | 768 次元の埋め込み（行の順番は JSONL と同じ） | `02_embed.py` |
| `deepseek/massive_scenario_labels.jsonl` | DeepSeek が train の一部に付けたシナリオのラベル | `07_llm_label.py` |
| `pubcom/cases.jsonl`, `pubcom/pdf/` | パブコメの結果公示案件の情報と、結果の PDF | `09_fetch_pubcom.py` |
| `pubcom/opinions.jsonl` | PDF から取り出した意見の要旨 | `10_extract_opinions.py` |
| `pubcom/categories.json` | DeepSeek が設計したカテゴリー | `11_design_categories.py` |
| `pubcom/labels.jsonl` | DeepSeek が付けたラベル | `12_label_opinions.py` |
| `pubcom/labels_rerun1.jsonl`, `pubcom/labels_rerun2.jsonl`, `pubcom/labels_pro.jsonl` | 再現性の確認用に 1,000 件へ付け直したラベル | `12_label_opinions.py`（`--sample 1000 --out ...`） |
| `pubcom/categories_topic.json`, `pubcom/labels_topic.jsonl`, `pubcom/labels_topic_rerun1.jsonl` | 話題で分けたカテゴリーと、そのラベル（rerun1 は再現性の確認用の 1,000 件） | `11_design_categories.py` / `12_label_opinions.py`（`--scheme topic`） |
| `pubcom/opinions.npy` | 意見の埋め込み（行の順番は opinions.jsonl と同じ） | `13_embed_opinions.py` |
| `livedoor/ldcc-20140209.tar.gz` | livedoor ニュースコーパス（配布されたまま） | `16_fetch_livedoor.py` |
| `livedoor/<split>.jsonl` | 連載名と【】で囲んだコーナー名を取り除いた記事（タイトル＋本文の先頭 500 文字）と正解のカテゴリー | `16_fetch_livedoor.py` |
| `livedoor/<split>.npy` | 記事の埋め込み（行の順番は JSONL と同じ） | `17_embed_livedoor.py` |
| `livedoor/labels.jsonl` | DeepSeek が学習用と評価用に付けたラベル | `18_label_livedoor.py` |
| `livedoor/labels_shots<N>.jsonl` | DeepSeek にカテゴリーごとに N 件の例を見せて付けたラベル | `18_label_livedoor.py --shots N` |
| `livedoor/shots<N>.json` | 例として見せた検証用の記事の ID | `18_label_livedoor.py --shots N` |
| `livedoor/with_corner/` | コーナー名を残していたときのデータ・埋め込み・ラベル（比較用） | `16`〜`18`（変更前） |
| `models/massive_<label>.npz` | ロジスティック回帰の重み（coef, intercept, classes） | `05_train_final.py` |
