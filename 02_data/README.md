# 02_data

スクリプトで取得したデータと埋め込みを置くディレクトリ。この README 以外は git に入れない。
中身は [01_scripts/](../01_scripts/) のスクリプトで作り直せる。

| パス | 内容 | 作るスクリプト |
| --- | --- | --- |
| `massive/<split>.jsonl` | MASSIVE（日本語）のテキストとラベル（シナリオ、意図） | `01_fetch_massive.py` |
| `massive/<split>.npy` | 768 次元の埋め込み（行の順番は JSONL と同じ） | `02_embed.py` |
| `deepseek/massive_scenario_labels.jsonl` | DeepSeek が train の一部に付けたシナリオのラベル | `07_llm_label.py` |
| `models/massive_<label>.npz` | ロジスティック回帰の重み（coef, intercept, classes） | `05_train_final.py` |
