# 01_scripts

スクリプトを置くディレクトリ。実行方法は各スクリプトの先頭に書いてある。

## Phase 1（MASSIVE）

番号の順に実行する。

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [01_fetch_massive.py](01_fetch_massive.py) | MASSIVE（日本語）を取得して `02_data/massive/` に保存する | このリポジトリの環境 |
| [02_embed.py](02_embed.py) | テキストを EmbeddingGemma 2 で埋め込んで保存する | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [03_train_eval.py](03_train_eval.py) | 判定機を学習・評価して `03_results/` に保存する | このリポジトリの環境 |
| [04_tune.py](04_tune.py) | 判定機のパラメーターを validation で選び、間違えやすい組み合わせも出す | このリポジトリの環境 |
| [05_train_final.py](05_train_final.py) | 判定に使うロジスティック回帰を学習し、確信度の補正（温度）としきい値を validation で決めて `02_data/models/` に保存する | このリポジトリの環境 |
| [06_predict.py](06_predict.py) | 新しいテキストを判定して、カテゴリーと確信度を返す。しきい値を下回ったものは「要確認」と出す | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |

## サンプル

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [embed_sample.py](embed_sample.py) | EmbeddingGemma 2 で口コミを埋め込むサンプル | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [jev_sample.py](jev_sample.py) | jev で口コミのカテゴリーを判定するサンプル（API キーが必要） | このリポジトリの環境 |
