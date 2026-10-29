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
| [07_llm_label.py](07_llm_label.py) | train の一部に DeepSeek でラベルを付ける（蒸留用。`.env` の `DEEPSEEK_API_KEY` を使う） | このリポジトリの環境（`uv run --env-file .env`） |
| [08_distill_eval.py](08_distill_eval.py) | DeepSeek のラベルで学習した判定機と、正解のラベルで学習した判定機を比べる。`--shots N` で few-shot のラベル、`--rules` で境目のルールを足したラベルを使う | このリポジトリの環境 |
| [06_predict.py](06_predict.py) | 新しいテキストを判定して、カテゴリーと確信度を返す。しきい値を下回ったものは「要確認」と出す | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |

## Phase 2（パブコメの意見、正解のない実データ）

番号の順に実行する。DeepSeek を使うものは `.env` の `DEEPSEEK_API_KEY` を使う（`uv run --env-file .env`）。
11・12・14 は `--scheme topic` を付けると、「意見の種類」ではなく「話題」で分けたカテゴリーで同じことをする（結果は `_topic` の付いたファイルに保存する）。

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [09_fetch_pubcom.py](09_fetch_pubcom.py) | e-Gov から、パブコメの結果の PDF を取得する（1 秒間隔） | このリポジトリの環境 |
| [10_extract_opinions.py](10_extract_opinions.py) | PDF から意見の要旨を原文のまま取り出す（DeepSeek。原文と一致しないものは捨てる） | このリポジトリの環境 |
| [11_design_categories.py](11_design_categories.py) | 意見のカテゴリーを DeepSeek に設計させる | このリポジトリの環境 |
| [12_label_opinions.py](12_label_opinions.py) | 意見に DeepSeek でラベルを付ける（これを正解にする） | このリポジトリの環境 |
| [13_embed_opinions.py](13_embed_opinions.py) | 意見を EmbeddingGemma 2 で埋め込む | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [14_train_eval_pubcom.py](14_train_eval_pubcom.py) | 判定機を学習し、DeepSeek のラベルを正解として評価する | このリポジトリの環境 |
| [15_label_stability.py](15_label_stability.py) | DeepSeek のラベルの再現性を確かめる（12 で `--sample 1000 --out ...` を付けて付け直したラベルと比べる） | このリポジトリの環境 |

## Phase 2（livedoor ニュース、正解のある実データ）

番号の順に実行する。蒸留した判定機と、正解で学習した判定機を比べる。

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [16_fetch_livedoor.py](16_fetch_livedoor.py) | livedoor ニュースコーパスを取得し、連載名を取り除いて学習・検証・評価用に分ける | このリポジトリの環境 |
| [17_embed_livedoor.py](17_embed_livedoor.py) | 記事を EmbeddingGemma 2 で埋め込む | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [18_label_livedoor.py](18_label_livedoor.py) | 学習用と評価用の記事に DeepSeek でラベルを付ける。`--shots N` で、検証用からカテゴリーごとに N 件の例を見せる（few-shot）。`--rules NAME` で採用した境目のルールを足し、`--only A,B` で判定が A か B の記事だけ付け直す | このリポジトリの環境（`uv run --env-file .env`） |
| [19_distill_eval_livedoor.py](19_distill_eval_livedoor.py) | DeepSeek のラベルで学習した判定機と、正解のラベルで学習した判定機を比べる | このリポジトリの環境 |
| [20_feedback_rules_livedoor.py](20_feedback_rules_livedoor.py) | 人の異議を、埋め込みが近い記事と並べて DeepSeek に言語化させ、境目のルールにする。言語化に使っていない記事で効くか確かめて、採用を決める | このリポジトリの環境（`uv run --env-file .env`） |

## Phase 2（国交省の自動車不具合情報、正解のある実データ）

番号の順に実行する。

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [21_fetch_mlit_defects.py](21_fetch_mlit_defects.py) | 自動車不具合情報ホットラインから、不具合情報（申告内容の要約と装置名）を年ごとに取得する（1 秒間隔） | このリポジトリの環境 |
| [22_split_mlit.py](22_split_mlit.py) | 文と装置名を正規化し、【】の印を取り除いて、学習・検証・評価用に分ける | このリポジトリの環境 |
| [23_embed_mlit.py](23_embed_mlit.py) | 申告内容の要約を EmbeddingGemma 2 で埋め込む | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [24_baseline_mlit.py](24_baseline_mlit.py) | 正解の装置名をそのまま学習した判定機の精度を測る（蒸留と比べるための基準） | このリポジトリの環境 |
| [25_clean_train_mlit.py](25_clean_train_mlit.py) | 教師データから、付け間違いの疑いや境目の申告を埋め込みで見つけて除き、学習し直して比べる | このリポジトリの環境 |
| [26_count_mlit.py](26_count_mlit.py) | 装置名ごとの件数を、1 つに決めて数える場合と、確率で按分する場合とで比べる | このリポジトリの環境 |

## 共通

| スクリプト | 内容 |
| --- | --- |
| [label_names.py](label_names.py) | ラベルの ID を、結果の Markdown に書く日本語の名前に変える対応表 |
| [livedoor_llm.py](livedoor_llm.py) | livedoor のラベル付けで使う、DeepSeek へのプロンプト（説明・例・境目のルール）と問い合わせ。18 と 20 から使う |

## サンプル

| スクリプト | 内容 | 実行する環境 |
| --- | --- | --- |
| [embed_sample.py](embed_sample.py) | EmbeddingGemma 2 で口コミを埋め込むサンプル | 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`） |
| [jev_sample.py](jev_sample.py) | jev で口コミのカテゴリーを判定するサンプル（API キーが必要） | このリポジトリの環境 |
