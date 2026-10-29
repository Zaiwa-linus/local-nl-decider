# 03_results

評価結果。各ファイルは [01_scripts/](../01_scripts/) のスクリプトが書き出す（`.md` は読む用、`.json` は数値の元データ）。

## [01_massive/](01_massive/)（Phase 1: MASSIVE、正解のラベルあり）

| ファイル | 内容 | 作るスクリプト |
| --- | --- | --- |
| [01_baseline.md](01_massive/01_baseline.md) | 判定機（ロジスティック回帰・k-NN）と次元数の比較 | `03_train_eval.py` |
| [02_tuned.md](01_massive/02_tuned.md) | パラメーター調整後の比較と、間違えやすい組み合わせ | `04_tune.py` |
| [03_confidence.md](01_massive/03_confidence.md) | 確信度の補正（温度）と、目標の正解率ごとのしきい値 | `05_train_final.py` |
| [04_distill.md](01_massive/04_distill.md) | DeepSeek のラベルで学習した場合と、正解のラベルで学習した場合の比較 | `08_distill_eval.py` |

## [02_pubcom_type/](02_pubcom_type/)（Phase 2: パブコメの意見を「意見の種類」で分けた場合）

| ファイル | 内容 | 作るスクリプト |
| --- | --- | --- |
| [01_categories.md](02_pubcom_type/01_categories.md) | DeepSeek が設計したカテゴリー | `11_design_categories.py` |
| [02_classifier.md](02_pubcom_type/02_classifier.md) | DeepSeek のラベルを正解とした判定機の精度 | `14_train_eval_pubcom.py` |
| [03_label_stability.md](02_pubcom_type/03_label_stability.md) | DeepSeek のラベルの再現性 | `15_label_stability.py` |

## [03_pubcom_topic/](03_pubcom_topic/)（Phase 2: パブコメの意見を「話題」で分けた場合）

| ファイル | 内容 | 作るスクリプト |
| --- | --- | --- |
| [01_categories.md](03_pubcom_topic/01_categories.md) | DeepSeek が設計したカテゴリー | `11_design_categories.py --scheme topic` |
| [02_classifier.md](03_pubcom_topic/02_classifier.md) | DeepSeek のラベルを正解とした判定機の精度 | `14_train_eval_pubcom.py --scheme topic` |

## [04_livedoor/](04_livedoor/)（Phase 2: livedoor ニュース、正解のラベルあり）

| ファイル | 内容 | 作るスクリプト |
| --- | --- | --- |
| [01_distill.md](04_livedoor/01_distill.md) | DeepSeek の正解率と、DeepSeek のラベルで学習した場合・正解のラベルで学習した場合の比較 | `19_distill_eval_livedoor.py` |
| [02_distill_shots5.md](04_livedoor/02_distill_shots5.md) | 同じ比較を、DeepSeek にカテゴリーごとに 5 件の例を見せて（few-shot）付けたラベルで行ったもの。例だけで学習した場合とも比べる | `19_distill_eval_livedoor.py --shots 5` |
| [03_feedback_rules.md](04_livedoor/03_feedback_rules.md) | 人の異議（正解のラベルで代用）を近い記事と並べて言語化した境目のルールと、採用するかの確かめ | `20_feedback_rules_livedoor.py` |
| [04_feedback_rules_dokujo.md](04_livedoor/04_feedback_rules_dokujo.md) | 同じ確かめを、独身女性のコラム / 映画、独身女性のコラム / 女性向けで行ったもの | `20_feedback_rules_livedoor.py --pairs ... --name 04_feedback_rules_dokujo` |
| [05_distill_shots5_rules.md](04_livedoor/05_distill_shots5_rules.md) | 採用した境目のルールを足して、独身女性のコラムか女性向けと判定された記事だけ付け直したラベルで、蒸留を評価したもの | `19_distill_eval_livedoor.py --shots 5 --rules` |

## [05_mlit/](05_mlit/)（Phase 2: 国交省の自動車不具合情報、正解のラベルあり）

| ファイル | 内容 | 作るスクリプト |
| --- | --- | --- |
| [01_baseline.md](05_mlit/01_baseline.md) | 正解の装置名をそのまま学習した判定機の精度（蒸留と比べるための基準） | `24_baseline_mlit.py` |
| [02_clean_train.md](05_mlit/02_clean_train.md) | 教師データから付け間違いの疑い・境目の申告を除いて学習し直した場合と、ランダムに除いた場合の比較 | `25_clean_train_mlit.py` |
| [03_count.md](05_mlit/03_count.md) | 装置名ごとの件数の数え方（1 つに決める / 確率で按分する）と、正解の件数とのずれ | `26_count_mlit.py` |
