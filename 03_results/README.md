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
