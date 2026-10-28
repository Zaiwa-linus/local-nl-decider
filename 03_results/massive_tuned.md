# MASSIVE（日本語）の評価結果（パラメーター調整後）

768 次元。パラメーターは validation の Macro-F1 で選び、test で評価した。

| ラベル | 判定機 | パラメーター | validation Macro-F1 | test Accuracy | test Macro-F1 | 学習時間（秒） | 判定の速さ（件/秒） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scenario | logreg | C=30 | 0.911 | 0.908 | 0.907 | 0.64 | 2,584,185 |
| scenario | knn | k=10 | 0.909 | 0.908 | 0.905 | 0.0 | 7,462 |
| scenario | mlp | hidden=256 | 0.908 | 0.912 | 0.910 | 2.15 | 1,282,527 |
| intent | logreg | C=100 | 0.864 | 0.859 | 0.838 | 2.17 | 2,496,270 |
| intent | knn | k=10 | 0.865 | 0.866 | 0.843 | 0.01 | 7,709 |
| intent | mlp | hidden=1024 | 0.845 | 0.865 | 0.838 | 7.74 | 404,575 |

## 間違えやすい組み合わせ（scenario、ロジスティック回帰、上位 10）

| 正解 | 予測 | 件数 |
| --- | --- | --- |
| general | qa | 20 |
| qa | general | 20 |
| general | calendar | 13 |
| music | play | 8 |
| general | news | 8 |
| news | general | 7 |
| general | play | 7 |
| news | qa | 6 |
| play | music | 6 |
| weather | general | 6 |

## 間違えやすい組み合わせ（intent、ロジスティック回帰、上位 10）

| 正解 | 予測 | 件数 |
| --- | --- | --- |
| general_quirky | qa_factoid | 14 |
| calendar_set | calendar_query | 14 |
| calendar_query | calendar_set | 13 |
| qa_factoid | general_quirky | 12 |
| general_quirky | news_query | 10 |
| news_query | general_quirky | 9 |
| general_quirky | calendar_query | 7 |
| play_music | play_audiobook | 6 |
| general_quirky | calendar_set | 6 |
| play_music | play_radio | 5 |
