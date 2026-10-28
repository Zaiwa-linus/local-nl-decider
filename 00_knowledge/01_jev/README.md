# jev（TypeSafe AI）

2026-10-29 時点で Web 上の情報から調べたメモ。正式な仕様は公式ドキュメント（https://docs.typesafe.ai/api）を見ること。
早期アクセス中のサービスなので、仕様や価格はすぐ変わる可能性がある。

## 概要

- TypeSafe AI が出している「System One」と呼ばれる判定用のモデル。2026-09-15 に早期アクセス版が公開された
- 文章を生成する LLM ではない。こちらが決めた選択肢から 1 つを選び、選択肢ごとの確率を返す
- 判定の理由は説明しない。文章も書かない
- タスクごとの学習はいらない。質問と選択肢を書けばそのまま使える（ゼロショット）
- 現在のモデル: `jev-latest`（= `jev-1.13.0`）、`jev-preview`（最新ビルド）
- 入力はテキストのみ（画像は特別なヘッダーを付けたときだけ）。1 リクエストあたり最大 64k トークン

## このプロジェクトでの使い道

Phase 2 で、ラベルのないテキストにラベルを付けるときに使う。

- `choice` 型の質問でカテゴリーを決めさせ、`probabilities` と `confidence` も保存する
- `confidence` が低いものは学習データから外すか、Claude や人が確認する
- カテゴリーの定義（`criteria` に書く名前と説明）は、Claude と一緒に作るのがよさそう（jev は文章を書けないため）

## API

### エンドポイントと認証

- エンドポイント: `POST https://api.typesafe.ai/v1/systemone`
- 認証: `Authorization` ヘッダーに `Bearer <API キー>` を入れる
- 使えるモデルの一覧: `GET /v1/models`
- ストリーミングはない

### リクエストの中身

| フィールド | 必須 | 内容 |
| --- | --- | --- |
| `model` | ○ | `jev-latest` など |
| `state` | ○ | 判定したいテキスト（文字列・オブジェクト・配列のどれでもよい） |
| `questions` | ○ | 名前を付けた質問の一覧（1 つ以上） |
| `assets` | | 画像（使う場合のみ） |

### 質問の種類

| type | 用途 | `criteria` | 返ってくる値 |
| --- | --- | --- | --- |
| `choice` | 選択肢から 1 つ選ぶ | 必須。`{選択肢名: 説明}`、1〜255 件 | `choice`, `confidence`, `probabilities` |
| `noul` | はい／いいえ | 任意。`{"true": 説明, "false": 説明}` | `noul`（0〜1 の確率） |
| `score` | 段階で評価する | 必須。段階の説明を並べた配列、最大 10 件（先頭が 0 点） | `score`（期待値）, `confidence`, `legend`, `probabilities` |
| `box` | 画像の中の位置を返す | — | `box`（top/bottom/left/right） |

### レスポンスの中身

| フィールド | 内容 |
| --- | --- |
| `model` | 実際に答えたモデル（`jev-latest` を指定すると `jev-1.13.0` など） |
| `answers` | 質問の名前ごとの答え。中身は上の「返ってくる値」 |
| `usage` | `input_tokens`（課金対象）と `output_tokens`（今は無料） |

### 例（口コミのカテゴリー分け）

[01_scripts/jev_sample.py](../../01_scripts/jev_sample.py) を参照。

### エラー

- `400`: リクエストがおかしい（`detail` に理由が入る）
- `422`: 入力の検証エラー（`detail` に一覧が入る）

## Python SDK

- パッケージ名は `typesafe-sdk`（使うときは `uv add` で入れる）
- 環境変数 `TYPESAFE_API_KEY` を読む
- `client.system_one(...)` に `state` と、`Noul` / `Choice` / `Score` で組み立てた `questions` を渡す
- 結果は `response.choices` / `response.nouls` / `response.scores` に型ごとに入る
- 同期版と非同期版のクライアントがある
- 初期のリリースで質問の型が変わったことがあるので、バージョンを固定して使う

## API キーの取り方

1. typesafe.ai のウェイトリストに登録する（今は招待制）
2. コンソールが使えるようになったら console.typesafe.ai/settings/keys でキーを作る
3. キーは一度しか表示されないので、すぐに控える
4. `export TYPESAFE_API_KEY="ts_..."`

## 価格と速度

- 入力: 100 万トークンあたり約 $0.042（OpenRouter の価格）。出力は無料
  - AIMLAPI 経由だと約 $0.058 という情報もある
- 応答時間: TypeSafe の発表では 70〜500 ms
  - 独立した検証では、ローカルで動かした Qwen とほぼ同じ速さだったという報告もある
- OpenRouter（`~typesafe/jev-latest`）、Vercel AI Gateway（`typesafe-ai/jev`）、LiteLLM 経由でも使える

## 苦手なこと（TypeSafe 自身が挙げているもの）

- 指示を文字どおりに受け取る。質問があいまいだと結果もあいまいになる
- 計算、数を数えること、日付の扱いが苦手（これらはコードでやる）
- 判定に関係ない文章が混ざると精度が落ちる
- 悪意のある文章に答えを誘導されることがある
- 用意した選択肢の中から、間違ったものを選ぶこともある

## 性能の報告（参考）

- 銀行のサポート問い合わせ 3,080 件の分類: jev 81.1%、Qwen 76.4%（Towards Data Science）
- エージェントの実行結果の合否判定 500 回: jev は人のレビューと全件一致、Claude Sonnet 4.6 は 80.0%（LangChain による初期の検証）
- 数字の多くは TypeSafe 自身か、それを引用した記事によるもの。自分たちのデータで確かめること

## 出典

- https://docs.typesafe.ai/api
- https://www.lindy.ai/blog/jev-vs-llm
- https://apidog.com/es/blog/jev-api-key/
- https://openrouter.ai/~typesafe/jev-latest
- https://docs.litellm.ai/docs/pass_through/typesafe.md
- https://realpython.com/ref/ai-coding-tools/jev/
