# エンベディング環境

このプロジェクトでは **EmbeddingGemma 2** を、sentence-transformers でローカルに動かして使う。**Ollama は使わない。**
（2026-10-29 に開発用の PC で調べた内容）

## 開発用 PC の環境

| 項目 | 内容 |
| --- | --- |
| マシン | Apple M1 Pro / メモリ 32GB |
| 埋め込み用の環境 | `~/development/tools/embeddinggemma2-image` の仮想環境（`.venv`）を使う |
| モデル | `google/embeddinggemma-2`（Hugging Face のキャッシュ `~/.cache/huggingface/hub` に約 1.5GB） |

### 埋め込み用の環境（`~/development/tools/embeddinggemma2-image`）

uv で作られた Python 3.12 の環境。中身は次のとおり。

| パッケージ | バージョン |
| --- | --- |
| sentence-transformers | 6.1.0 |
| transformers | 5.19.0 |
| torch | 2.14.1（MPS が使える） |

- 埋め込みのスクリプトは `uv run --project` でこの環境を指定して実行する
- この環境は埋め込み専用。判定機の学習に使うパッケージ（scikit-learn など）は、このリポジトリの uv 環境に `uv add` で足す

## EmbeddingGemma 2 の概要

- Google DeepMind が出している、オープンな埋め込みモデル。Gemma 4 をもとに作られている
- テキスト、画像、動画、音声を同じ 768 次元の空間に埋め込める
- パラメーター数は全体で 740M。うちテキスト部分は 270M。画像と音声のエンコーダーは読み込まないこともできる
- 100 以上の言語に対応。最大 8,192 トークンまで入る
- ライセンス: Apache 2.0
- 前のバージョン（EmbeddingGemma 300M）より、多言語の MTEB スコアが少し高い（61.36 と 61.15）

### 次元を減らす（MRL）

768 次元を、先頭から 512 / 256 / 128 次元に切り詰めて使える。切り詰めたら正規化し直す。

| 次元 | MTEB（多言語） |
| --- | --- |
| 768 | 61.36 |
| 512 | 61.17 |
| 256 | 60.41 |
| 128 | 57.89 |

256 次元までは精度がほとんど落ちない。判定機を速く・軽くしたいときは 256 次元を試す。

### タスク用のプレフィックス

テキストの先頭に、タスクを表す短い文字列を付けて学習されている。付けないと精度が下がる。

| 用途 | `prompt_name` | 実際に付く文字列 |
| --- | --- | --- |
| 分類 | `Classification` | `task: classification \| query: ` |
| クラスタリング | `Clustering` | `task: clustering \| query: ` |
| 類似度 | `SentenceSimilarity` | `task: sentence similarity \| query: ` |

- このプロジェクトでは、学習も判定も **`Classification` に統一する**（同じプレフィックスを付けないと特徴量がずれる）
- Phase 2 でラベルのないテキストをグループ分けするときは、`Clustering` も試す

### 注意点

- **float16 は使わない**。値が NaN になったり、エラーを出さずに精度が落ちたりする。Mac（MPS）では float32 を使う
- sentence-transformers 6.1.0 以上が必要

## 使い方

- サンプル: [01_scripts/embed_sample.py](../../01_scripts/embed_sample.py)（実行方法はスクリプトの先頭に書いてある）
- テキストだけ使うときは、画像と音声のエンコーダーを読み込まない設定にする（メモリが 740M 分から 270M 分に減る）
- モデルはキャッシュに入っているので、`HF_HUB_OFFLINE=1` を付けてネットワークにつながずに動かす

## 動作確認の結果

日本語の口コミ 4 種類 × 64 件 = 256 件を、上の設定（MPS、float32、テキストのみ、batch 64）で埋め込んだ。

- モデルの読み込み: 約 3 秒
- 速さ: 約 250 件/秒
- 短い文で測った値なので、実際のデータで測り直すこと

## 出典

- モデルカード: https://huggingface.co/google/embeddinggemma-2
- ドキュメント: https://ai.google.dev/gemma/docs/embeddinggemma
