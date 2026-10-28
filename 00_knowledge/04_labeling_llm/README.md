# ラベル付け（蒸留）に使う LLM

Phase 2 では、LLM にラベルを付けさせ、それを学習データにして判定機を作る（蒸留）。
そのため、**出力を別のモデルの学習に使ってよいと規約で認められている LLM** を使う必要がある。
（2026-10-29 時点の調べ）

## 比較

| LLM | 出力で別のモデルを学習してよいか | 備考 |
| --- | --- | --- |
| jev（TypeSafe） | **不可**。蒸留と、出力をまねる学習が禁止（MCA §2.3(b)） | 判定に使うだけなら可 |
| OpenAI | 不可。競合するモデルの開発に出力を使うことが禁止 | |
| Claude（Anthropic） | 不可。Anthropic と競合するモデルの学習に出力を使うことが禁止 | 用途を限った小さな判定機が「競合」に当たるかはあいまい |
| **DeepSeek API** | **可**。規約で「別のモデルの学習（モデルの蒸留など）」に使えると明記 | 下記 |
| ローカルのオープンモデル（Apache 2.0 のもの） | 可（Apache 2.0 は出力の使い方を制限しない） | チェックポイントごとにライセンスを確認する必要あり |

## DeepSeek API

### 規約（Open Platform Terms of Service、2026-04-29 施行）

- §4.2(2): 出力の権利は利用者に譲渡される
- §4.2(3): 入力と出力は幅広い用途に使える。その例に **「別のモデルの学習（モデルの蒸留など）」** がはっきり挙がっている
- §4.1: 出力が規約や法律に違反しないこと、他人の権利を侵害しないことが条件
- §8.1: 出力を利用者向けに表示するときは、AI が生成したものだと明示する
- §10.1: 準拠法は中国本土の法律

### プライバシー（注意点）

- 送ったデータは **中国のサーバーに保存される**
- 送ったデータが DeepSeek のモデルの学習に使われることがある。拒否するには privacy@deepseek.com に申し出る
- 保存期間は決まっていない（アカウントがある間は保持）
- **公開データ（MASSIVE など）を送るのは問題ないが、個人情報を含むデータや社外秘のデータは送らない**

### モデルと料金（100 万トークンあたり）

| | deepseek-flash（V4.1-Flash） | deepseek-v4-pro |
| --- | --- | --- |
| 入力（キャッシュなし） | $0.15（ピーク時 $0.30） | $0.66（ピーク時 $1.32） |
| 入力（キャッシュあり） | $0.003（ピーク時 $0.006） | $0.022（ピーク時 $0.044） |
| 出力 | $0.60（ピーク時 $1.20） | $1.98（ピーク時 $3.96） |
| 同時リクエスト数の上限 | 2,500 | 500 |
| コンテキスト | 1M | 1M |

- ピーク時間は平日の UTC 01:00〜04:00 と 06:00〜10:00（日本時間 10:00〜13:00 と 15:00〜19:00）。それ以外は半額
- JSON 出力に対応。OpenAI 形式（`https://api.deepseek.com`）と Anthropic 形式（`https://api.deepseek.com/anthropic`）の両方で呼べる
- 思考モード（thinking）が標準で有効。ラベル付けのような単純な作業なら、思考なしのほうが速くて安い

### 費用の見積もり（MASSIVE 2,000 件）

- 1 件あたり入力約 400 トークン（指示と 18 グループの説明。共通部分はキャッシュが効く）、出力は数トークン
- flash で 2,000 件なら、多く見ても数セント

### このリポジトリで試した結果（MASSIVE 2,000 件）

- 設定: `deepseek-flash`、思考なし、`temperature=0`、JSON 出力、同時 32 リクエスト
- 速さ: 2,000 件を 45.7 秒（約 44 件/秒）
- トークン: 入力 837,133（うちキャッシュ 512,000）、出力 14,322。料金表から計算して約 $0.06
- 形式が崩れた応答: 0 件
- 正解との一致率: 90.4%
- 詳しくは [03_results/massive_distill.md](../../03_results/massive_distill.md)

## ローカルのオープンモデル

- Gemma 4、Qwen3 系の中小モデルなどは Apache 2.0 で公開されている（チェックポイントごとに要確認。Qwen の最大モデルなどは独自ライセンス）
- データが外に出ない。規約の心配もない
- 開発用 PC（M1 Pro / 32GB）で動くのは 8B〜14B 程度まで。精度と速さは API より落ちる

## 出典

- DeepSeek Open Platform Terms of Service: https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html
- DeepSeek Privacy Policy: https://cdn.deepseek.com/policies/en-US/deepseek-privacy-policy.html
- DeepSeek 料金: https://api-docs.deepseek.com/quick_start/pricing
- TypeSafe Master Customer Agreement: https://typesafe.ai/legal/mca
- Claude の出力を学習に使えるか: https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model
