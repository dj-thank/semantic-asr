# Grouped realtime の計測準備

対象は PR #74 の grouped runtime を使う明示的な研究用 WAV runner。
今回追加するのは実行時の計測と比較条件の記録であり、実音声の精度改善を示す結果ではない。
現環境には Reazon/Silero モデルと評価音声がなく、新規の実音声推論は実行していない。

## 同じ条件で2つの経路を記録する

権利確認済みの mono PCM16 / 16 kHz WAV と、既に取得・確認済みのローカルモデルを使用する。
音声・モデルの取得は runner の外で行う。参照文や評価辞書を decoder に渡さない。

```bash
python scripts/realtime_reazon.py "$AUDIO_WAV" \
  --model-dir "$REAZON_DIR" --model-sha256 "$REAZON_SHA256" \
  --vad-model "$SILERO_ONNX" --vad-model-sha256 "$SILERO_SHA256" \
  --threads 2 --allow-local-research --measure-runtime \
  --events-jsonl runs/first-pass-measured.jsonl

python scripts/realtime_reazon.py "$AUDIO_WAV" \
  --model-dir "$REAZON_DIR" --model-sha256 "$REAZON_SHA256" \
  --vad-model "$SILERO_ONNX" --vad-model-sha256 "$SILERO_SHA256" \
  --threads 2 --allow-local-research --measure-runtime --grouped-refine \
  --events-jsonl runs/grouped-measured.jsonl
```

出力先は新規にする。実験前に音声件数・総時間・試行数・計算時間・保存容量を固定する。
今回の flag は native 推論の強制停止や全音声実行の wall/storage quota を提供しない。
実音声の実行前に、外部の監督プロセスと実行環境の上限を別途設定する。
既存の研究パイプラインへの grouped 計測経路の組込みは未実装。
単一の WAV を繰り返しても、独立した自然会話・話者を評価したことにはならない。

`--measure-runtime` は既定 OFF。無効時の run summary は従来の形を維持する。
group outcome の時間フィールドは追加メタデータで、従来の evidence digest は変更しない。

## 時間の定義

| JSON の指標 | 測る区間 | 含まないもの |
|---|---|---|
| `setupMs` | first-pass adapter、VAD、session の生成 | optional import、事前 hash、worker の遅延生成 |
| `fastFinalDecodeMs` | 既存 final event の decoder 呼出し | VAD 判定、queue、JSON 出力 |
| `finalEmittingCallMs` | final を返した feed/flush 呼出し | 呼出し前の VAD/PCM hash、JSON 出力 |
| `refineQueueWaitMs` | worker へ受付してから処理開始まで | coordinator が submit する前の処理 |
| `refineWorkerMs` | worker の処理開始から decode 結果検査完了まで | queue、owner の poll/JSON 出力 |
| `refineFactoryMs` | decoder factory の呼出し | 後続の warm decode。後続は `null` |
| `refineCompletionAfterSubmitMs` | worker 受付から decode 結果検査完了まで | owner の poll、画面表示、JSON 出力 |

既存 outcome の `decodeDurationMs` は後方互換のため保持する。これは factory を含み得る
worker service 時間であり、純粋な認識処理だけの値に読み替えない。
factory 時間も OS cache を空にした cold-load ベンチマークとは限らない。
speech-end から final まで、group-close から画面反映までの latency は今回測っていない。

`wallSeconds` / `rtf` は従来どおりモデル生成後のループと EOF drain を含む。
`--realtime` では意図的な sleep を含み、`rtfIncludesRealtimeSleep=true` を記録する。
sleep を含む値と含まない値を同じ処理速度の指標として比較しない。

## 全件の分母と有限の保持

6種類の時間分布ごとに count/min/max/mean と p50/p95 を返す。
百分位は既存 benchmark と同じ線形補間を使用する。
`--timing-max-samples` は分布ごとの保持件数で、既定10,000、許容1〜100,000。
上限超過後も全件 count/min/max/mean を更新するが、p50/p95 は `null`、`complete=false`。
初期の一部だけを全件の分布として表示しない。

短過ぎる group、queue capacity、terminal worker、reset/timeout の取消には、
実行していない処理時間を0として付けない。未計測は `null` とし、
`untimedRefineOutcomes` と既存の outcome status 件数に残す。
`runtimeMetrics.complete` は時間標本の保持が完全かを表す。認識の成功や全groupの計測を
意味しない。認識は run status、時間の欠落は個々の count/untimed 件数で別に確認する。

## 比較条件の識別

summary 内の `runtimeMetrics.identity` は次のものを保持する。

- 入力ファイルの SHA-256 と、実際に投入した全 PCM の SHA-256
- chunk sample 数と直列化された speech 判定による VAD trace SHA-256
- ASR/VAD モデル、ASR runtime、NumPy、CPU/VAD threads
- first-pass/VAD/decode の設定と realtime sleep の有無
- runner、音声前処理、adapter、group runtime、時間集計等の実装ファイル識別値

VAD trace は `semantic-asr-vad-trace-v1\0` を初期値とし、各chunkのsample数を
8 byte big-endian、その speech bool を1 byteで順に hash する。
`comparisonIdentitySha256` はこの identity を既存 `sha256_json` で結び付ける。
group設定/capacityは変更対象のため共通identityから分けて記録する。

共通identityが違うrunは同一入力・設定のpaired比較として扱わない。
入力ファイルと実装ファイルは開始前後に確認し、差分や WAV header と投入sample数の
不一致があれば、completed summary を出さずに失敗する。
hash は実行内容や音声の正しさの証明ではなく、比較条件を照合する診断値である。
hardwareの同等性、依存環境全体、話者の独立性や音声の未見性も証明しない。
OS/Python/architecture は別の environment フィールドに残す。

grouped mode は追加モデルを使い、同じ threads 指定でも全体の CPU/RAM 予算は増え得る。
同じ機器・外部の総計算上限・固定した試行順序/回数と品質条件を揃え、費用と品質を別に測る。
未計測の peak RAM/CPU や同一計算予算での優越性を、この時間集計から推定しない。

## 残る品質評価

参照文をdecodeへ渡さず、保存したfinal/outcomeを固定の参照・cohortと比較する評価器は別作業。
parent textを単純連結すると重複pre-rollが混ざり得るため、group全文を発話に文字列分割したり、
groupを独立録音としてbootstrapしたりしない。既存のrights/lineage、literal CER、
expected cohort付きpaired count比較を再利用する。
候補coverage/oracle CER、最終CER、誤修正、否定/数字/固有名詞と時間/資源量を分けて評価する。
同モデルの再認識候補を独立した音響検証や自動適用結果として扱わず、既定OFFを維持する。
