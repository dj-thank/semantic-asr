# ローカル音素観測とQwen主認識の試験経路

音声からの読みと文脈に合う表記を分けて確かめるための、明示的に有効化する経路。
学習済みモデルの採用・全体精度の向上・gold音素/モーラ評価の完了を意味しない。

`semantic-asr run --profile qwen-ja-cpu-v1` は既存のQwen3ASRAdapterを主認識器として使う。
別途用意したローカルモデルの `--qwen-model-dir` と `--qwen-artifact-sha256` が必須。
CPU float32、入力ごとに1候補、最大256生成token。認識結果は未校正の暫定結果として返す。
通常のWhisper profileの既定値は変更していない。

各profileで次の引数を追加すると、同じ音声区間からHuBERT CTCによる音素を観測する。

```text
--phone-model-dir <local-model-directory> --phone-artifact-sha256 <sha256>
--phone-check-candidates
```

モデルは48ラベルの日本語HuBERT phoneme CTCなど、対応するHuBERTForCTCのローカルartifact。
音素観測の入力はmono 16 kHz PCM16 WAVのみ。他形式は主認識で読めても、
この音素経路では自動変換せずunavailableとして報告する。
読み込みには別環境に `torch`、`transformers`、`numpy`、候補照合には `pyopenjtalk-plus` が必要。
実行時にモデルをダウンロードしない。qwen主認識には追加で `qwen-asr` が必要。
基礎パッケージのimport/helpにはこれらの依存・モデルは不要。

音素診断は `transcript.json` の `diagnostics.phoneEvidence` に保存する。観測のために
参照文をencoderへ渡さず、音声SHA-256、実sample区間、PCM、モデル、ラベル、前処理、
runtime版を結び付ける。録音終端のms丸めによる端数sampleも保持する。
モーラは同じ音素列の可逆なグループ化であり、独立した音響票ではない。
時刻はencoderのframe gridによる推定で、正解境界やforced alignmentではない。

候補照合を有効にした場合、既存候補のうち一致する全文区間を明示したものだけを、
既存の厳密CTC forwardで採点する。G2Pは候補読みの提案であり、発音の観測・正解ではない。
各候補の失敗・時間切れ・部分区間を分母へ残す。同音異義語はこの音響照合だけでは解決しない。
この経路は観測済みの本文とそのhashを変更せず、文脈を使う選択のための証拠を追加する。

観測は1窓30秒以下、最大16窓、窓間判定で120秒。候補照合は1窓最大8候補、
候補間判定で10秒。OSによる厳密な割込み制限ではない。失敗した窓はunavailableとし、
取得済み文字起こしは返す。新しいbackendの成功をDEVICE/PROVIDER/PUBLIC PASSへ拡大しない。

対象は#34の局所的な準備と実行経路。#21/#23/#30の全受入を閉じるものではない。
