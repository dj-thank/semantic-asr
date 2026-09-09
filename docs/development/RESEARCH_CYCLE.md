# 有限の実音声サイクル

PR #55 の環境準備・検証手順と、PR #56 の実音声入口を統合した実行経路です。
認識・学習・校正は既存実装を使い、実行設定を先に固定して1試行を行います。
READMEの observed/normalized 分離、フィラー・反復保存を維持します。

```text
固定音声・モデルの確認 → 候補生成（既存checkpoint）
→ trainだけでrankerを学習 → calibrationだけで校正
→ 設定・成果物・評価対象IDを凍結 → 正解文なしで選択
→ 正解文を読み、同一候補の元のASR top1と比較 → 誤り分類 → レポート
```

## 実行

`AGENTS.md` と両方の引き継ぎ文書を先に確認してください。音声・モデルは事前に
明示的に取得します。実行時のモデル取得は無効です。成果物はcheckoutの外へ保存します。

```bash
python scripts/codex_pipeline.py research \
  --manifest /external/manifest.jsonl \
  --model /external/faster-whisper-model --model-artifact-sha256 EXACT_SHA256 \
  --allow-local-research --evaluation-role regression-exposed \
  --max-trials 1 --max-records 24 --max-audio-seconds 600 \
  --max-wall-seconds 1800 --max-storage-bytes 10737418240 \
  --ranker pairwise --epochs 20 --seed 17 --bootstrap-iterations 2000 \
  --beam-size 5 --hypotheses 5 --device cpu --compute-type int8 \
  --output-dir /external/cycle-001
```

同じコマンドへ`--resume`を追加すると、OSの排他ロックを取得し、入力・モデル・音声・
設定・コード・環境・成功成果物が一致する場合だけ再開します。完了stageは再実行しません。
設定や環境を変更した場合は新しいrunを使います。実行時間と学習試行数は再開後も累積します。
突然終了したrunは、保存された開始時刻から再開までの経過時間も保守的に消費扱いにします。
中断した学習のoptimizer再開は未対応です。最大試行数に到達した学習を黙って再試行しません。

候補を取得済みなら同じpost-candidate入口を直接利用できます。

```bash
python scripts/run_real_audio_pipeline.py --bounded \
  --candidates /external/candidates.jsonl --output-dir /external/post-cycle \
  --allow-raw-export --evaluation-role regression-exposed \
  --audio-seconds MEASURED_SECONDS --max-audio-seconds 600 --max-trials 1 \
  --max-wall-seconds 1800 --max-storage-bytes 10737418240
```

`--audio-seconds`は生成receiptで検証した音声量を渡します。この入口は再推論せず、
渡された候補の出自と音声量の正しさを単独で実証するものではありません。
従来の`--bounded`なしのpost-candidateコマンドも互換性のため残しています。
新しい実音声workflowは`codex_pipeline.py research`からこの有限経路だけを呼びます。

## 証拠と終了コード

- 親の`receipt.json`: モデル・WAV・manifest・コード・環境・累積時間・生成候補digest。
- `pipeline/config.json`: 固定入力hash、設定、予算、seed、コード・環境。
- `pipeline/cycle.json`: 成功stageの順序、試行台帳、失敗、累積予算、成果物hash。
- `pipeline/freeze.json`: 学習/校正artifactと全分割のhash、評価対象ID。
- `pipeline/paired-report.json`: strict/lenient corpus CER、utterance mean、群単位paired
  bootstrap、誤修正・改善・悪化・不変・追加証拠要求件数。
- `pipeline/error-taxonomy.json`: 候補不足と選択失敗を分離。音響/G2Pなど未確認の原因はunknown。

終了コードは0=完了、1=失敗、2=事前条件不成立、3=予算到達または中断です。
終了0でも必須成果物がなければ失敗です。評価対象の欠落を除外して成功にはしません。
正解文・仮説・ローカルパス・モデルを含む研究成果物を自動アップロードしません。

## 完成範囲

この経路は固定1試行のソフトウェア統合です。候補だけを受ける選択境界と、凍結後の評価を
別処理にしています。比較は同一候補生成予算ですが、rankerの追加計算があるので同じ総計算量とは
主張しません。別データでの精度向上やproduction昇格は自動判定しません。

現行のmanifest型はtrain/calibration/testのみです。#26の役割契約を変更せず、devをtrainやtestに
偽装して候補選びには使用しません。devでの有限grid選択、#35のoptimizer再開、音響/LoRA本学習、
gold音素/モーラ、未閲覧・話者独立のpublication評価は別の未完了条件です。#30全体を閉じる証拠ではありません。
公開テストセットを分割して使った動作確認も、未使用テストの品質実験とは呼びません。

検証は`python scripts/codex_verify.py --profile installed --output-dir /external/verification`。
Linuxの明示的CPU環境では`codex_pipeline.py check --lane training-cpu`も全テストとisolated wheelを
検証します。Windows固有のsymlink権限skipは実行済みと数えず、結果へ明記します。
