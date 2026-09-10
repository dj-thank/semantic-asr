# Codex handoff — READMEの意図を実行できる形にする

2026-09-06の準備スナップショット。親は #23、実音声の有限改善サイクルは #30 / #21。
これは研究完了報告ではない。開始時にlive Issue/PRと実際のHEADを再確認する。
監査したmain: `7c36c9323626a83fdb89dddccb93f7252930666e`、
tree: `b1beddecd18464f8343b248dba2d624078fb4314`。

## 1. 達成すること

目的は「文脈的に自然な文を作ること」ではなく、**実際の日本語発話を音に忠実に復元すること**。
README、`docs/ARCHITECTURE_V0.3_MULTILEVEL_DELIBERATION.md`、
`docs/DOCUMENT_JOINT_BEAM.md`、現在の実装を一緒に読む。

| READMEの意図 | 実装・検証の到達条件 |
|---|---|
| A候補だけでなくB/C/Dも証拠 | N-bestのpathとscore domainを保存。oracle@Kとcoverage不足を先に測る。 |
| 発話全体の文脈も考慮 | 実際の順序を持つ録音でno-context/left/bidirectional/shuffled/distractorを比較。独立文の連結で代用しない。 |
| 音素・モーラも使う | audio-derived posteriorと候補から作るG2Pを区別。同じ音のphone/moraを独立票として二重加算しない。 |
| 自然でも音と違う文を採らない | 文法honeytrap、否定・数字・固有名詞、候補外生成の音響再確認、abstentionの回帰テスト。 |
| 実際の発話を保存 | observed / normalizedの分離、フィラー・言い直し・本当の反復、未適用のdocument判断で本文不変。 |
| 手作業でつなぎ続けなくてよい | 同一コマンドをCodexとCIで実行。実音声は別の有限stage driverに統合し、失敗・中断も証拠化。 |
| 軽く、必要な時だけ重くする | CPU baselineと同一budgetで比較。RTF、p50/p95、RAM/VRAM、追加モデル呼出率も記録。 |

## 2. 今すぐ動く、モデル不要の検証

Codex cloudの環境setupでは、package indexへの許可されたアクセスで次を実行する。
リポジトリに置いたscriptだけでcloud設定が自動更新されたことにはならない。

```bash
bash scripts/codex_setup.sh
```

Windowsまたは通常のPython環境では同じ処理を直接実行する。

```bash
python -m pip install -e '.[dev]'
python -m pip check
python scripts/codex_verify.py --plan
python scripts/codex_verify.py --profile installed --output-dir ../semantic-asr-evidence/run-001
```

出力先は新規かつcheckout外。既存の結果は上書きしない。指定しなければ一意のtempディレクトリを使う。
通常の開発途中のdirty treeも正直に記録し、実行中にsource/index/HEADが変われば失敗する。
`installed` はその環境で実行した証拠であって、dependency-free環境を検証したという意味ではない。

```bash
# PyTorchを含まないbase環境で実施する場合
python scripts/codex_verify.py --profile core --output-dir ../semantic-asr-evidence/core-001

# CPU PyTorch / NumPy / Safetensorsを別環境へ用意した場合
python scripts/codex_verify.py --profile cpu --output-dir ../semantic-asr-evidence/cpu-001
```

固定stage: format check → lint → compile → pytest/JUnit → 8件の保存判定replay →
demo → synthetic ranker optimization → CLI discovery → **isolated wheel build** →
新規venvへのwheel install → **checkout外・isolated Pythonのimport/export確認** → wheel demo/smoke。
CLIは`python -m semantic_asr`を使う。`python -m semantic_asr.cli_root`には実行ガードがなく、
何もせず0で終了するため代用しない。必須成果物が生成されなければrunnerも失敗する。

既定のwall budgetは全体1200秒、各stage600秒。`--total-seconds` / `--stage-seconds`で有限値を指定できる。
失敗後のstageは実行しない。timeoutは子プロセス群を停止する。自動修正・再試行・pushはしない。
モデルのダウンロードは禁止する環境設定を渡すが、これはOSレベルのnetwork sandboxの代わりではない。
wheelの**build依存**は隔離buildで取得が必要な場合がある。許可されない環境では失敗を残し、
`--no-isolation` / `--skip-dependency-check`で同等の検証をしたことにしない。

`report.json`にはcommit/tree、実効source/index hash、package versions、コマンド、所要時間、
終了コード、test数、skip/xfail、成果物hashが残る。rawログ・wheel・synthetic出力はローカルのみ。
CIはreportと必要なsource-format診断だけをallowlistで公開し、ディレクトリ全体をuploadしない。
終了コード: `0`は固定チェック成功（skipがあれば`passed-with-skips`）、`1`は失敗、
`2`はpreflight失敗、`124`はtimeout、`130`は中断。skipは実行済みと数えない。

この入口は研究schedulerやcheckpoint形式を増やさない。既存の本体テスト・CLIを束ねるだけ。
`.github/workflows/codex-readiness.yml`はこの入口をLinux 3.11/3.12、Windows 3.12で実行する。
既存CIのoptional CPU suiteは引き続き必要。setupは現状のdev extraを使い、transitive lock完成を名乗らない（#28）。

## 3. Codexに渡す実装タスク

次の内容をCodexのtaskに渡す。GitHub連携済みなら、この準備PRへの`@codex`コメントでも同じ依頼を渡せる。
依頼の投稿、Codexの受付、実行完了は別状態。受付の反応・task URL・結果がなければ起動完了とは報告しない。

```text
READMEの「音に忠実な日本語復元」を実現してください。計画を書くだけで終わらず、
AGENTS.md、docs/development/CODEX_AUTOPILOT.md、live #23/#30と依存Issue、現在のsourceを読み、
1つの未予約の実装単位を選び、再現テスト→実装→検証→証跡付きPRまで進めてください。
最初にこの準備PRのcodex_verifyを全matrixで確認し、次に#30の実音声stage統合を進めます。
#26/#28/#29の必要な契約が未統合なら、本番音声runを成功扱いせず、独立に作れるdriverと負例テストを進めること。

実音声は provision→collect→score→fit-dev→calibrate→freeze→evaluate-heldout→
classify-errors→report の有限cycleにする。既存CLI/experiment_runner/rights/score型を再利用。
referenceをruntime selectorへ渡さず、評価はfreeze後。データ・設定が変わったresumeは拒否する。
全trial/失敗/negative resultを保存し、同budget baselineと比較する。
旧72音声とpilot40音声を未使用testに戻さない。8fixture replay、96保存判定audit、新規推論を区別。
PR #50の自動統合、予約枝の上書き、拒否済みpatchの別経路再公開、自動merge/default昇格はしない。

権利・モデル・データ・computeがない場合は具体的なblockerと最後の成功stageを残す。
有料API/GPUを勝手に用意せず、既に許可されたローカル資源でできる独立作業を完了する。
最後に正確なSHA/tree・commands・test/skip・artifact hashと、engineering/experiment/promotionを分けて報告する。
```

## 4. 実音声自動パイプラインの受入条件（#30、未完了）

これは次に実装する契約であり、`codex_verify.py`が既に音声学習を実行するという意味ではない。
既存 `scripts/run_real_audio_pipeline.py` は候補生成**後**のpartition/ranker/calibration/benchmark driver。
`experiment_runner.py`のgeneration checkpoint/lockと`experiment.py`のpaired評価を再利用する。
新しいrights registry、score型、汎用agent frameworkは作らない。

| Stage | 入口と、成功と認める証拠 |
|---|---|
| provision | 権利操作・モデル/データrevision・label schema・source/PCM hash・speaker/session・exposureを検証。train/dev/calibration/testの役割を凍結。 |
| collect | 固定音声からpath-preserving候補を生成。referenceをdecode prompt/contextへ流さない。同一音声/設定のprefixだけresume。 |
| score | no-change baseline、音声/phone/mora/contextを別証拠として固定。model/tokenizer/preprocess/score domainを結ぶ。 |
| fit-dev | trainで実更新、devでのみ有限候補gridを選択。全trial/失敗/不採用理由・seed・optimizer historyを保存。 |
| calibrate / freeze | 専用calibration splitで校正し、採用設定と入力artifactのdigestをfreeze receiptに固定。 |
| evaluate-heldout | receipt検証後に、固定expected IDsで全systemを同じgroup drawsで比較。両systemから消えた難例も拒否。 |
| classify-errors | coverage / selection / segmentation / G2P / acoustic / orthographic / unknown。音を確認せず原因を断定しない。 |
| report | strict/lenient corpus CER、utterance mean、meaning-critical/false correction/coverage、RTFと資源量。全失敗も残す。 |

`max_trials / max_audio_seconds / max_wall_seconds / max_storage_bytes`を実行設定の必須正整数にする。
外部API費用は初期値0、GPUの新規provisionは不可。正の費用を許すには別の明示承認が必要。
制限到達は停止・partial/failedであり、サンプルを勝手に捨ててsuccessへ変えない。
本学習のcheckpoint/resumeは#35の承認された契約が前提。世代の違うcheckpointを自己流で混ぜない。

最初のCodex PRはstageの状態遷移、input/output digest、freeze-before-test、budget、reference隔離、
異常時終了、設定違いresume拒否をテストし、既存driverへ接続する。実資源がない時もここは準備できる。
その後、権利・lineage・環境・評価契約が揃った小さな固定日本語subsetで1cycleを実行する。
これだけではacoustic head/LLM重みの研究、長録音context評価、production昇格は完了しない。

### 現在見つかっている整合性の課題

* `real-audio-research.yml`の`generate-candidates`呼出しには現在必要な`--allow-raw-export`がない。
  承認済みlocal-research入力とcheckout外出力という契約をworkflow全体で揃える。
* workflowはdriverと処理を二重に持つ。修正後は**同じ**driverを手動/CIで使い、別実装にしない。
* runtime revisionはCTranslate2の番号だけでは不十分。実際のsource、faster-whisper、CTranslate2、
  model loaderへ渡したrevisionを記録し、存在しない固定値を名乗らない。
* READMEの公開test-set materialization例は操作説明であって、公開testの再分割を
  新しい未使用testや実験の学習許可に変えるものではない。#26の役割・権利契約を優先する。
* post-candidate scriptが完走しても#30のdev選択/freeze/resume/budget/全96判定の再推論が完成したとは言えない。
  分割・ranker学習入力・校正のCLI schemaを実際のfixtureで結合テストする。

## 5. 作業順序と統合

#19/#24/#25は既存の完了成果を再利用。#52は#54で統合済みだが親#29は別の受入条件を持つ。
#26には`codex/dataset-lineage-contract-26`の予約があり、#28/#35もliveコメントでownershipを確認する。
#50はこの監査時点でmain向けの**draft**、公開head `055e4689078d7a238ad0097802c90f5a749e52c6`。
古いhandoffの「旧#17向け」をlive状態として使わない。未公開local candidateのテスト結果を公開headへ帰属させない。
拒否済みpatchの再公開をこのタスクへ紛れ込ませない。統合には別の権限・レビュー・同一treeでの全検証が必要。

順序: #26/#28の契約を確認 → #29の評価境界 → #30 driver → #34 runtime →
#35 resume → #36 acoustic / #37 reranker/LoRAの別実験 → #38本物の会話context → #40再現報告。
依存未完了でも独立な再現テスト・driverのstate machine・read-only検証の準備は進める。

終了条件は「各Issueの証拠が揃ったこと」。精度が上がらなかった実験も完了結果になり得る。
既定profileへの昇格は、事前固定したquality/cost/rights gateと独立レビューが揃うまで行わない。

## 6. 公式のCodex設定参考（2026-09-06確認）

- AGENTS / customization: https://developers.openai.com/codex/customization/overview
- repository skill discovery: https://developers.openai.com/codex/skills/
- cloud environment setup: https://developers.openai.com/codex/cloud/environments
- GitHub task delegation: https://developers.openai.com/codex/third-party/github

このPRはCodexのアカウント設定やcloud環境を勝手に変更しない。repo skillはCLI等の対応surfaceで
発見される。cloud taskでもAGENTSとこのhandoffを明示的に読むため、skillの自動検出だけに依存しない。
