# Codexへの引き継ぎと自動パイプライン

**目的は「発話された日本語を、意味を勝手に直さず復元する」READMEの実現です。**
コードを増やすこと、CIを緑にすること、文章を自然にすることを精度改善の代用にしません。
このページは、Codex Cloud/CLIから同じ実行入口を使うための手順です。
Codexサービスへのタスク作成、アカウント設定、無人の永続稼働をこのPRが行うわけではありません。

## 最初の一回

リポジトリを開き、[AGENTS.md](../../AGENTS.md)、[README](../../README.md)、
[実行索引](README.md)、担当Issueの本文と最新コメントを読んでください。

```bash
python scripts/codex_pipeline.py plan
```

`plan`はインストール、モデル読み込み、ダウンロード、GitHub書込みを行いません。
実装済みの段階を表示します。Issueの完了状態を推測するものではありません。

### Codex Cloudの環境

Python 3.12を選び、setup scriptで次を実行します。setupの`activate`や`export`は
エージェントの別セッションには引き継がれないので、以下ではPythonの絶対/相対パスを明示します。

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]' 'setuptools>=77' wheel
.venv/bin/python -m pip check
```

maintenance scriptでも`.venv/bin/python -m pip install -e '.[dev]'`と`pip check`を実行し、
選択したbranchの依存に合わせます。依存が変わったらキャッシュも再確認してください。
モデルなしの検証にはAPI key、音声データ、GPU、エージェント段階のインターネットは不要です。
Windowsでは`.venv/bin/python`を`.venv\Scripts\python.exe`に読み替えます。

```bash
.venv/bin/python scripts/codex_pipeline.py check \
  --lane core --output-dir ../semantic-asr-evidence/core-001 \
  --max-wall-seconds 1200 --max-storage-bytes 1073741824
```

出力先は毎回新しいcheckout外のディレクトリにします。既存の実行結果は上書きしません。
開発途中の未commit差分は`dirty`とworkspace digestに記録されます。
PR最終確認では`--require-clean`を付け、**公開した実際のSHA/tree**で再実行します。

### CPUのoptional学習テスト

coreと別の環境を使います。`phonetic`と`rerank`を全部入りで混ぜないでください。
これは新規音響/LoRAモデル学習ではなく、既存の実backward等を含むソフトウェア検証です。

```bash
python -m venv .venv-cpu
.venv-cpu/bin/python -m pip install 'torch>=2.4' --index-url https://download.pytorch.org/whl/cpu
.venv-cpu/bin/python -m pip install -e '.[dev]' 'setuptools>=77' wheel \
  'numpy>=1.26' 'soundfile>=0.12' 'safetensors>=0.7,<1'
.venv-cpu/bin/python -m pip check
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-cpu/bin/python scripts/codex_pipeline.py check \
  --lane training-cpu --output-dir ../semantic-asr-evidence/cpu-001 \
  --max-wall-seconds 1200 --max-storage-bytes 1073741824
```

coreではPyTorchがあると停止します。training-cpuでは必要な依存の欠落やテストのskipを
成功にしません。coreのoptional skipはJUnitに残し、optional機能の実行証拠には使いません。
今後optional laneが増える場合は、skipを隠すのではなく検証行列を明示的に更新します。
`.venv-cpu/`はGitへ追加しません（専用ignoreを追加済み）。

## 自動で実行される開発検証

```text
format差分検査 → lint → compile → pytest/JUnit
 → 保存済み回帰fixture → demo → synthetic ranker/optimization
 → CLI help → wheel build → checkout外の新規venvでimport/demo/smoke/help
 → source不変確認 → receiptとartifact SHA-256
```

`check`はソースを修正しません。format違反のときは`format.log`に提案diffを残して失敗します。
Codexがその差分をレビューして修正し、再検証します。CIからformat修正をpushする仕組みではありません。
wheel検証は`-I`、`PYTHONPATH`除去、checkout外cwdとインストール先assertを使います。

Cloud/ローカルの標準buildは、setupで入れたビルド依存を使う`--no-isolation`です。
`--isolated-build`を明示したCIでは`python -m build`の分離環境を使います。
前者だけの成功を後者の成功とは報告しません。

`.github/workflows/codex-pipeline.yml`がPR、mainへのpush、手動実行で同じコマンドの
training-cpu laneを走らせます。既存CIのLinux Python 3.11/3.12・Windows Python 3.12、
他の契約検証を置き換えません。各workflowの**同じ最終head/統合tree**の結果を確認します。
PRイベントのmerge SHA、PR head、main SHAは別物です。receiptの`source`が検証対象です。

GitHubへ自動保存するのは、このモデル不要laneのreceipt、ログ、JUnitのみ（14日保持）です。
失敗時も保存します。期限前に必要な証拠を権利確認済みの長期保存先へ移してください。
環境のpackage一覧は観測記録であって、推移依存hashを固定したlockではありません（#28）。

## 明示的に実行する実音声パイプライン

前提は、**利用を確認したローカル音声・manifest、固定済みモデル、適合した環境**です。
既存のmanifest/rights検査と`run_real_audio_pipeline.py`を再利用します。
新たなrights registry、score型、checkpoint codec、汎用agent基盤を作りません。

```text
権利・split・WAV実時間・入力hash・モデル版の事前検査
 → 既存generate-candidates（既存の検証付きpartial出力）
 → 既存post-candidate driver
    partition → train-only ranker → calibration-only calibration
    → test選択 → raw baselineとcascade/MBR/oracleの比較
 → 入力とsourceの再検査 → local receipt
```

新入口は、mono 16 kHz PCM16 WAV、checkout外の絶対audioPath、全行のsplit、
`rightsDecision=allow`とlicense情報、非空train/calibration/testを要求します。
不足権利を勝手にallowへ書き換えて通してはいけません。
同一WAV bytesのsplit跨ぎは拒否しますが、これで話者・session・派生音声の分離全体を
保証するわけではありません。`speaker_disjointness_verified=false`を明記します（#26）。

モデルとデータの取得は、この実行より**前に別操作として**行います。
モデル取得が必要な場合は、READMEの固定revisionと権利、ディスク容量を確認してください。
実行時は`HF_HUB_OFFLINE=1`等を子プロセスへ設定し、未導入モデルで勝手にダウンロードを
始めません。依存不足は`blocked`、backend/cacheの不備は実際の失敗ログに残します。
これらの環境変数はモデルのネットワーク動作を制限するもので、OSの通信sandboxではありません。

以下は、READMEの手順で**既に作成した**Reazonのmanifestとキャッシュを使う例です。
既に見た公開テスト素材は`regression-exposed`であり、新しい未使用評価ではありません。
600行の各音声が合計3,600秒を超える場合、この例は停止します。実行後の結果に合わせて
予算や評価役割を変更せず、開始前に音声量と用途を決めてください。

```bash
python scripts/codex_pipeline.py research \
  --manifest ../semantic-asr-public-data/reazon/manifest.jsonl \
  --model large-v3-turbo --model-revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf \
  --device cpu --compute-type int8 --evaluation-role regression-exposed \
  --allow-local-research --max-records 600 --max-audio-seconds 3600 \
  --max-wall-seconds 3600 --max-storage-bytes 2147483648 \
  --output-dir ../semantic-asr-public-data/reazon/codex-run-001
```

`.[asr]`を導入したPythonを使います。local model directoryの場合は`--model-revision`ではなく
`--model-artifact-sha256`を使用し、既存`semantic_asr.revisions.sha256_artifact`の契約で検証します。
実験はclean commitから行います。ひとつのmanifestで1回だけ実行し、test指標を見て自動で
再学習するloopはありません。`max_records`、`max_audio_seconds`、`max_wall_seconds`、
`max_storage_bytes`は省略不可です。beam/hypothesesは各50以下、bootstrapは10,000以下です。

wall/storageは子プロセス監視であり、OSの厳密なquotaではありません。監視間隔中の超過、
事前hash検証、外部キャッシュ、OS一時領域までの総資源上限は、runner側のディスクquota・
コンテナ制限でも管理してください。timeoutでは子プロセス群を停止し`partial`を残します。
Windowsでもcheckを実行できますが、実音声の管理runnerはLinuxが対象です。

`pipeline/test.jsonl`というファイル名やコマンド完走は、未使用test・gold phone/mora・
実会議品質・統計的優越性・モデル昇格の証明ではありません。この入口の評価役割は
`development`または`regression-exposed`だけです。#26/#28/#29/#34と事前固定した昇格条件が
揃うまでは、fresh held-out promotion laneは未実装として扱います。

### GitHubから実音声を動かす場合

`Real audio research (self-hosted, opt-in)`は手動実行のみです。
まずhosted runnerで明示許可とmain refを確認し、その後に
`[self-hosted, linux, x64, semantic-asr-research]`の既存runnerを使います。
PRコードをprivate dataのあるrunnerで自動実行せず、GPUを購入・新規確保しません。
runner登録・ローカルmanifest・model cache・Python 3.12は別途必要です。

旧workflowの変更点は次の通りです。

- `--allow-raw-export`不足を直し、重複したpost-candidateコマンド列を既存driverへ委譲。
- `allow_local_research`は初期値false。有限予算と評価役割を明示し、clean sourceを要求。
- `experiment_name`による任意出力先をやめ、run ID/attempt別の新規外部出力を使用。
- `upload_text_artifacts`を廃止。音声・参照文・仮説・パス・重み・生ログは自動アップロードしない。

実行中のprivateなエラー詳細も外部ディレクトリのログへ残し、Actionsの通常出力には状態だけを
表示します。失敗ログ/receiptにもprivate pathが含まれ得るので、そのままGitHubへ投稿しません。
公開用redactionと操作別publish権利の実装は#26/#40の受入が必要です。

## 状態と、次にCodexが進める作業

| 状態/終了コード | 意味 | 次の操作 |
|---|---|---|
| `completed` / 0 | このモードのソフトウェア段階が完走 | 同一revisionの他CI、研究の受入条件を別に確認 |
| `failed` / 1 | コマンド失敗、入力/結果不正、source変更 | log/JUnitから再現し最小修正。結果を捨てない |
| `blocked` / 2 | 依存/明示許可/clean commit等が不足 | 不足を報告。権利や検証を緩めない |
| `partial` / 3 | 予算超過または中断 | 完走と呼ばず、既存checkpoint契約に従って別途再開判断 |

`promotion`は常に`not-evaluated`です。`research`が更新するのは既存の軽量rankerで、
新しい音響・LLM/LoRA重みを学習したという主張は出しません。
新入口は既存候補生成checkpointの機能を変更せず、独自resumeは提供しません（#35）。

2026-09-06時点の確認起点はmain `7c36c9323626a83fdb89dddccb93f7252930666e`、
tree `b1beddecd18464f8343b248dba2d624078fb4314`です。これは今後のlatestを固定する指定ではありません。
#19/#24/#25は完了済み。#50はdraftで、公開headとローカル修復候補の検証を混同できません。
未公開/安全確認で拒否されたパッチを、別経路でこの変更へ混ぜてはいけません。
#26/#28/#35等の予約branchと最新コメントを再確認し、同じファイルを並行編集しません。

| READMEの意図 | 既存の入口 | 次の受入作業 |
|---|---|---|
| 発話/正規化分離、音響を優先 | contracts、longform、integrity tests | #31/#34、#50のexact-head回帰、誤修正/反復保存 |
| 公開データから再現する | manifest、generate、post-candidate driver | #26権利/lineage、#28依存lock、#30全段階の完全再現 |
| 音素/モーラで候補を確かめる | phonetic evidence、既存回帰fixture | #33複数読み、#34実音声runtime、#29 gold/proxy分離 |
| 必要な重みを本当に学習する | training、real-weight pilot | #35 resume/reload、#36音響対照、#37同一候補LoRA対照 |
| 長時間の会議で役に立つ | document/longform opt-in | #38自然会話、#53音声grounding介入、Koemoの端末gate |
| 軽量で信頼できる公開物 | CPU profile、wheel/API、既存benchmarks | #39同一予算比較、#40権利/再現/主張の公開審査 |

この増分では#23/#30/#35等を閉じません。CI修復、ソフトウェア統合、実験、昇格を別PR・別状態にします。
Issue #30の全96判定については、8件のbundled replayと混同せず、公開を許可された完全な入力を使って
既存の`audit_public_decisions.py`を実行してください。入力SHA-256も必須です。

```bash
python scripts/audit_public_decisions.py /absolute/external/public-decisions.jsonl \
  --sha256 a29f5accb9f617c473d1ab9415b00cd1eecb2fdb806708091f24c72e3d3fc6da \
  --output ../semantic-asr-evidence/full96-replay.json
```

このhashは以前の96判定bundleのidentityです。ファイルが存在しない・一致しないときは取得経路を
確認し、別データのhashへ付け替えて既存結果の再現と呼ばないでください。過去判定の再生は新規推論ではありません。

## Codexへ渡す依頼文

```text
AGENTS.md、README.md、docs/development/CODEX_HANDOFF.mdを読み、READMEの目的を実装する。
まず実際のmain/作業branchのSHA/tree、担当Issueと予約、PR #50の公開状態を確認する。
python scripts/codex_pipeline.py plan を実行し、setup済みの別core/CPU環境でcheckを再現する。
実装は一つの未完Issueに絞り、最小再現→修正→否定テスト→全検証→証跡→PRまで進める。
不足するデータ/権利/モデル/runnerを列挙する。研究は明示許可・有限予算で一度だけ実行する。
raw音声/参照文/秘密を公開せず、拒否された書込みを迂回しない。研究defaultsを変更しない。
完了した公開SHA/tree、コマンド、失敗/skip、artifact hash、残るgateを報告する。
ローカル成功をGitHub CI成功、synthetic学習を実精度改善、PR作成をmerge完了と扱わない。
```

Codex CLIでは、設定済みの認証と通常の権限内で
`codex exec --sandbox workspace-write "AGENTS.mdとdocs/development/CODEX_HANDOFF.mdを読んで、担当Issueの最小修正と検証を進める"`
という一回の実行にも使えます。外部証跡先への書込みは、そのCodex環境で許可された場所を選びます。
許可を得るためにsandboxやexecpolicyを無効化しません。

一次資料（2026-09-06確認）:
[AGENTS.md](https://developers.openai.com/codex/guides/agents-md)、
[Cloud environments](https://developers.openai.com/codex/cloud/environments)、
[non-interactive mode](https://developers.openai.com/codex/noninteractive)。
これらはCodexの操作仕様の根拠であり、Semantic ASRの認識精度の根拠ではありません。
