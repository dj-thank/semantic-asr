# Semantic ASR

**発話された日本語を、意味を勝手に直さず、音響証拠・モーラ・複数ASR・局所的な意味リスクから復元する。**

Semantic ASRは、日本語音声を単に「自然な文章」へ変換する後処理ツールではありません。Whisper系N-best、decoder path、モーラ影表現、独立ASR、学習済みreranker、語彙証拠、フィラー・言い直しの保存証拠を融合し、**実際に話された内容**と**読みやすく整えた文章**を別の証拠オブジェクトとして保存する、ローカル優先の日本語音声認識基盤です。

```text
observedTranscript != normalizedTranscript
```

発話者が実際に「昨日、学校を行きました」と話した場合、言語モデルが「昨日、学校に行きました」を好んでも、前者を消しません。

## v0.2の中心

```text
audio
  │
  ├─ path-preserving faster-whisper / CTranslate2 N-best
  ├─ score-domain-safe surface pooling
  ├─ Semantic Minimum Bayes Risk
  ├─ adaptive candidate K
  ├─ optional linear / ModernBERT / Qwen3 reranker
  ├─ held-out calibration
  ├─ constrained evidence fusion
  ├─ fusion–MBR agreement check
  └─ uncertainty-only acoustic verification / second ear
          │
          ▼
immutable observed transcript
          │
          └─ separately linked normalized transcript
```

v0.2で追加した主な機構:

- 同一表記へ到達した複数beamを捨てず、同一score domain内だけ`logsumexp`で確率質量を集約
- raw score、log likelihood、logit、preference、calibrated probabilityを型として分離
- surface/mora/数字・日時・金額・否定・固有表現・保存性を扱うSemantic MBR
- posterior mass、risk、semantic criticality、diversityによるadaptive K
- 依存なしで学習できるpairwise linear ranker
- optional ModernBERT/CrossEncoder raw-logit ranker
- optional Qwen3-Reranker-0.6B raw yes/no logit margin
- candidate moraが必要な音声frameを選ぶ小型Query-Selected Acoustic Verifier
- evidence actionの固定化を防ぐquantile-balanced sparse router
- 8B/12B級offline teacherのnext-token probabilityを端末で再利用するhashed cache
- 日本語hard negative生成と、CI内で実際に最適化されるresearch smoke training

詳細:

- [`docs/ARCHITECTURE_V0.2.md`](docs/ARCHITECTURE_V0.2.md)
- [`docs/RESEARCH_2026-08-31.md`](docs/RESEARCH_2026-08-31.md)
- [`docs/RERANKER_TRAINING.md`](docs/RERANKER_TRAINING.md)
- [`docs/KOEMO_INTEGRATION.md`](docs/KOEMO_INTEGRATION.md)
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)
- [`docs/DISCRETE_UNIT_EVIDENCE.md`](docs/DISCRETE_UNIT_EVIDENCE.md)

## 設計原則

- 文法的な自然さより、音声への忠実性を優先する
- 「ない／ある」「2人／3人」「3,000円／30,000円」の意味反転を局所検出する
- 重いモデルは曖昧な区間だけに投入する
- 自信不足は誤った確定ではなく`provisional`として残す
- フィラー、言い直し、学習者誤りを勝手に消さない
- LLMがN-best外の文を作る場合は、新候補として音響再検証する
- 評価、権利、モデル版、score semantics、校正状態を再現可能にする
- CPU tierが勝てば、より大きいLLMを採用しない

## アーキテクチャ

### Candidate path pool

CTranslate2が返すduplicate textを最良path一つに潰しません。

```text
path 1 ─┐
path 3 ─┼─→ 同じ表記 → logsumexp(path scores)
path 7 ─┘
```

ただし異なるモデル、区間、prompt、decode namespaceのscoreを一つの正規化分布として足しません。異種モデルの一致は`cross_model`証拠として別に保存します。

### Score semantics

次は互いに別物です。

```text
raw score
log likelihood
logit
preference
probability
```

チャットLLMがJSONに`0.9`と書いても、それは校正済み確率ではありません。専用rerankerもraw logitを返します。自動受理やrisk計算で確率として使うには、held-out calibrationが必要です。

### Semantic MBR

候補`y`の期待損失をN-best posterior上で最小化します。

```text
R(y) = Σ_h P(h | x) L(y, h)
```

既定loss:

```text
surface edit
mora edit
number/date/time/currency/negation/entity loss
preservation disagreement
```

FusionとMBRRが不一致なら、既定ではMBRへ即座に置き換えず、追加音響証拠または`provisional`を要求します。

### Adaptive K

固定top-5ではなく、候補分布が平坦、riskが高い、重要語が矛盾する場合にKを増やします。十分なposterior massを回収した時点、またはtailの追加質量が小さい時点で停止します。

### Learned reranker

三つのtierがあります。

```text
CPU最小      pairwise linear ranker
CPU/小GPU    Japanese CrossEncoder / ModernBERT
小GPU        Qwen3-Reranker-0.6B
```

すべて候補ランキング用です。観測文字列を生成する権限はありません。

### Query-Selected Acoustic Verifier

候補モーラ列をqueryとして、候補判定に必要な音声frameを選びます。

```text
candidate mora query
        ├─ selected acoustic branch
        ├─ global context branch
        └─ mora-internal branch
                │
          bounded learned gates
                │
          candidate ranking logit
```

これはQwen3.8のsparse selection、Kimi K3のAttention Residuals/高疎MoE、GLM-5.3のmHCをASR証拠処理へ翻訳した研究設計です。各モデルの内部kernelやweightsを複製したという主張ではありません。

### Discrete-unit pronunciation evidence（research-only）

音声を固定SSL encoder／固定codebookの離散unit列へ変換し、native token LMのsurprisalを再検証ルーティングへ、同じcodebookでText2DUnitが予測した標準unit列とのcentroid DTWを候補別の音響証拠へ使う実験kernelを追加しました。audio-only surprisalは同一発話の全候補で共通なので候補rankingには加えません。zero-shot scoreは`-normalized_centroid_DTW`だけです。日本語CER改善は未計測で、既定では無効です。

詳細な同一codebook条件、artifact固定、計算量guard、評価matrixは[`docs/DISCRETE_UNIT_EVIDENCE.md`](docs/DISCRETE_UNIT_EVIDENCE.md)を参照してください。

### Grammar Honeytrap

言語モデルが自然だと判断しても、音響・モーラ・独立ASRの支持が弱ければペナルティを与えます。言語priorを音響証拠として扱いません。

### Observed / normalized split

```text
observed transcript
  acoustic candidateから選択
  evidence hashで改変検知

normalized transcript
  observed evidence hashへリンク
  deterministic / rank-only / guarded-rewrite
```

## インストール

Python 3.11以上を使用します。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[asr]'
```

Qwen3-ASR／Forced Aligner:

```bash
python -m pip install -e '.[asr,qwen]'
```

公開 Hugging Face データセットからローカル検証用 WAV/manifest を作る場合:

```bash
python -m pip install -e '.[public-data]'
```

CrossEncoder／Qwen3 reranker:

```bash
python -m pip install -e '.[asr,rerank]'
```

PyTorch acoustic verifier／補助ヘッド:

```bash
python -m pip install -e '.[train]'
```

基本packageは依存ゼロです。

## モデルなしで試す

v0.1互換デモ:

```bash
semantic-asr demo
```

v0.2の候補cascade:

```bash
semantic-asr cascade examples/candidates.json \
  --selection-policy fusion \
  --output runs/cascade.json
```

実際に小型rankerを学習するsynthetic smoke:

```bash
semantic-asr research-smoke --output runs/research-smoke.json
```

これは学習・MBR・adaptive Kのcode pathを検証しますが、実音声CER改善の証拠ではありません。

## 最短で使う（一呼び出し API）

```python
from semantic_asr import transcribe

result = transcribe("meeting.wav", profile="cpu-ja-v1")
print(result.observed_text)      # 実際に話された内容（改変検知ハッシュ付き）
print(result.normalized_text)    # 読みやすく整えた別レイヤー
for segment in result.segments:  # window 単位の区間と判定状態
    print(segment.start_seconds, segment.end_seconds, segment.status, segment.observed)
result.write("transcripts")      # json / observed.txt / txt / md / srt / vtt
```

`profile` は不変の名前付き設定です（`cpu-ja-v1`: CPU int8 large-v3-turbo・beam 5・30 秒 window padding・loop guard。`cpu-ja-quality-v1`: beam 12。`gpu-ja-v1`: CUDA float16）。バックエンドのノブは呼び出しに漏らさず、組み合わせが変わるときは新しい profile を足します。Koemo のような既存呼び出し側には `transcribe_segments(audio)` が `[(start, end, text), ...]` を返します。モデルを温めたまま何度も呼ぶ場合は `load_transcriber(profile)` を一度作り、`transcribe(..., transcriber=warm)` に渡してください。

### 固有名詞を安全に補助する（ContextCatalog）

固有名詞や製品名は、音声評価より前に凍結した JSON カタログから検索し、一致した語だけを
decoder hotword にできます。空クエリ・不一致は `abstained` となり、カタログ全体を無条件に
注入しません。監査情報にはカタログの digest、query hash、entry ID、phrase hash、score を
残し、生の phrase や query は残しません。ID 自体が機微なら、不透明な ID を使います。

```python
from semantic_asr import ContextCatalog, transcribe

catalog = ContextCatalog.from_json("examples/context_catalog.example.json")
result = transcribe(
    "meeting.wav",
    catalog=catalog,
    context_query="森脇さんとSemantic ASRの進捗確認",
    context_tags=("person",),
)
```

CLI では `--catalog`、`--context-query`、必要なら繰り返し可能な `--context-tag` を使います。
詳細と評価上の禁止事項は [`docs/CONTEXT_CATALOG.md`](docs/CONTEXT_CATALOG.md) を参照してください。

## 実音声で動かす（2026-09-02 以降）

公開テストセットから権利注記付き manifest を作り、N-best 候補を生成し、分割・学習・較正・ベンチマークまでを CLI で回します。公開データ extra は `datasets`、`numpy`、`scipy`、`soundfile` を含みます。

WAV、参照文、絶対 `audioPath` を含む manifest の materialization は明示的なローカル操作です。出力先は checkout の外に置き、`--allow-raw-export` を必ず付けてください（`data/reazon` など checkout 内の出力は拒否されます）。

```bash
python scripts/prepare_public_manifest.py reazonspeech-test --output-dir ../semantic-asr-public-data/reazon --limit 600 --dataset-revision dd08bfb9dfc1cef4e4d0609fd78c3755d48b926f --allow-raw-export
semantic-asr generate-candidates ../semantic-asr-public-data/reazon/manifest.jsonl   --output ../semantic-asr-public-data/reazon/all-candidates.jsonl   --ranker-output ../semantic-asr-public-data/reazon/all-ranker.jsonl   --allow-raw-export   --model large-v3-turbo --model-revision 0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf   --runtime-revision "$(git rev-parse HEAD)+faster-whisper-1.2.1+ctranslate2-4.8.2"   --device cpu --compute-type int8 --cpu-threads 6   --beam-size 12 --hypotheses 12
python scripts/run_real_audio_pipeline.py   --candidates "../semantic-asr-public-data/reazon/all-candidates.jsonl"   --output-dir ../semantic-asr-public-data/reazon/pipeline   --allow-raw-export
```

`generate-candidates` の候補 JSONL と post-candidate pipeline の成果物は参照文・仮説を
含むため、`runs/` など checkout 内には保存しません。pipeline は reference-bearing input を
処理するとき `--allow-raw-export`（local-research authorization）を要求し、各行の
`rightsDecision=allow` と `license`/`licenseId`（生成物では `generation.licenseId`）を確認します。
`review`、`deny`、権利情報の欠落は停止します。出力先は symlink 解決後も checkout 外で、
filesystem root ではない必要があります。

Qwen3-ASR の second-ear probe は既定では参照文・仮説を出力しません。確認用の JSONL も checkout 外へ置き、参照文・仮説が必要なローカル研究時だけ `--local-research-output` を明示します。

```bash
python scripts/probe_second_ear.py ../semantic-asr-public-data/reazon/manifest.jsonl --output ../semantic-asr-public-data/reazon/second-ear.jsonl
python scripts/probe_second_ear.py ../semantic-asr-public-data/reazon/manifest.jsonl --output ../semantic-asr-public-data/reazon/second-ear-local.jsonl --local-research-output
```

`generate-candidates` は既定で loop guard（30 秒 window への padding、duration 依存の token 上限、圧縮率・反復 n-gram・文字数予算による degenerate 判定、温度 fallback）を有効にします。`--no-loop-guard` で v0.2 の挙動に戻せます。`--extra-samples N` でサンプル候補を追加し、sample-based MBR を試せます。

既知の標準モデル・公開データは immutable Hugging Face commit に固定されます。別モデルや別データでは exact 40 文字 commit を明示してください。`modelRevision` は実際の loader に渡され、metadata だけ異なる値を名乗る生成は停止します。境界診断は report 専用で、candidate 選択や primary strict CER には接続されません。

長い生成は `<output>.partial` に1発話ずつ flush します。中断後に同じ manifest・model revision・generation config で再実行すると、audio/config/provenance を検証した完全なprefixだけを再利用します。プロセス停止で最後の未終端行だけが破損した場合はその行を切り戻し、改行だけ欠けた完全行は補修しますが、改行済みの破損行は拒否します。不一致があれば上書きや暗黙再開をせず停止し、全件完了後にだけ final JSONL へ atomic promotion します。

逐語的な日本語評価では、`spoken_reference_surface` が `(F ...)` / `(D ...)` などの注釈wrapperだけを外し、フィラーや言い直しの発話内容を保持します。`filler_event_score` はフィラーを別指標として測り、曖昧・匿名化spanがある参照は exact CER を fail-closed で未定義にします。読みやすさのためのフィラー削除は normalized transcript 側だけで行い、observed transcript の成功には数えません。

測定記録と発見した欠陥は [`docs/RESEARCH_2026-09-02.md`](docs/RESEARCH_2026-09-02.md) を参照してください。
現状からの技術選定、deep Module の seam、モデル／context／confidence／edge の優先順位は [`docs/ARCHITECTURE_ROADMAP_2026-09-02.md`](docs/ARCHITECTURE_ROADMAP_2026-09-02.md) にまとめています。

## v0.2長時間文字起こし

Path-preserving N-bestだけを使う:

```bash
semantic-asr transcribe-v2 meeting.m4a \
  --model large-v3-turbo \
  --maximum-hypotheses 12 \
  --patience 1.4 \
  --language ja \
  --output-dir transcripts
```

学習済みlinear ranker:

```bash
semantic-asr transcribe-v2 meeting.wav \
  --ranker-backend linear \
  --ranker-profile artifacts/linear-ranker.json \
  --ranker-calibration calibration/ranker.json \
  --output-dir transcripts
```

Japanese CrossEncoder:

```bash
semantic-asr transcribe-v2 meeting.wav \
  --ranker-backend cross-encoder \
  --ranker-model sbintuitions/modernbert-ja-130m \
  --ranker-device cpu \
  --output-dir transcripts
```

モデルによってsequence-classification headの追加学習が必要です。汎用masked-LM checkpointを未学習のままproduction rankerとはみなしません。

Qwen3 reranker:

```bash
semantic-asr transcribe-v2 meeting.wav \
  --ranker-backend qwen3 \
  --ranker-model Qwen/Qwen3-Reranker-0.6B \
  --ranker-device cuda:0 \
  --ranker-dtype float16 \
  --output-dir transcripts
```

Qwen3-ASR second ear／Forced Aligner:

```bash
semantic-asr transcribe-v2 meeting.wav \
  --qwen-second-ear \
  --qwen-aligner \
  --qwen-model Qwen/Qwen3-ASR-0.6B \
  --qwen-aligner-model Qwen/Qwen3-ForcedAligner-0.6B \
  --output-dir transcripts
```

## Ranker datasetと学習

参照文を1行ずつ書いた`references.txt`からhard negativeを作る:

```bash
semantic-asr synthetic-data references.txt \
  --output data/synthetic-ranker.jsonl \
  --maximum-negatives 8
```

軽量rankerを学習:

```bash
semantic-asr train-ranker data/train.jsonl \
  --output artifacts/linear-ranker.json \
  --epochs 80
```

実運用ではsynthetic dataだけで学習せず、実ASR N-best、speaker-disjoint calibration、locked testを使用します。

## Offline teacher probability cache

実際のteacher logitsから得たnext-token probabilityを、端末向けcacheへ変換できます。

```bash
semantic-asr lm-cache-build teacher-probabilities.jsonl \
  --output artifacts/teacher-cache.json \
  --key-hex <deployment-local-secret-in-hex> \
  --teacher local-12b \
  --teacher-revision <exact-revision>
```

cacheはraw contextを保存せず、keyed SHA-256 context digest、target token ID、probability、teacher provenanceだけを保持します。秘密keyはrepositoryへcommitしません。

## Legacy local teacher

v0.1互換のOllama／OpenAI-compatible teacherも残しています。

```bash
semantic-asr transcribe-v2 interview.wav \
  --teacher-protocol ollama \
  --teacher-model qwen3:4b
```

このteacherがJSONに出力する数値は、candidate-only rank preferenceとして扱います。専用rerankerのraw logitやheld-out calibrated probabilityと同一視しません。teacherはobserved transcriptを直接変更できず、rank-only normalized layerまたは追加証拠要求にのみ使われます。

安全境界:

- loopback HTTPのみ
- URL認証情報・query・fragmentを拒否
- environment proxyを無効化
- redirectを拒否
- candidate ID完全一致を検証
- candidate外のobserved text生成を禁止
- abstentionをcacheでも維持

## 出力

```text
meeting.semantic-asr.json  全証拠・候補・校正・不確実性・計画
meeting.observed.txt       発話されたと判断した改変検知対象
meeting.txt                別レイヤーの読みやすい文章
meeting.md                 監査情報とtimeline
meeting.srt                字幕
meeting.vtt                Web字幕
```

絶対入力pathは既定で出力しません。

## 評価

```text
recognition:
  CER / Kana-CER / Mora Error Rate / oracle@K / rank regret

meaning-critical:
  number / date / time / currency / negation / entity errors

preservation:
  filler / repair / learner-error preservation
  unsupported insertion / unsupported correction

confidence:
  ECE / Brier / NLL / AURC / risk-coverage

cost:
  RTF / p50-p95 latency / RAM / VRAM
  verifier / second-ear / teacher invocation rate
```

必須ablationは[`docs/RESEARCH_2026-08-31.md`](docs/RESEARCH_2026-08-31.md)に固定しています。

## 公開データと権利

公開されていることと、学習・派生特徴・再配布が許されることは同義ではありません。`allow / deny / review`を操作ごとに持つrights registryを実行時に検査します。manifest preparation の権利既定値は `review` で、リポジトリが exact revision まで明示的に対応した公開 asset だけが自動的に `allow` になります。その他の asset は明示的な `--rights-decision allow` と利用者の確認が必要です。

```bash
semantic-asr rights data/rights_registry.example.json \
  jmdict-current derive_features
```

`prepare_public_manifest.py` に `--rights-registry` を渡すと、`derive_features` と `redistribute_raw` の両方が `allow` であることを既存 registry に要求します。`review` は「たぶん使える」ではなく処理停止です。生成 WAV、参照文、絶対パスを含む成果物は公開せず、checkout 外の local-research ディレクトリでのみ扱ってください。`.gitignore` にも代表的な誤出力先を記載していますが、ignore は権利確認や公開許可の代わりではありません。

## Koemo

Koemoはcapture/AEC/live/UI/model lifecycleを担当し、Semantic ASRをauthoritative final transcript coreとしてpackage利用します。Koemoのregex correctionはnormalized layerにのみ適用し、observed evidence作成前のraw ASR candidateを変更しません。

移行契約は[`docs/KOEMO_INTEGRATION.md`](docs/KOEMO_INTEGRATION.md)を参照してください。

## 検証

```bash
python -m compileall -q src tests scripts
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
semantic-asr research-smoke --output runs/research-smoke.json
python -m build --wheel
```

CPU PyTorch検証:

```bash
python -m pytest -q \
  tests/test_training_optional.py \
  tests/test_acoustic_verifier_optional.py
```

CIはmodel-free test、synthetic ranker optimization、Linux/Windows、Python 3.11/3.12、CPU verifier backward、wheel clean installを分離して実行します。

## Claim boundary

コード契約、synthetic training、CI成功は、実音声精度改善とは別です。権利確認済み日本語音声と正解transcriptで測定するまでは、次を主張しません。

- Whisperより必ず高精度
- CERが特定割合改善
- 100%正しい文字起こし
- 特定GPUで必ず一定VRAM以内
- Qwen/Kimi/GLMの内部kernelを再実装した
- synthetic fixtureで学んだrankerがproduction-readyである

## ライセンス

Apache License 2.0。詳細は[`LICENSE`](LICENSE)を参照してください。

## Codexで開発と検証を引き継ぐ

READMEの意図と未完了の実音声自動パイプラインを、
[Codex引き継ぎ手順](docs/development/CODEX_AUTOPILOT.md)に結び付けています。
`AGENTS.md`とrepo skillに加え、CodexとCIが同じ検証入口を使います。

```bash
python -m pip install -e '.[dev]'
python scripts/codex_verify.py --plan
python scripts/codex_verify.py --output-dir ../semantic-asr-evidence/run-001
```

format/lint・テスト・保存判定replay・synthetic学習・隔離wheel build・checkout外の
wheel実行までを順に検証し、失敗時に停止してsourceと結果のhashを記録します。
これは新規実音声の推論・本学習・精度改善の実測とは別です。実音声cycleのstage、
権利・分割・freeze・有限budgetの受入条件とCodexへの実装依頼は引き継ぎ手順を参照してください。
