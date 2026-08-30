# Semantic ASR

**発話された日本語を、意味を勝手に直さず、音響証拠・モーラ・複数ASR・局所的な意味リスクから復元する。**

Semantic ASRは、日本語音声を単に「自然な文章」へ変換する後処理ツールではありません。Whisper系N-best、モーラ影表現、独立ASR、語彙証拠、フィラー・言い直しの保存証拠を融合し、**実際に話された内容**と**読みやすく整えた文章**を別の証拠オブジェクトとして保存する、ローカル優先の日本語音声認識基盤です。

```text
observedTranscript != normalizedTranscript
```

発話者が実際に「昨日、学校を行きました」と話した場合、言語モデルが「昨日、学校に行きました」を好んでも、前者を消しません。

## 目標

- 文法的な自然さではなく、音声への忠実性を最優先する
- 「ない／ある」「2人／3人」「3,000円／30,000円」のような意味反転を局所検出する
- 曖昧な区間だけを再聴取し、重いモデルを全音声へ常時投入しない
- 自信が不足するときは、誤った確定ではなく`provisional`として棄権する
- 発音誤り、助詞誤り、フィラー、自己修復を評価可能な形で残す
- Qwen3-ASRやローカルQwen3.8を、音響証拠に逆らえない補助者として使う
- 評価、データ権利、キャッシュ、モデル版、校正状態を再現可能にする

## アーキテクチャ

```text
audio / video
  │
  ├─ long-form window planner + cache
  │
  ├─ faster-whisper / CTranslate2 N-best
  │      ├─ sequence score
  │      ├─ average log probability
  │      ├─ rank / hypothesis count
  │      └─ prompt / hotword provenance
  │
  ├─ Mora Shadow
  │      ├─ reading
  │      ├─ mora CTC
  │      ├─ phone / boundary / accent / F0
  │      └─ error-preservation evidence
  │
  ├─ independent second ear: Qwen3-ASR
  ├─ optional Qwen3 Forced Aligner
  └─ rights-gated lexical memory
          │
          ▼
calibration
  ├─ held-out temperature profile
  ├─ robust median/MAD fallback
  └─ beam score-rank confidence
          │
          ▼
five-stream evidence fusion
  ├─ acoustic
  ├─ mora
  ├─ lexical
  ├─ preservation
  └─ cross-model consensus
          │
          ├─ posterior
          ├─ entropy
          ├─ evidence disagreement
          ├─ evidence coverage
          ├─ selective risk
          └─ accepted / provisional
          │
          ▼
Semantic Lattice
  ├─ Consensus Spine
  └─ Contradiction Islands
          ├─ number / quantity
          ├─ date / time
          ├─ currency / percentage
          ├─ negation meaning flip
          ├─ modality / intent
          ├─ entity / technical term
          ├─ particle
          └─ special mora
          │
          ▼
query-selected evidence acquisition
  utility = expected information gain / estimated cost
  ├─ Whisper local re-listening
  ├─ Qwen3-ASR second ear
  ├─ Qwen3 Forced Aligner
  ├─ rights-gated lexicon lookup
  └─ local Qwen teacher
          │
          ▼
immutable observed transcript
          │
          └─ separate normalized transcript
```

詳細は[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)を参照してください。

## 主要な独自機構

### Semantic Contradiction Islands

候補文全体を一つのランキングとして扱わず、候補が食い違う箇所だけを局所化します。

```text
候補A: 明日は行きません。料金は3,000円です。
候補B: 明日は行きます。料金は30,000円です。
```

この場合、`negation-meaning-flip`、`number-or-quantity`、`currency`として高リスク化し、通常の句読点差より先に追加証拠を取得します。

### Mora Shadow

`今日`と`きょう`のような表記差は、読みが同じならモーララティス上で一致できます。反対に、促音・撥音・長音・拗音など本当に音が違う箇所は矛盾として残ります。

```text
きゃく   → キャ / ク
がっこう → ガ / ッ / コ / ウ
しんぶん → シ / ン / ブ / ン
スーパー → ス / ー / パ / ー
ティ     → ティ
ファイル → ファ / イ / ル
```

### Grammar Honeytrap

ローカルLLMが強く好む候補でも、音響＋モーラ＋独立ASRの支持が弱ければペナルティを与えます。LLMの文法知識を「音響スコア」として扱いません。

### Selective Risk / Abstention

候補の事後確率、証拠ストリーム間の不一致、欠落証拠、上位候補差から選択リスクを求めます。閾値を満たさなければ、結果を捨てるのではなく`provisional`として保存し、再聴取候補を提示します。

### Query-Selected Evidence

追加推論は、情報利得と推論コストの比で選択します。これはQwen3.8-Flash-Nextの効率的な状態選択・ゲート設計から着想した**音声オーケストレーション上の翻訳**であり、QSAやGated DeltaNetの内部カーネルを再実装したという主張ではありません。

## インストール

Python 3.11以上を使用します。

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[asr]'
```

Qwen3-ASR／Forced Alignerを使用する場合:

```bash
python -m pip install -e '.[asr,qwen]'
```

補助ヘッドを学習する場合:

```bash
python -m pip install -e '.[train]'
```

## まず試す

モデルを使わない決定論的デモ:

```bash
semantic-asr demo
```

既存N-bestを融合:

```bash
semantic-asr fuse examples/candidates.json --output runs/fused.json
```

## 完全な長時間文字起こし

```bash
semantic-asr transcribe meeting.m4a \
  --model large-v3-turbo \
  --language ja \
  --output-dir transcripts
```

固有名詞と文脈を渡す:

```bash
semantic-asr transcribe meeting.wav \
  --initial-prompt '日本語音声認識と生成AIの技術会議です。' \
  --hotwords 'Semantic ASR,Qwen3-ASR,Qwen3.8,CTranslate2,モーラ' \
  --context '発話誤りとフィラーを消さない。' \
  --output-dir transcripts
```

### GTX 1660 SUPERなど低VRAM環境

```bash
semantic-asr transcribe meeting.wav \
  --model small \
  --device cuda \
  --compute-type int8_float16 \
  --output-dir transcripts
```

CUDAランタイムが使えない場合:

```bash
semantic-asr transcribe meeting.wav \
  --model small \
  --device cpu \
  --compute-type int8 \
  --output-dir transcripts
```

ハードウェア、ドライバー、モデル版によりVRAMは変わるため、特定GPUで必ず収まるとは断定していません。

## Qwen3-ASRを第二の耳として使う

```bash
semantic-asr transcribe meeting.wav \
  --qwen-second-ear \
  --qwen-model Qwen/Qwen3-ASR-0.6B \
  --qwen-device-map cuda:0 \
  --qwen-dtype float16 \
  --qwen-timestamps \
  --output-dir transcripts
```

Qwen3-ASRは、原則として曖昧なContradiction Islandだけに投入します。公式高レベルAPIが1入力につき1文字起こしを返す経路を、Whisper N-bestと同じものとして偽装しません。

Forced Aligner:

```bash
semantic-asr transcribe meeting.wav \
  --qwen-aligner \
  --qwen-aligner-model Qwen/Qwen3-ForcedAligner-0.6B
```

Forced Alignmentは候補の時刻配置証拠であり、その候補が実際に発話されたことを単独で証明するものではありません。

## ローカルLLM

### Ollama

```bash
semantic-asr transcribe interview.wav \
  --teacher-protocol ollama \
  --teacher-model qwen3:4b \
  --teacher-endpoint http://127.0.0.1:11434/api/chat
```

### ローカルQwen3.8 OpenAI互換サーバー

```bash
semantic-asr transcribe interview.wav \
  --teacher-protocol openai \
  --teacher-model Qwen/Qwen3.8-Flash-Next \
  --teacher-endpoint http://127.0.0.1:8000/v1/chat/completions
```

教師は既存候補IDへの確率と`abstain`だけを返せます。新しい文字起こしの生成、候補追加、観測文字列の上書きは禁止です。

安全境界:

- ループバックHTTPのみ
- URL認証情報・クエリ・フラグメントを拒否
- 環境変数プロキシを無効化
- HTTPリダイレクトを拒否
- 候補IDの完全一致を検証
- 思考過程を保存しない
- 棄権状態をキャッシュでも維持
- 教師結果は`observedTranscript`を直接変更できない

## 出力

入力が`meeting.m4a`の場合:

```text
meeting.semantic-asr.json  全証拠・候補・校正・不確実性・行動計画
meeting.observed.txt       実際に発話されたと判断した日本語
meeting.txt                読みやすさ用の別レイヤー
meeting.md                 監査情報とタイムライン
meeting.srt                字幕
meeting.vtt                Web字幕
```

絶対入力パスは既定で出力しません。

## 校正

校正用JSONL:

```json
{"confidence": 0.73, "correct": true}
```

```bash
semantic-asr calibrate heldout.jsonl \
  --output calibration/observed-posterior.json
```

出力:

- ECE
- Brier score
- Negative Log-Likelihood
- Risk-Coverage Curve
- 校正プロファイルSHA-256

詳細は[`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)を参照してください。

## 評価

通常のCERだけでなく、次を重視します。

```text
CER
Kana-CER
Mora Error Rate
number/date/time/currency error rate
negation error rate
critical-entity error rate
punctuation F1
filler / disfluency preservation
unsupported correction rate
ECE / Brier / NLL / AURC
coverage at fixed risk
RTF / peak VRAM / cache hit rate
information gain per additional inference millisecond
```

最低限のアブレーション:

```text
Whisper single-best
Whisper N-best
+ calibrated fusion
+ Mora Shadow
+ semantic contradiction islands
+ selective re-listening
+ Qwen second ear
+ local teacher
full system
```

## 公開データ

公開されていることと、学習・派生特徴・再配布が許されることは同義ではありません。`allow / deny / review`を操作ごとに持つ権利台帳を実行時に検査します。

```bash
semantic-asr rights data/rights_registry.example.json \
  jmdict-current derive_features
```

`review`は「たぶん使える」ではなく処理停止です。詳細は[`docs/PUBLIC_DATA.md`](docs/PUBLIC_DATA.md)を参照してください。

## 学習用補助ヘッド

共有音声エンコーダーへ、次を追加できます。

```text
mora CTC
phone CTC
mora boundary
accent class
F0 regression
preservation class
```

保存分類の想定ラベル:

```text
ordinary
filler
repair / self-correction
learner error
```

通常のWhisperデコーダーを最初から置き換えず、日本語固有の音響監督信号を追加する設計です。

## 検証

```bash
python -m compileall -q src tests scripts
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m pytest -q
semantic-asr demo --output runs/demo.json
python -m build --wheel
```

現在のmodel-freeテストは、モーラ、改変検出、校正、融合、Semantic Lattice、証拠計画、キャッシュ、長時間処理、出力、教師安全境界、評価指標、PyTorch補助ヘッドを検証します。

基本環境ではPyTorchを入れず、補助ヘッドのテストは意図的にskipします。別のCPU環境へ
`torch>=2.4`を導入し、`python -m pytest -q tests/test_training_optional.py`で実行してください。
CIはLinux/Python 3.11・3.12、Windows/Python 3.12、CPU補助ヘッドを分けて検証し、
ソースの自動整形・commit・pushは行いません。実音声やモデル重みはこの検証に不要です。

### 主張しないもの

実モデルと公開・権利確認済み正解コーパスで測定するまでは、次を主張しません。

- Whisperより必ず高精度
- CERが特定割合改善
- 100%正しい文字起こし
- 特定GPUで必ず一定VRAM以内
- Qwen3.8の内部アーキテクチャを音声モデルへ直接実装済み

コード契約の成功と、実音声認識精度の改善は別物です。

## 研究根拠

2026年8月29日時点の一次資料と、直接実装・発想の翻訳・未検証仮説の境界を[`docs/RESEARCH_2026-08-29.md`](docs/RESEARCH_2026-08-29.md)へ固定しています。

## ライセンス

Apache License 2.0。詳細は[`LICENSE`](LICENSE)を参照してください。
