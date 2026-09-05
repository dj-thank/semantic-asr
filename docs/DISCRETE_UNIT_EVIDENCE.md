# Discrete-unit surprisal / centroid-DTW evidence

Status: **research-only / disabled by default**

Primary source:

- Syeda Faiza Ahmed Sara and Shammur Absar Chowdhury, “Light-weight Pronunciation
  Assessment via Discrete Speech Token Surprisal,” INTERSPEECH 2026,
  `arXiv:2606.19910v2`.

## 目的

論文の離散音声トークン機構を、Semantic ASRの候補選択にそのまま「正解判定器」として
持ち込むのではなく、次の二つの証拠へ分離して利用する。

1. **audio-only surprisal**
   - 凍結したSSL encoder・K-means codebookで音声を離散単位化する。
   - native speechだけで学習したtoken n-gram LMから、frame-rateのsurprisalを計算する。
   - surprisal標準偏差・native 90 percentile超過率・unit数を、不確実性や再検証の
     ルーティング特徴として使う。
2. **candidate-specific centroid DTW**
   - 各ASR候補テキストを、同じcodebook空間のcanonical unit列へ変換する。
   - 観測音声unit列とcanonical unit列を連続重複除去した上でDTW整列する。
   - K-means centroid間のL2距離を局所コストにし、path長で正規化したコストを候補固有の
     音響整合性として扱う。

```text
observed audio
    │
    ├─ frozen Audio2DUnit ── raw unit sequence ── native token LM
    │                                             └─ surprisal profile
    │                                                   └─ routing only
    │
ASR candidate text ── frozen Text2DUnit ── canonical units
    │                                      │
    └──────────────────────────────────────┼─ centroid-distance DTW
                                           └─ candidate-specific cost
```

## Semantic ASRでの境界

### audio-only surprisalは候補順位に直接入れない

同じ発話から得たaudio-only surprisalは、N-bestの全候補で同一である。そのため、これを
各候補のscoreへ足しても候補間の識別情報にはならない。実装では
`SurprisalProfile.as_uncertainty_evidence()`を候補非依存の未校正証拠として返し、再聴取、
second-ear、acoustic verifierを呼ぶかどうかの特徴に限定する。

### zero-shot候補順位は `-DTW distance`

論文の英語発音評価では、教師なしの単独特徴としてcentroid-DTW distanceが最も強かった。
そのため`DiscreteUnitAcousticRanker`はnative token LMなしでも動作し、候補固有の
zero-shot rank scoreを次だけに限定する。

```text
alignment_cost = normalized centroid-DTW distance   # COST, lower is better
rank_score     = -alignment_cost                    # UNCALIBRATED_SCORE
```

native token LMを与えた場合だけ、mismatch rate、mismatch-surprisal std、
weighted-surprisal stdも記録する。ただし固定係数で勝手に混合しない。日本語のlocked
calibration splitで学習・校正したprofileが得られるまで、自動受理用の確率として扱わない。

### observed transcriptを書き換えない

このbranchは既存候補をrankするだけである。Text2DUnitが生成したunit列や、DTWから新しい
文字列を生成しない。N-best外の文字列を提案する別stageを将来追加する場合も、既存の
acoustic re-verification gateを通す。

## 凍結artifact contract

次の全artifactは、同一の`DiscreteUnitSpace.digest`へ結び付ける。

- SSL encoder identifierとimmutable revision
- encoder layer
- sample rate
- language/domain label
- K-means codebook size
- codebook SHA-256
- native token LM
- centroid distance matrix
- Text2DUnit adapter
- observed/canonical unit sequences

layer、codebook、revisionのどれか一つでも異なる場合は例外にし、似たtoken IDを同じ意味と
見なさない。Text2DUnitの出力語彙とAudio2DUnitのcodebookが偶然同じサイズでも、digestが
異なれば使用できない。さらにText2DUnit出力の`source_sha256`は入力候補文のUTF-8 SHA-256と
一致しなければならず、別候補の古いunit列や誤ったadapter応答をscoreへ混入させない。

`DiscreteTokenLanguageModel`、`SurprisalThreshold`、`CentroidDistanceTable`は、内部digestと
envelope digestを含むJSON artifactとして`save()` / `load()`できる。読み込み時には
count/contextの整合性、schema version、serialized keyの一意性、matrixの
対称性・対角ゼロ、unit-space identity、SHA-256を再検証し、pickleや暗黙のruntime stateへ
依存しない。`SurprisalProfile`もtoken列・thresholdから再計算したmean、standard deviation、
spike rateと一致しなければ拒否する。

## 論文との対応

| 論文の機構 | 実装 | 扱い |
|---|---|---|
| frozen SSL + K-means | `DiscreteUnitSpace`, `DiscreteUnitSequence` | 外部adapterからunit列を受け取る |
| 3-gram native TLM | `DiscreteTokenLanguageModel` | dependency-free backoff n-gram |
| token surprisal | `token_surprisals()` | `-log2 P(unit | history)` bits |
| 90 percentile spike threshold | `fit_spike_threshold()` | frozen native calibration setから再計算 |
| consecutive deduplication | `DiscreteUnitSequence.collapse()` | raw frameへの逆写像を保持 |
| centroid L2 matrix | `CentroidDistanceTable` | symmetric・finite・zero diagonalを検証 |
| DTW / path normalization | `align_collapsed_units()` | 最小costの中で最短pathを選ぶdeterministic tie-break、path長正規化 |
| LMなしDTW特徴 | `centroid_dtw_features()` | zero-shot候補順位に利用可能 |
| mismatch surprisal std | `transcript_guided_features()` | mismatchに投影されたraw frameで算出 |
| weighted surprisal std | 同上 | `S_i × (1 + alpha × distance_i)` |
| Ridge feature fusion | `pronunciation_feature_vector()` | raw featureだけを出力。学習は別artifact |

## 意図的に固定しなかったもの

論文ではSpeechOcean762上のablationからHuBERT base layer 9、`K=512`を採用している。
これは英語発音評価で得られた選択であり、日本語ASRに対する最適値ではない。そのため
Semantic ASRでは、encoder、layer、codebook sizeを既定の日本語production profileへ
埋め込まない。日本語native corpusとlocked test splitで比較してからprofile名を追加する。

論文中の9.0-bit spike thresholdも固定値として移植しない。thresholdは、実際に利用する
凍結native unit corpusの90 percentileから算出し、そのcorpus digest、LM digest、unit-space
digestをreceiptへ残す。

DTW pathで一つのobserved unitが複数canonical unitへ対応する場合、論文本文ではraw frameへ
戻す局所距離の集約規則が一意に記述されていない。実装は`mean`を既定とし、`maximum`も
明示選択できる。選択値は`DTWConfig.digest`へ含める。

同じ最小total costを持つDTW pathが複数ある場合、局所的な方向優先だけで選ぶと、入力を
入れ替えた際にpath長が変わり、path長で正規化したdistanceまで変わり得る。実装は
`total cost`を第一目的、`path length`を第二目的として最短の最適pathを選び、それでも同率の
場合だけdiagonal、canonical-step、observed-stepの順で決定する。この最適化規則も
`DTWConfig.digest`へ含める。

## 最小利用例

以下は数理・contractのテスト用であり、学習済みText2DUnitの代替ではない。

```python
from semantic_asr.contracts import CandidateEvidence
from semantic_asr.discrete_unit_evidence import (
    CentroidDistanceTable,
    DiscreteTokenLanguageModel,
    DiscreteUnitAcousticRanker,
    DiscreteUnitSequence,
    DiscreteUnitSpace,
    StaticTextToDiscreteUnitEncoder,
)

space = DiscreteUnitSpace(
    encoder="org/hubert-ja",
    encoder_revision="<immutable revision>",
    layer=9,
    codebook_size=512,
    codebook_sha256="<64 hex characters>",
    language="ja",
)

native_sequences = (...)  # Audio2DUnitで事前生成した凍結native units
observed = DiscreteUnitSequence(units=(...), space=space)
centroid_table = CentroidDistanceTable.from_centroids(space, centroids)
token_lm = DiscreteTokenLanguageModel.fit(native_sequences, order=3)

text2unit = StaticTextToDiscreteUnitEncoder(
    {"候補A": (...), "候補B": (...)},
    space=space,
    revision="fixture-v1",
)
ranker = DiscreteUnitAcousticRanker(
    observed=observed,
    distance_table=centroid_table,
    text_encoder=text2unit,
    token_lm=token_lm,  # 省略するとDTW-only。与えると追加surprisal特徴を記録
)

token_lm.save("artifacts/native-token-lm.json")
threshold = token_lm.fit_spike_threshold(native_sequences)
threshold.save("artifacts/surprisal-threshold.json")
centroid_table.save("artifacts/centroid-distances.json")

scores = ranker.score(
    (
        CandidateEvidence(candidate_id="a", text="候補A"),
        CandidateEvidence(candidate_id="b", text="候補B"),
    )
)
```

production利用では`StaticTextToDiscreteUnitEncoder`を使わず、native Japanese
`(transcript, collapsed Audio2DUnit tokens)`で学習し、revisionとconfiguration digestを
固定したadapterを実装する。token LMを省略してもcentroid-DTWだけで候補順位を計算できる。
token LMを与えた場合だけmismatch-surprisal系の特徴が記録されるが、zero-shot順位は変わらない。
完全なsynthetic実行例は
[`examples/discrete_unit_evidence.py`](../examples/discrete_unit_evidence.py)を参照する。

## 実装ファイル

- `discrete_units.py`: unit-space identity、raw/collapsed sequence、native token LM、surprisal
- `discrete_unit_alignment.py`: centroid matrix、bounded DTW、raw-frame投影、特徴抽出
- `discrete_unit_ranker.py`: same-codebook Text2DUnit contractとCandidateRanker adapter
- `discrete_unit_evidence.py`: public re-export surface

core計算は標準ライブラリだけで動作する。SSL encoder、K-means学習、Text2DUnitモデルの
ロードは、モデルrevision・artifact digest・権利条件を呼び出し側で固定する。

## 評価計画

### locked arms

同一audio manifest、同一N-best、同一decode cacheで少なくとも次を比較する。

1. ASR path posteriorのみ
2. 既存Semantic MBR
3. audio-only surprisalをroutingだけに追加
4. candidate centroid-DTW ranker
5. DTW + mismatch featuresをheld-out linear/ridge profileで融合
6. DTW + 既存mora/acoustic verifier

### negative controls

- codebook IDをランダム置換したdistance table
- audioとText2DUnitで異なるcodebookを使用した入力（実行拒否されること）
- unrelated transcript units
- homophone / long-vowel / geminate / moraic nasalのhard negatives
- 数字、否定、金額、日時、固有名詞のsemantic-critical pairs
- noise、reverberation、silence、code-switch、長時間発話
- native calibration dataとtest speakerの重複検査

### 報告指標

- CERとutterance-mean CER
- semantic-critical error rate
- oracle@Kとcandidate recall
- candidate-ranking pairwise accuracy
- calibration後のECE、Brier、AURC
- abstention coverage / selective risk
- RTF、peak memory、追加stage invocation rate
- artifact mismatch / resource guardのfail-closed率

採用条件は、locked testでCERまたはsemantic-critical errorを改善し、selective riskを悪化させず、
追加costが対象profileのbudget内に収まることとする。論文のPCCをSemantic ASRのCER改善へ
読み替えない。

## 現時点の非主張

- 著者実装のbit-exact reproductionではない。
- SpeechOcean762 / L2-ARCTICの結果を再現したとは主張しない。
- HuBERT layer 9や`K=512`が日本語で最適とは主張しない。
- discrete surprisalが日本語CERを改善したとはまだ主張しない。
- Text2DUnit由来のcanonical unitを校正済み音響確率とは扱わない。

この実装が提供するのは、論文の特徴計算を再現可能なartifact contractの下で検証するための
基盤である。
