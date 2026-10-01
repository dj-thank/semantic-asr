# 候補不足に対する、音声からの有限な再探索

Issue #71。実装基点: `fdf2dbb6d77beb6311855229b1439d964be1c8bf`、
tree `6a6cf59112866554018003d0f1ebd6ee7bff79db`。

## 目的と根拠

「候補の中から選ぶ」だけでなく、**選択肢そのものが足りないとき、音声から探し直す**。
LLMで自然な文に書き換えたり、新しいモデル名だけを追加したりする変更ではない。

既存の `plan_evidence` はgateがconfidentなら停止する。矛盾区間がない場合も、
unscored primaryの特別なsecond-ear経路以外では追加認識を計画しない。
スコアを持つ候補が1つでも、その集合内のposteriorは1.0、entropyは0になりうる。
**候補集合内で1位であることと、正解が候補集合に含まれることは別問題。**

既存の根拠は [draft PR #70](https://github.com/dj-thank/semantic-asr/pull/70)。
96件中57件が候補1つ、baseline 210/4,630 errors、固定候補集合のoracle 170 errors、
exact CTC 204 errors。公開FLEURSのdevelopment / exposed-regression結果であり、
今回の変更の測定ではない。候補1つの57件すべてが誤認識という意味でもない。
信頼区間はゼロを含み、一般的な精度向上は確立していない。
固定候補の完全な選択でも減らせるのは最大40 errors。残る170 errorsを減らすには、
候補生成側を改善する必要がある。

## 実装

```text
一次認識 → 重複・句読点などを除いた表記の種類数
  → 実質1種類かつopt-inなら、通常のconfidence判定より前に再探索
      → 予算内のsecond earを優先
      → 不在/計画時の予算不適合なら、より広いprimary探索
      → 最大1回、元と同じ音声窓全体を認識
  → 元候補を残して既存のscore-domain-safe mergeとfusionで比較
  → provisionalのobservedと、別レイヤーのnormalized
```

`SemanticASRTranscriber(..., expand_collapsed_candidates=True)` で有効にする。
既定はFalse。名前付きプロファイル、モデル、重み、公開CLIの既定は変更しない。
今回はcore transcriberのオプションであり、全呼び出し側の有効化ではない。

既存の `lenient_surface_key` を再利用する。これはNFKC・空白/句読点/記号非依存の
表記クラスであり、音素の独立性、同音語、正解被覆率の推定ではない。
候補が2種類以上あって両方誤っているケースは、このsliceでは検出しない。

primary fallbackはbeamとhypothesesの両方が一次認識以上、少なくとも片方が大きい
場合だけ許可する。同じ探索を繰り返して候補拡張と呼ばない。
second earも独立性を保証するものではない。同系統のモデルは相関しうるため、
出典を `secondary-decoder` として残す。

## 使用例

事前に権利・revisionを確認して準備したローカルadapterと音声を使う。
この例はモデルや音声の取得・外部送信を許可するものではない。

```python
from semantic_asr.longform import SemanticASRTranscriber
from semantic_asr.planner import EvidenceBudget

# primaryは固定revisionの既存ASRAdapter。secondaryは準備済みの第二ASRまたはNone。
engine = SemanticASRTranscriber(
    primary,
    second_ear=secondary,
    beam_size=5,
    hypotheses=5,
    relisten_beam_size=12,
    relisten_hypotheses=8,
    evidence_budget=EvidenceBudget(total_cost_ms=12_000, max_actions=1),
    expand_collapsed_candidates=True,
)
result = engine.transcribe("approved-local-audio.wav")
for segment in result.segments:
    print(segment.observed.text, segment.observed.decision)
    print(segment.diagnostics["candidateExpansion"])
```

## 予算と証拠の境界

各窓で追加探索は最大1回。既存EvidenceExecutionが実行・cache hit・失敗を区別する。
追加候補が同じ、backend失敗、空候補の場合に自動再試行はしない。
collapsed窓ではbalanced routerを迂回し、`single-expansion-limit` を記録する。
それ以外の窓の既存routerは変更しない。

costとgainはヒューリスティック。gain 0.5は正解確率でも実測の期待改善でもない。
予算は各窓のoptional呼び出しに対するもので、一次認識・モデルロード・音声全体の
総時間・メモリの上限ではない。同期呼び出しを途中停止するhard deadlineはなく、
実測超過の記録を実時間上限の保証とは呼ばない。

元と同じ窓の音声を使い、部分認識の短い文字列で全文を置き換えない。
CandidateEvidence、モデル/窓/decoderの出典、cache、score domainを再利用する。
既存executorがbackend/空候補失敗を記録し、元候補を残す。
成功と候補集合の拡大は別。cache hitを新しい実推論として数えない。

再探索対象の出力は、追加成功・候補不変・予算不足・backend失敗のいずれもprovisional。
過去のcorrectness calibrationを新policyへ引き継がず、推測した候補外確率も追加しない。
`initialDistinctSurfaceCount` / `finalDistinctSurfaceCount` / `addedDistinctSurfaceCount`
は表記クラス数であって正解が増えた件数ではない。
`attempted` / `completed` / `reason` は既存の実行receiptと対応させる。

## 検証と次の実験

`tests/test_candidate_expansion.py` は合成候補・メモリ内decoderの契約テスト。
実音声の精度、Whisper/Reazon実backendの速度、独立ASRの効果の測定ではない。
変更前にsingleton posterior=1.0 / no-relistenを再現し、opt-inテストの失敗を記録した。

実音声比較では既存の5-role lineageとpaired evaluationを使い、音源・話者を分け、
設定・予算を先に固定する。既に見た音声を未見評価とは呼ばない。
比較条件はbaseline、常時wide beam、collapsed-only wide beam、collapsed-only second ear。
候補集合のoracle CER/正解被覆と、実際の選択CERを分離する。否定/数値/固有語、
誤修正、保留、p50/p95遅延、実呼び出し数を併記し、費用を揃えて比較する。

新しい実音声・モデル・課金API・学習は今回実行していない。
PR #70を変更/mergeせず、既定化やモデル昇格もしない。
