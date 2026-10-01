# 音声区間の再認識から新しい候補を作る（既定 OFF）

## 目的と今回の範囲

候補の選び直しでは解けない「正解が候補集合にない」問題に対し、
**明示された音声区間の再認識 → 前後を保持した全文候補 → 独立検証済み候補の既存ラティス接続**
を追加する。既存の production / realtime / longform 経路は変更しない。

関連: [Issue #69](https://github.com/dj-thank/semantic-asr/issues/69)、
[PR #70](https://github.com/dj-thank/semantic-asr/pull/70)。
#70 の未マージコード・既存研究ブランチは取り込んでいない。
基点: `fdf2dbb6d77beb6311855229b1439d964be1c8bf`、
基点 tree: `6a6cf59112866554018003d0f1ebd6ee7bff79db`。
本変更はこれらの研究 Issue を完了扱いにせず、精度向上や本番昇格も主張しない。

## 探索して採用した設計

全文候補をさらに選び直すだけでは、集合に存在しない語は出せない。
一方、短い再認識結果を全文候補として扱うと、前後の発話を失う。
LLM に全文の書き直しを任せる方法では、自然さが音響証拠と混同される。
今回は、変更を一つの明示された文字・音声区間に限定する方式を採用した。

例（ソフトウェアテスト用の作例。実音声の新規実測ではない）:

```text
初回: えー、歯垢原品、歯垢原品を売る、いや売らない。
部分:      手工芸品
候補: えー、手工芸品、歯垢原品を売る、いや売らない。
```

二度目の同じ語、フィラー、反復、言い直し、否定はそのまま残る。
文字列の全置換や `find()` による曖昧な対応付けは行わない。
文字位置は Python Unicode code-point 単位であり、バイト位置ではない。

## 実装の入口

- `semantic_asr.span_candidates`: モデル非依存の `SpanWindow` / `SpanAnchor` /
  `SpanDraft` と `generate_span_drafts()`。出力は**未検証の提案**であり、
  `CandidateEvidence` や `ObservedTranscript` ではない。
- `semantic_asr.span_candidate_runtime.run_span_redecode()`: 既存の
  `DecodeRequest` / `ASRAdapter` を使った区間再認識。`enabled=False` が既定。
- `span_verification_request()`: 元文と新候補を同じ**親ウィンドウ全体**で
  独立音響検証するための allowlist。正解文・自然さ評価・ファイルパス・部分尤度は渡さない。
- `build_verified_redecode_lattice()`: 既存の `VerifiedSpanProposal` と
  `build_semantic_deliberation_lattice()` を再利用する接続部分。
  元候補一つに対する**補助ラティス**を作る。既存の複数候補・文書ラティスを置換しない。

独立 verifier 自体の実装・学習・実行はこの変更に含めない。verifier は元文と
新候補を同じ親音声区間で比較し、適用可能な既存校正を使って canonical な
`VerifiedSpanProposal` を返す必要がある。提案生成の成功だけでは採用しない。

## 呼出しの流れ

1. 既存 first-pass の候補 ID・原文・親区間・録音ファイル SHA-256 で
   `SpanWindow` を作る。対象音声の利用権限は既存研究手順で確認する。
2. 音声と文字の対応を確認した alignment の manifest SHA-256 と、
   文字位置・録音上の絶対ミリ秒位置から `SpanAnchor` を作る。
3. 同じ親区間の canonical `DecodeRequest`、既存 adapter、model/tokenizer/
   runtime/config を含む `decoder_fingerprint` を渡して `run_span_redecode(...,
   enabled=True)` を呼ぶ。結果の `texts` は元文が先頭の未検証候補集合。
4. 独立 verifier で `span_verification_request(expansion)` を処理する。
   verifier が abstain した候補を receipts に入れない。
5. verifier が返した receipts だけを `build_verified_redecode_lattice()` に渡す。
   新候補を挿入するには元文の比較可能な receipt も必要。
   挿入後の選択・保留は既存 deliberation policy の責務であり、この API は決定しない。

receipt metadata は `spanVerificationRequestSha256` に、上記 request の
canonical `sha256_json()` 値を、`verificationScope` に `whole-window` を持つ。
ID は request の `retained` または `draft:<digest>` と完全一致させる。
source recording、全文、候補集合、時間区間を確認し、未知の ID・異なる文・
部分音声のスコアの流用・元文の検証欠如を拒否する。

この接続は各候補一つの独立音響 channel のみを受け付け、比較候補間で
channel / source / calibration profile を一致させる。
phone とそこから得た mora の二重加点や、文脈スコアの付加を行わない。
**receipt の型と hash が正しくても、verifier が本当に音声を評価した証明にはならない。**
verifier の信頼性・音声入力の binding・校正適用性は既存の検証契約が必要。

## 境界と予算

既存ラティスには文字数比例の時刻 fallback があるため、それを自動的に
切り出し位置へ変換しない。alignment digest は出所を記録するだけで、
alignment 自体の正しさを保証しない。前後に音声の余白を足す場合は、その
余白に対応する文字まで同じ置換区間へ含める。音声だけ padding しない。

一つの提案は一つの区間だけを変更する。重複・重なりのある区間も独立した
候補として扱い、逐次置換による offset のずれや組合せ爆発を避ける。
同一の全文候補は重複排除し、独立した支持票として数えない。

既定の上限は 4 呼出し、各 5 仮説、16 新候補、要求音声合計 20 秒、
置換テキスト 256 文字。入力順が処理優先順位となる。
失敗した呼出しも呼出し数・音声予算を消費する。未実行・失敗・候補なしを
分離し、例外の私的パスや本文をログへコピーしない。
これは壁時計時間の上限ではない。timeout・プロセス分離は adapter 側で設定する。

初回の prompt/hotwords は crop へ引き継がない。再認識前後に録音ファイル
SHA-256 を照合する。これは録音のバイト identity であり、抽出 PCM の hash や
認識モデルが実際にその音声を用いたことの証明ではない。
部分のスコア・token IDs を新しい全文へ付け替えない。元の decoder evidence と
実行 request は hash として provenance に保持し、元文の証拠は変更しない。

## 検証と未完了項目

モデル不要のテストで、候補外の語の追加、前後保持、同語の二度目だけの変更、
Unicode、重複、各予算、無限 iterator、途中例外、source/window の不一致を確認する。
canonical integration テストは実際の `DecodeRequest` / `CandidateEvidence` /
`VerifiedSpanProposal` / ラティス実装を使用するが、adapter と音響 utility は
**合成 fixture**。音声認識精度の実測ではない。

```bash
python -m pytest -q tests/test_span_candidates.py tests/test_span_candidate_runtime.py
python -m ruff format --check src tests scripts
python -m ruff check src tests scripts
python -m compileall -q src tests scripts
python -m pytest -q
```

同一予算の候補生成比較、候補 oracle CER、正解候補 coverage、最終 CER、
false correction、RTF、speaker/source-disjoint の評価が次の実音声受入条件。
候補選択だけでなく候補生成が改善したかを分離して測る。既に見た #70 の
音声を未見の評価集合へ戻さない。新規推論・追加学習・自動昇格・自動マージは未実施。
