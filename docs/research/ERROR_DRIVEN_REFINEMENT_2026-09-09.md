# 誤り起点の局所改善 — 2026-09-09

Issue #21 の未完研究に属する、ローカルの有限実験。既定の認識経路やモデルは変更しない。
基点は `26d69600e2d859ddb2a36d023abb999b4ec17569`、tree は
`dc2969169bb01d39fc9e1e435e02270dc9d8c675`。

## 問題と変更

直前の実験用 hybrid は、旧 phone/context 選択が first pass を保持した場合に、
新 decoder の出力を無条件で使っていた。これが「密度→水道」などの誤修正を追加した。

`phonetic_refinement.guard_decoder_fallback` は既存の `PhoneContextCandidate` と
`PhoneContextDecision` を使い、修正前後の同一音声・全 window・posterior・profile
での証拠だけを比較する。欠損は first pass 保持、証拠混在は例外にする。

今回の実験候補は、音響尤度の悪化と文脈 preference の組合せ、同じ読みの候補に
対する文脈側の拒否を使う。弱い baseline 適合度では音響側の拒否を保留する。
尤度を correctness probability と呼ばない。G2P は読みの提案であり音声正解ではない。

`project_decoder_display_edits` は拒否した語を保持しつつ、限定した句読点・空白と
既存 ITN で同じ数値と確認できる表記だけを表示案へ投影する。raw first pass、
採点済み候補の decision、投影後の表示案は別々に保存する。表示案に元候補の
音響スコアを付け替えない。数値や ASCII token の結合・分割、小数点・符号・
引用境界・否定・実際の反復の削除を負例で確認する。

## 固定した開発結果

全224件は既に参照を見た **exposed regression**。このデータで仮説と閾値を選んだ
ので、独立テスト・一般化・統計的優越・promotion の証拠ではない。

| 対象 | 旧 hybrid の文字誤り | 今回の表示案 | 参照文字数 |
|---|---:|---:|---:|
| original96 | 273 | 269 | 5,386 |
| cycle2-64 | 143 | 137 | 3,324 |
| hybrid64 | 190 | 184 | 3,453 |
| 合計 | 606 | 590 | 12,163 |

表記差を除いた既存 lenient 指標は 396→381 / 11,358。
両指標とも8文で改善、悪化0文。変更は同点1文を含む9文であり、8文全体が完全に
正しくなったという意味ではない。例えば「打ち伸ばす」は保持できたが、同じ文の
「安打」「厚縁生ば」は残る。実音声を人間が通して聴いた gold review は未実施。

## 試行と反証

12仮説を段階ごとの有限予算で実行し、失敗・不採用結果も保存した。
単純な内容一致、句読点投影、読み一致、音響 veto だけでは、正しい表記や修正を
戻して全体が悪化した。次に G2P の `pau` を除いた別 profile を作り、句読点が
語の音響比較へ混入する影響を減らした。元の profile とスコアを混用していない。

開発で選んだ phone regression 許容量は `0.005`、baseline score floor は `-0.08`。
joint preference の言語重み `0.2` は以前の固定済み phone/context policy から再利用。
既定 API はこれらの開発値へ変更していない。候補は常に provisional。

独立レビューで「世宗王→セジュン王」が文字距離を減らしても内容を損なう疑いが
見つかった。音響・言語が対立するこの変更を保留する条件へ改め、最終候補は
「世宗王」を保持する。数字間の空白・句読点を削除すると数を結合する負例も
追加して修正した。集計値だけで受け入れなかった。

## 実行と証拠

- 変更した52音声、615.84秒だけを同一の既存 HuBERT / phone-trial3 で2回採点。
  第1回は従来の読み、第2回は pause-free の読み。追加の音声生成・学習は0。
- 元の Qwen3-ASR-1.7B decoder で52組・104候補の text-only preference を採点。
  音響証拠でも、生成器から独立したモデルでもない。
- 各モデル実行は最大20分、CPU6 threads。音響はRSS8GB、言語はRSS16GB。
  追加の paid resource、モデルダウンロード、外部送信・公開・既定化は0。
- 単体・境界テスト: `python -m pytest -q tests/test_decoder_fallback_guard.py tests/test_phonetic_refinement.py`。
- 全検証: `python scripts/codex_verify.py --profile installed`。結果・skip・xfail、
  wheel の checkout 外確認、最終 source identity は成果物 receipt を参照する。

タスク内の実験ソースと段階別 freeze は `work/error-loop-20260909/`。
最終再生は `replay_stage7.py`、集計は `stage7-measurement.json`。
利用者向けの結果・音声比較・hash manifest は `outputs/error-loop-20260909/`。
これらはローカルの無視対象であり、私的パスや音声を Git へ含めない。

次の対象は、残った語の誤認識について既存 ASR 間の局所候補を作り、編集した文全体を
音響・文脈で再採点する実験。今回の全 window 候補を、未検証の部分音声 decode で
置き換えない。次の改善も単体再現→固定した全件回帰→内容点検の順で行う。
