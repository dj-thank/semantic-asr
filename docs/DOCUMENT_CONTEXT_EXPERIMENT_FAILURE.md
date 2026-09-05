# Document-context experiment validation failure

## Focused suite
```text
F.FF.........FFFFF..                                                     [100%]
=================================== FAILURES ===================================
________________ test_reference_is_not_part_of_planning_digest _________________

    def test_reference_is_not_part_of_planning_digest() -> None:
>       row = case()
              ^^^^^^

tests/test_document_experiment_protocol.py:52: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_protocol.py:36: in case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
___________ test_complete_reference_in_external_context_is_rejected ____________

    def test_complete_reference_in_external_context_is_rejected() -> None:
        leaked = FrozenExternalContext(
            name="leaked",
            context=DocumentContext(left_context=reference().text),
            provenance_sha256="d" * 64,
        )
    
>       with pytest.raises(ValueError, match="complete evaluation reference"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'complete evaluation reference'
E         Actual message: 'first-pass long-form evidence hash mismatch'

tests/test_document_experiment_protocol.py:99: AssertionError
_______________ test_manifest_rejects_speaker_or_session_leakage _______________

    def test_manifest_rejects_speaker_or_session_leakage() -> None:
>       row = case()
              ^^^^^^

tests/test_document_experiment_protocol.py:104: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_protocol.py:36: in case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
______ test_planner_receives_no_reference_and_candidates_are_frozen_once _______

    def test_planner_receives_no_reference_and_candidates_are_frozen_once() -> None:
        seen = []
    
        def planner(view):
            seen.append(view)
            assert not hasattr(view, "reference")
            return fake_plan()
    
>       prepared = prepare_document_experiment(manifest(), protocol(), planner)
                                               ^^^^^^^^^^

tests/test_document_experiment_runner.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
_______ test_all_arms_share_candidates_and_context_arm_improves_fixture ________

    def test_all_arms_share_candidates_and_context_arm_improves_fixture() -> None:
>       prepared = prepare_document_experiment(manifest(), protocol(), lambda view: fake_plan())
                                               ^^^^^^^^^^

tests/test_document_experiment_runner.py:148: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
_________ test_report_writes_canonical_evidence_with_negative_results __________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-0/test_report_writes_canonical_e0')

    def test_report_writes_canonical_evidence_with_negative_results(tmp_path: Path) -> None:
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:171: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
__________________ test_scored_character_budget_fails_closed ___________________

    def test_scored_character_budget_fails_closed() -> None:
        class BudgetBreaker(ToyScorer):
            def score_path(self, path, arm, **kwargs):
                row = super().score_path(path, arm, **kwargs)
                return DocumentLanguageScore(
                    value=row.value,
                    raw_average_log_likelihood=row.raw_average_log_likelihood,
                    forward_average_log_likelihood=row.forward_average_log_likelihood,
                    backward_average_log_likelihood=row.backward_average_log_likelihood,
                    source=row.source,
                    profile_digest=row.profile_digest,
                    path_digest=row.path_digest,
                    arm_digest=row.arm_digest,
                    scored_characters=1_000_000,
                    scorer_calls=row.scorer_calls,
                )
    
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:211: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
________________ test_candidate_set_digest_binds_planner_output ________________

    def test_candidate_set_digest_binds_planner_output() -> None:
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:229: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
=========================== short test summary info ============================
FAILED tests/test_document_experiment_protocol.py::test_reference_is_not_part_of_planning_digest - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_protocol.py::test_complete_reference_in_external_context_is_rejected - AssertionError: Regex pattern did not match.
  Expected regex: 'complete evaluation reference'
  Actual message: 'first-pass long-form evidence hash mismatch'
FAILED tests/test_document_experiment_protocol.py::test_manifest_rejects_speaker_or_session_leakage - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_planner_receives_no_reference_and_candidates_are_frozen_once - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_all_arms_share_candidates_and_context_arm_improves_fixture - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_report_writes_canonical_evidence_with_negative_results - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_scored_character_budget_fails_closed - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_candidate_set_digest_binds_planner_output - ValueError: first-pass long-form evidence hash mismatch
8 failed, 12 passed in 0.92s
```

## Full suite
```text
..s........sss.......................................................... [ 14%]
.....................FF..........F.FF.FFFFF............................. [ 28%]
........................................................................ [ 42%]
.....................................ssss............................... [ 56%]
..............................................................xxx....... [ 70%]
........................................................................ [ 84%]
........................................................................ [ 98%]
.......ss.                                                               [100%]
=================================== FAILURES ===================================
____________ test_promotion_rejects_gain_shared_by_shuffled_control ____________

    def test_promotion_rejects_gain_shared_by_shuffled_control() -> None:
        decision = evaluate_document_context_promotion(
>           report(shuffled_also_prefers_correction=True),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            policy(),
        )

tests/test_document_context_promotion.py:156: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_context_promotion.py:126: in report
    manifest = fixture_manifest()
               ^^^^^^^^^^^^^^^^^^
tests/test_document_context_promotion.py:72: in fixture_manifest
    case = DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
_ test_promotion_passes_when_ordered_arm_uniquely_improves_without_regression __

    def test_promotion_passes_when_ordered_arm_uniquely_improves_without_regression() -> None:
        decision = evaluate_document_context_promotion(
>           report(shuffled_also_prefers_correction=False),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            policy(),
        )

tests/test_document_context_promotion.py:168: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_context_promotion.py:126: in report
    manifest = fixture_manifest()
               ^^^^^^^^^^^^^^^^^^
tests/test_document_context_promotion.py:72: in fixture_manifest
    case = DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
________________ test_reference_is_not_part_of_planning_digest _________________

    def test_reference_is_not_part_of_planning_digest() -> None:
>       row = case()
              ^^^^^^

tests/test_document_experiment_protocol.py:52: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_protocol.py:36: in case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
___________ test_complete_reference_in_external_context_is_rejected ____________

    def test_complete_reference_in_external_context_is_rejected() -> None:
        leaked = FrozenExternalContext(
            name="leaked",
            context=DocumentContext(left_context=reference().text),
            provenance_sha256="d" * 64,
        )
    
>       with pytest.raises(ValueError, match="complete evaluation reference"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       AssertionError: Regex pattern did not match.
E         Expected regex: 'complete evaluation reference'
E         Actual message: 'first-pass long-form evidence hash mismatch'

tests/test_document_experiment_protocol.py:99: AssertionError
_______________ test_manifest_rejects_speaker_or_session_leakage _______________

    def test_manifest_rejects_speaker_or_session_leakage() -> None:
>       row = case()
              ^^^^^^

tests/test_document_experiment_protocol.py:104: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_protocol.py:36: in case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
______ test_planner_receives_no_reference_and_candidates_are_frozen_once _______

    def test_planner_receives_no_reference_and_candidates_are_frozen_once() -> None:
        seen = []
    
        def planner(view):
            seen.append(view)
            assert not hasattr(view, "reference")
            return fake_plan()
    
>       prepared = prepare_document_experiment(manifest(), protocol(), planner)
                                               ^^^^^^^^^^

tests/test_document_experiment_runner.py:139: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
_______ test_all_arms_share_candidates_and_context_arm_improves_fixture ________

    def test_all_arms_share_candidates_and_context_arm_improves_fixture() -> None:
>       prepared = prepare_document_experiment(manifest(), protocol(), lambda view: fake_plan())
                                               ^^^^^^^^^^

tests/test_document_experiment_runner.py:148: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
_________ test_report_writes_canonical_evidence_with_negative_results __________

tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-1/test_report_writes_canonical_e0')

    def test_report_writes_canonical_evidence_with_negative_results(tmp_path: Path) -> None:
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:171: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
__________________ test_scored_character_budget_fails_closed ___________________

    def test_scored_character_budget_fails_closed() -> None:
        class BudgetBreaker(ToyScorer):
            def score_path(self, path, arm, **kwargs):
                row = super().score_path(path, arm, **kwargs)
                return DocumentLanguageScore(
                    value=row.value,
                    raw_average_log_likelihood=row.raw_average_log_likelihood,
                    forward_average_log_likelihood=row.forward_average_log_likelihood,
                    backward_average_log_likelihood=row.backward_average_log_likelihood,
                    source=row.source,
                    profile_digest=row.profile_digest,
                    path_digest=row.path_digest,
                    arm_digest=row.arm_digest,
                    scored_characters=1_000_000,
                    scorer_calls=row.scorer_calls,
                )
    
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:211: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
________________ test_candidate_set_digest_binds_planner_output ________________

    def test_candidate_set_digest_binds_planner_output() -> None:
>       frozen_manifest = manifest()
                          ^^^^^^^^^^

tests/test_document_experiment_runner.py:229: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
tests/test_document_experiment_runner.py:88: in manifest
    cases=(experiment_case(),),
           ^^^^^^^^^^^^^^^^^
tests/test_document_experiment_runner.py:70: in experiment_case
    return DocumentExperimentCase(
<string>:14: in __init__
    ???
src/semantic_asr/document_experiment/protocol.py:170: in __post_init__
    self.first_pass.verify()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = LongformResult(source_name='fixture.wav', source_audio_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...a72451ac12426ec6d7396', diagnostics={'provisionalWindowCount': 0}, evidence_schema='semantic-asr-longform-evidence-v2')

    def verify(self) -> None:
        if self.evidence_schema != "semantic-asr-longform-evidence-v2":
            raise ValueError("unsupported long-form evidence schema")
        require_integer(self.duration_ms, name="duration_ms", minimum=1)
        if not self.segments:
            raise ValueError("long-form evidence requires segments")
        if len(self.source_audio_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_audio_sha256
        ):
            raise ValueError("source audio identity must be a SHA-256 digest")
        previous_start = -1
        for index, segment in enumerate(self.segments):
            window = segment.window
            if (
                window.index != index
                or window.start_ms <= previous_start
                or window.end_ms > self.duration_ms
            ):
                raise ValueError("long-form window sequence does not match the recording")
            previous_start = window.start_ms
            segment.observed.verify()
            if segment.observed.source_audio_sha256 != self.source_audio_sha256:
                raise ValueError("segment evidence belongs to a different source recording")
            segment.normalized.verify(segment.observed)
        if self.observed_text != join_segment_text(self.segments):
            raise ValueError("long-form observed text does not match its segments")
        if self.normalized_text != join_segment_text(self.segments, normalized=True):
            raise ValueError("long-form normalized text does not match its segments")
        if sha256_json(self.evidence_payload()) != self.evidence_sha256:
>           raise ValueError("first-pass long-form evidence hash mismatch")
E           ValueError: first-pass long-form evidence hash mismatch

src/semantic_asr/longform.py:176: ValueError
=========================== short test summary info ============================
SKIPPED [1] tests/test_acoustic_verifier_optional.py:5: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_audio_sample_integrity.py:13: could not import 'numpy': No module named 'numpy'
SKIPPED [1] tests/test_feature_padding.py:7: could not import 'numpy': No module named 'numpy'
SKIPPED [1] tests/test_training_optional.py:6: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_training_v2_optional.py:5: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_api.py:106: could not import 'numpy': No module named 'numpy'
SKIPPED [2] tests/test_api.py:234: could not import 'numpy': No module named 'numpy'
SKIPPED [1] tests/test_api.py:240: could not import 'numpy': No module named 'numpy'
SKIPPED [1] tests/test_mora_training_regressions.py:25: could not import 'torch': No module named 'torch'
SKIPPED [3] tests/test_mora_training_regressions.py:40: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_weight_pilot.py:49: could not import 'torch': No module named 'torch'
SKIPPED [1] tests/test_weight_pilot.py:80: could not import 'torch': No module named 'torch'
XFAIL tests/test_runtime_reliability.py::test_legacy_decode_request_integer_validation_gap[True] - Legacy DecodeRequest still accepts invalid beam sizes; legacy adapter repair deferred.
XFAIL tests/test_runtime_reliability.py::test_legacy_decode_request_integer_validation_gap[1.2] - Legacy DecodeRequest still accepts invalid beam sizes; legacy adapter repair deferred.
XFAIL tests/test_runtime_reliability.py::test_legacy_decode_request_integer_validation_gap[nan] - Legacy DecodeRequest still accepts invalid beam sizes; legacy adapter repair deferred.
FAILED tests/test_document_context_promotion.py::test_promotion_rejects_gain_shared_by_shuffled_control - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_context_promotion.py::test_promotion_passes_when_ordered_arm_uniquely_improves_without_regression - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_protocol.py::test_reference_is_not_part_of_planning_digest - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_protocol.py::test_complete_reference_in_external_context_is_rejected - AssertionError: Regex pattern did not match.
  Expected regex: 'complete evaluation reference'
  Actual message: 'first-pass long-form evidence hash mismatch'
FAILED tests/test_document_experiment_protocol.py::test_manifest_rejects_speaker_or_session_leakage - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_planner_receives_no_reference_and_candidates_are_frozen_once - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_all_arms_share_candidates_and_context_arm_improves_fixture - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_report_writes_canonical_evidence_with_negative_results - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_scored_character_budget_fails_closed - ValueError: first-pass long-form evidence hash mismatch
FAILED tests/test_document_experiment_runner.py::test_candidate_set_digest_binds_planner_output - ValueError: first-pass long-form evidence hash mismatch
10 failed, 491 passed, 15 skipped, 3 xfailed in 6.50s
```
