"""Complete-path decoding with finite factors and hard acoustic-retention guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import sha256_json
from .deliberation_evidence import (
    AUDIO_CHANNELS,
    GENERATED_ORIGINS,
    BoundedUtility,
    DecisionStatus,
    ResolutionMode,
    UtilityChannel,
    _is_sha256,
    _strict_float,
)
from .deliberation_lattice import (
    DeliberationLattice,
    DeliberationSpan,
    DocumentContext,
    LatticeArc,
    path_digest,
)
from .global_scorer import GlobalPathScore, GlobalSequenceScorer


@dataclass(frozen=True, slots=True)
class DeliberationPolicy:
    channel_weights: tuple[tuple[UtilityChannel, float], ...]
    beam_size: int = 64
    global_context_weight: float = 1.0
    retention_bonus: float = 0.02
    maximum_span_audio_regression: float = 0.20
    maximum_mean_audio_regression: float = 0.10
    minimum_final_margin: float = 0.02
    require_independent_audio_for_generated: bool = True
    provisional_on_generated: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.beam_size, bool) or self.beam_size < 1:
            raise ValueError("beam_size must be positive")
        numeric = {
            "global_context_weight": self.global_context_weight,
            "retention_bonus": self.retention_bonus,
            "maximum_span_audio_regression": self.maximum_span_audio_regression,
            "maximum_mean_audio_regression": self.maximum_mean_audio_regression,
            "minimum_final_margin": self.minimum_final_margin,
        }
        normalized_numeric = {
            name: _strict_float(value, name=name) for name, value in numeric.items()
        }
        if normalized_numeric["global_context_weight"] < 0:
            raise ValueError("global_context_weight must be non-negative")
        if normalized_numeric["maximum_span_audio_regression"] < 0:
            raise ValueError("maximum_span_audio_regression must be non-negative")
        if normalized_numeric["maximum_mean_audio_regression"] < 0:
            raise ValueError("maximum_mean_audio_regression must be non-negative")
        if normalized_numeric["minimum_final_margin"] < 0:
            raise ValueError("minimum_final_margin must be non-negative")
        weights: list[tuple[UtilityChannel, float]] = []
        seen: set[str] = set()
        valid_channels = {
            "first_pass",
            "asr_acoustic",
            "phone",
            "mora",
            "mora_shadow",
            "discrete_unit",
            "lexical",
            "preservation",
            "cross_model",
            "semantic",
            "transition",
        }
        for channel, value in self.channel_weights:
            if channel not in valid_channels:
                raise ValueError(f"unknown utility channel weight: {channel}")
            if channel in seen:
                raise ValueError("channel weights must be unique")
            seen.add(channel)
            weight = _strict_float(value, name=f"weight for {channel}")
            if weight < 0:
                raise ValueError("channel weights must be non-negative")
            weights.append((channel, weight))
        if not weights or all(value == 0 for _, value in weights):
            raise ValueError("at least one positive channel weight is required")
        object.__setattr__(self, "channel_weights", tuple(sorted(weights)))
        for name, value in normalized_numeric.items():
            object.__setattr__(self, name, value)

    @classmethod
    def conservative_default(cls) -> DeliberationPolicy:
        return cls(
            channel_weights=(
                ("first_pass", 0.85),
                ("asr_acoustic", 1.0),
                ("phone", 0.8),
                ("mora", 0.8),
                ("mora_shadow", 0.25),
                ("discrete_unit", 0.5),
                ("lexical", 0.15),
                ("preservation", 0.5),
                ("cross_model", 0.45),
                ("semantic", 0.25),
                ("transition", 0.35),
            )
        )

    @property
    def weights(self) -> dict[UtilityChannel, float]:
        return dict(self.channel_weights)

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "channelWeights": self.channel_weights,
                "beamSize": self.beam_size,
                "globalContextWeight": self.global_context_weight,
                "retentionBonus": self.retention_bonus,
                "maximumSpanAudioRegression": self.maximum_span_audio_regression,
                "maximumMeanAudioRegression": self.maximum_mean_audio_regression,
                "minimumFinalMargin": self.minimum_final_margin,
                "requireIndependentAudioForGenerated": (
                    self.require_independent_audio_for_generated
                ),
                "provisionalOnGenerated": self.provisional_on_generated,
                "factorSemantics": "utility-value-times-factor-weight-v1",
            }
        )


@dataclass(frozen=True, slots=True)
class PathHypothesis:
    arcs: tuple[LatticeArc, ...]
    base_score: float
    mean_audio_support: float
    context_score: float = 0.0
    final_score: float = 0.0
    context_source: str | None = None
    context_profile_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.arcs:
            raise ValueError("path hypothesis requires at least one arc")
        if len({arc.span_id for arc in self.arcs}) != len(self.arcs):
            raise ValueError("path hypothesis may select only one arc per span")
        if not self.text:
            raise ValueError("a complete deliberation path must not be empty")
        for name in (
            "base_score",
            "mean_audio_support",
            "context_score",
            "final_score",
        ):
            _strict_float(getattr(self, name), name=name)
        if not -1.0 <= self.mean_audio_support <= 1.0:
            raise ValueError("mean_audio_support must be in [-1, 1]")
        if not -1.0 <= self.context_score <= 1.0:
            raise ValueError("context_score must be in [-1, 1]")
        if (self.context_source is None) != (self.context_profile_digest is None):
            raise ValueError("context source and profile digest must be supplied together")
        if self.context_profile_digest is not None and not _is_sha256(self.context_profile_digest):
            raise ValueError("context_profile_digest must be a SHA-256 value")

    @property
    def digest(self) -> str:
        return path_digest(self.arcs)

    @property
    def text(self) -> str:
        return "".join(arc.text for arc in self.arcs)


@dataclass(frozen=True, slots=True)
class SpanResolution:
    span_id: str
    retained_arc_id: str
    selected_arc_id: str
    mode: ResolutionMode
    retained_audio_support: float | None
    selected_audio_support: float | None


@dataclass(frozen=True, slots=True)
class GlobalDeliberationDecision:
    selected: PathHypothesis
    retained: PathHypothesis
    alternatives: tuple[PathHypothesis, ...]
    status: DecisionStatus
    margin: float
    resolutions: tuple[SpanResolution, ...]
    reasons: tuple[str, ...]
    lattice_digest: str
    policy_digest: str
    context_digest: str
    scorer_source: str | None = None
    scorer_profile_digest: str | None = None

    def __post_init__(self) -> None:
        margin = _strict_float(self.margin, name="decision margin")
        if margin < 0.0:
            raise ValueError("decision margin must be non-negative")
        if (self.scorer_source is None) != (self.scorer_profile_digest is None):
            raise ValueError("decision scorer source and profile must be supplied together")
        if self.scorer_profile_digest is not None and not _is_sha256(self.scorer_profile_digest):
            raise ValueError("scorer_profile_digest must be a SHA-256 value")
        object.__setattr__(self, "margin", margin)

    @property
    def observed_text(self) -> str:
        return self.selected.text

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "selectedPathDigest": self.selected.digest,
                "retainedPathDigest": self.retained.digest,
                "alternativePathDigests": [path.digest for path in self.alternatives],
                "selectedText": self.selected.text,
                "status": self.status,
                "margin": self.margin,
                "resolutions": self.resolutions,
                "reasons": self.reasons,
                "latticeDigest": self.lattice_digest,
                "policyDigest": self.policy_digest,
                "contextDigest": self.context_digest,
                "scorerSource": self.scorer_source,
                "scorerProfileDigest": self.scorer_profile_digest,
            }
        )


def _weighted_arc_score(arc: LatticeArc, weights: Mapping[UtilityChannel, float]) -> float:
    return sum(
        weights.get(utility.channel, 0.0) * utility.weighted_value for utility in arc.utilities
    )


def _arc_factor_weight(arc: LatticeArc) -> float:
    return max((utility.factor_weight for utility in arc.utilities), default=0.0)


def _audio_support(
    arc: LatticeArc,
    weights: Mapping[UtilityChannel, float],
) -> float | None:
    rows = [
        (
            weights.get(utility.channel, 0.0) * utility.factor_weight,
            utility.value,
        )
        for utility in arc.utilities
        if utility.channel in AUDIO_CHANNELS
        and weights.get(utility.channel, 0.0) > 0
        and utility.factor_weight > 0
    ]
    if not rows:
        return None
    total_weight = sum(weight for weight, _ in rows)
    return sum(weight * value for weight, value in rows) / total_weight


def _mean_audio_support(
    path: Sequence[LatticeArc],
    weights: Mapping[UtilityChannel, float],
) -> float:
    rows = [
        (_arc_factor_weight(arc), value)
        for arc in path
        if (value := _audio_support(arc, weights)) is not None and _arc_factor_weight(arc) > 0
    ]
    if not rows:
        return -1.0
    total = sum(factor for factor, _ in rows)
    return sum(factor * value for factor, value in rows) / total


def _eligible_arcs(
    span: DeliberationSpan,
    *,
    policy: DeliberationPolicy,
) -> tuple[LatticeArc, ...]:
    weights = policy.weights
    retained = span.retained_arc
    retained_audio = _audio_support(retained, weights)
    output: list[LatticeArc] = []
    for arc in span.arcs:
        if not arc.observed_eligible:
            continue
        if (
            policy.require_independent_audio_for_generated
            and arc.origin in GENERATED_ORIGINS
            and not arc.independent_audio_channels
        ):
            continue
        candidate_audio = _audio_support(arc, weights)
        if arc.arc_id != retained.arc_id and retained_audio is not None:
            if candidate_audio is None:
                continue
            if retained_audio - candidate_audio > policy.maximum_span_audio_regression:
                continue
        output.append(arc)
    if retained.arc_id not in {arc.arc_id for arc in output}:
        output.append(retained)
    return tuple(sorted(output, key=lambda arc: arc.arc_id))


def _transition_map(
    lattice: DeliberationLattice,
) -> dict[tuple[str, str], BoundedUtility]:
    return {(row.left_arc_id, row.right_arc_id): row.utility for row in lattice.transitions}


def _transition_score(
    left: LatticeArc,
    right: LatticeArc,
    transitions: Mapping[tuple[str, str], BoundedUtility],
    weights: Mapping[UtilityChannel, float],
) -> float:
    utility = transitions.get((left.arc_id, right.arc_id))
    if utility is None:
        return 0.0
    return weights.get("transition", 0.0) * utility.weighted_value


def _retained_path(
    lattice: DeliberationLattice,
    *,
    policy: DeliberationPolicy,
) -> PathHypothesis:
    arcs = tuple(span.retained_arc for span in lattice.spans)
    weights = policy.weights
    transitions = _transition_map(lattice)
    base = 0.0
    previous: LatticeArc | None = None
    for arc in arcs:
        base += _weighted_arc_score(arc, weights)
        base += policy.retention_bonus * _arc_factor_weight(arc)
        if previous is not None:
            base += _transition_score(previous, arc, transitions, weights)
        previous = arc
    return PathHypothesis(
        arcs=arcs,
        base_score=base,
        mean_audio_support=_mean_audio_support(arcs, weights),
        final_score=base,
    )


def _enumerate_base_paths(
    lattice: DeliberationLattice,
    *,
    policy: DeliberationPolicy,
) -> tuple[PathHypothesis, ...]:
    weights = policy.weights
    transitions = _transition_map(lattice)
    beam: list[tuple[tuple[LatticeArc, ...], float]] = [((), 0.0)]
    for span in lattice.spans:
        expanded: list[tuple[tuple[LatticeArc, ...], float]] = []
        for prefix, prefix_score in beam:
            for arc in _eligible_arcs(span, policy=policy):
                score = prefix_score + _weighted_arc_score(arc, weights)
                if arc.arc_id == span.retained_arc_id:
                    score += policy.retention_bonus * _arc_factor_weight(arc)
                if prefix:
                    score += _transition_score(prefix[-1], arc, transitions, weights)
                expanded.append((prefix + (arc,), score))
        expanded.sort(
            key=lambda row: (
                -row[1],
                tuple(arc.arc_id for arc in row[0]),
            )
        )
        beam = expanded[: policy.beam_size]
    return tuple(
        PathHypothesis(
            arcs=arcs,
            base_score=score,
            mean_audio_support=_mean_audio_support(arcs, weights),
            final_score=score,
        )
        for arcs, score in beam
        if any(arc.text for arc in arcs)
    )


def _resolution_mode(selected: LatticeArc, retained: LatticeArc) -> ResolutionMode:
    if selected.arc_id == retained.arc_id:
        return "retained-first-pass"
    if (
        selected.pronunciation_key is not None
        and selected.pronunciation_key == retained.pronunciation_key
    ):
        return "context-resolved-orthography"
    if selected.origin in GENERATED_ORIGINS:
        return "acoustically-verified-proposal"
    return "acoustic-context-consensus"


def _global_scores(
    paths: Sequence[PathHypothesis],
    scorer: GlobalSequenceScorer,
    *,
    context: DocumentContext,
) -> tuple[dict[str, GlobalPathScore], str, str]:
    score_many = getattr(scorer, "score_many", None)
    if callable(score_many):
        rows = tuple(score_many(tuple(path.arcs for path in paths), context=context))
    else:
        rows = tuple(scorer.score(path.arcs, context=context) for path in paths)
    if len(rows) != len(paths):
        raise ValueError("global sequence scorer returned the wrong number of path scores")
    by_digest: dict[str, GlobalPathScore] = {}
    for row in rows:
        if row.context_digest != context.digest:
            raise ValueError("global sequence score is bound to different context")
        if row.path_digest in by_digest:
            raise ValueError("global sequence scorer returned a duplicate path score")
        by_digest[row.path_digest] = row
    expected = {path.digest for path in paths}
    if set(by_digest) != expected:
        raise ValueError("global sequence scorer returned unknown or missing path scores")
    sources = {row.source for row in rows}
    profiles = {row.profile_digest for row in rows}
    if len(sources) != 1 or len(profiles) != 1:
        raise ValueError("one deliberation decision cannot mix global scorer identities")
    return by_digest, next(iter(sources)), next(iter(profiles))


def decode_global_lattice(
    lattice: DeliberationLattice,
    *,
    policy: DeliberationPolicy,
    context: DocumentContext | None = None,
    sequence_scorer: GlobalSequenceScorer | None = None,
) -> GlobalDeliberationDecision:
    """Decode a complete transcript path with hard acoustic-retention constraints."""

    context = context or DocumentContext()
    retained = _retained_path(lattice, policy=policy)
    base_paths = list(_enumerate_base_paths(lattice, policy=policy))
    if not base_paths:
        raise ValueError("no observed-eligible path survived deliberation guards")

    guarded_paths = [
        path
        for path in base_paths
        if retained.mean_audio_support - path.mean_audio_support
        <= policy.maximum_mean_audio_regression
    ]
    if not guarded_paths:
        guarded_paths = [retained]
    elif retained.digest not in {path.digest for path in guarded_paths}:
        guarded_paths.append(retained)

    score_rows: dict[str, GlobalPathScore] = {}
    scorer_source: str | None = None
    scorer_profile: str | None = None
    if sequence_scorer is not None:
        score_rows, scorer_source, scorer_profile = _global_scores(
            guarded_paths,
            sequence_scorer,
            context=context,
        )

    rescored: list[PathHypothesis] = []
    for path in guarded_paths:
        row = score_rows.get(path.digest)
        context_value = 0.0 if row is None else row.value
        final = path.base_score + policy.global_context_weight * context_value
        rescored.append(
            PathHypothesis(
                arcs=path.arcs,
                base_score=path.base_score,
                mean_audio_support=path.mean_audio_support,
                context_score=context_value,
                final_score=final,
                context_source=None if row is None else row.source,
                context_profile_digest=None if row is None else row.profile_digest,
            )
        )
    rescored.sort(
        key=lambda path: (
            -path.final_score,
            -path.base_score,
            tuple(arc.arc_id for arc in path.arcs),
        )
    )
    selected = rescored[0]
    has_runner_up = len(rescored) > 1
    margin = selected.final_score - rescored[1].final_score if has_runner_up else 0.0

    resolutions: list[SpanResolution] = []
    reasons: list[str] = []
    weights = policy.weights
    selected_generated = False
    for span, arc in zip(lattice.spans, selected.arcs, strict=True):
        retained_arc = span.retained_arc
        mode = _resolution_mode(arc, retained_arc)
        if mode != "retained-first-pass":
            reasons.append(f"changed-span:{span.span_id}:{mode}")
        if arc.origin in GENERATED_ORIGINS:
            selected_generated = True
        resolutions.append(
            SpanResolution(
                span_id=span.span_id,
                retained_arc_id=retained_arc.arc_id,
                selected_arc_id=arc.arc_id,
                mode=mode,
                retained_audio_support=_audio_support(retained_arc, weights),
                selected_audio_support=_audio_support(arc, weights),
            )
        )

    status: DecisionStatus = "accepted"
    if has_runner_up and margin < policy.minimum_final_margin:
        status = "provisional"
        reasons.append("low-global-margin")
    if not has_runner_up:
        reasons.append("single-surviving-path")
    if selected_generated and policy.provisional_on_generated:
        status = "provisional"
        reasons.append("selected-generated-proposal")
    if sequence_scorer is not None and selected.context_score != 0:
        reasons.append("global-context-applied")

    retained_scored = next(
        (path for path in rescored if path.digest == retained.digest),
        retained,
    )
    return GlobalDeliberationDecision(
        selected=selected,
        retained=retained_scored,
        alternatives=tuple(rescored),
        status=status,
        margin=margin,
        resolutions=tuple(resolutions),
        reasons=tuple(dict.fromkeys(reasons)),
        lattice_digest=lattice.digest,
        policy_digest=policy.digest,
        context_digest=context.digest,
        scorer_source=scorer_source,
        scorer_profile_digest=scorer_profile,
    )
