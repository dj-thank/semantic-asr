"""Strict boundary hardening for the canonical score contract.

The primary contract remains in :mod:`semantic_asr.score_contract`.  This module is
loaded before the public package surface and replaces only parsing, immutability,
and cross-object validation hooks that need to fail closed on untrusted data.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import StrEnum
from types import MappingProxyType

from . import _score_contract_base as contract


class ScoreKind(StrEnum):
    """Legacy coarse score names retained as one canonical enum."""

    RAW = "raw"
    PROBABILITY = "probability"
    LOGIT = "logit"
    LOG_LIKELIHOOD = "log_likelihood"
    PREFERENCE = "preference"


_NORMALIZATION_ALIASES = {
    "sum": contract.ScoreNormalization.SEQUENCE,
    "sequence-sum": contract.ScoreNormalization.SEQUENCE,
    "per-token": contract.ScoreNormalization.MEAN_TOKEN,
    "mean-token": contract.ScoreNormalization.MEAN_TOKEN,
    "per-frame": contract.ScoreNormalization.MEAN_FRAME,
    "mean-frame": contract.ScoreNormalization.MEAN_FRAME,
    "length-normalized": contract.ScoreNormalization.TOKEN_POWER,
    "path-normalized": contract.ScoreNormalization.PATH_NORMALIZED,
}


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def _strict_str(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _strict_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
    return value


def _strict_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _normalization(value: object) -> contract.ScoreNormalization:
    if isinstance(value, contract.ScoreNormalization):
        return value
    if not isinstance(value, str):
        raise TypeError("normalization must be a string")
    if value in _NORMALIZATION_ALIASES:
        return _NORMALIZATION_ALIASES[value]
    return contract.ScoreNormalization(value)


def _aliased(
    row: Mapping[str, object],
    camel: str,
    snake: str | None = None,
    *,
    default: object = None,
    required: bool = False,
) -> object:
    names = (camel,) if snake is None else (camel, snake)
    present = [name for name in names if name in row]
    if len(present) == 2 and row[present[0]] != row[present[1]]:
        raise contract.ScoreMigrationError(
            f"conflicting aliases {present[0]!r} and {present[1]!r}"
        )
    if present:
        return row[present[0]]
    if required:
        raise contract.ScoreMigrationError(f"missing required field: {camel}")
    return default


def _reject_unknown(
    row: Mapping[str, object], *, allowed: set[str], name: str
) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise contract.ScoreMigrationError(f"{name} contains unknown fields: {unknown}")


class FrozenDict(Mapping[str, object]):
    """Recursively immutable JSON object that cannot be bypassed via ``dict`` APIs."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, object] | None = None) -> None:
        source = _strict_mapping(values or {}, name="metadata")
        active: set[int] = set()
        rows: dict[str, object] = {}
        for key, value in source.items():
            if not key:
                raise TypeError("metadata keys must be non-empty strings")
            rows[key] = _freeze(value, path=f"metadata.{key}", active=active)
        object.__setattr__(self, "_data", MappingProxyType(rows))

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"

    def __setitem__(self, _key: str, _value: object) -> None:
        raise TypeError("score metadata is immutable")

    def __delitem__(self, _key: str) -> None:
        raise TypeError("score metadata is immutable")

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, _memo: dict[int, object]) -> "FrozenDict":
        return self


def _freeze(value: object, *, path: str, active: set[int]) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return contract._strict_float(value, name=path)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a recursive mapping")
        active.add(identity)
        try:
            return FrozenDict(
                {
                    key: _freeze(item, path=f"{path}.{key}", active=active)
                    for key, item in _strict_mapping(value, name=path).items()
                }
            )
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{path} contains a recursive sequence")
        active.add(identity)
        try:
            return tuple(
                _freeze(item, path=f"{path}[]", active=active) for item in value
            )
        finally:
            active.remove(identity)
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _freeze_json(value: object, *, path: str = "metadata") -> object:
    return _freeze(value, path=path, active=set())


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
