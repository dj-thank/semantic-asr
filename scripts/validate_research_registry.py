#!/usr/bin/env python3
from __future__ import annotations

import json

from semantic_asr.research_registry import SourceStatus, default_research_registry


def main() -> int:
    registry = default_research_registry()
    source_by_id = {source.source_id: source for source in registry.sources}
    for translation in registry.translations:
        for source_id in translation.source_ids:
            source = source_by_id[source_id]
            if source.status != SourceStatus.PINNED_PRIMARY:
                raise RuntimeError(
                    f"translation {translation.translation_id} uses unpinned source {source_id}"
                )
    print(
        json.dumps(
            {
                "registryDigest": registry.digest,
                "version": registry.version,
                "pinnedSources": [
                    source.source_id
                    for source in registry.sources
                    if source.status == SourceStatus.PINNED_PRIMARY
                ],
                "provisionalSources": [
                    source.source_id
                    for source in registry.sources
                    if source.status == SourceStatus.PROVISIONAL
                ],
                "translations": [
                    translation.translation_id
                    for translation in registry.translations
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
