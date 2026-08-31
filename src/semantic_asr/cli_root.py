from __future__ import annotations

import sys

from .cli_v2 import main as compatibility_main
from .frontier_cli import FRONTIER_COMMANDS
from .frontier_cli import main as frontier_main


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in FRONTIER_COMMANDS:
        return frontier_main(values)
    return compatibility_main(values)
