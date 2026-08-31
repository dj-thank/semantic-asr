from __future__ import annotations

import sys

from .cli_v2 import main as compatibility_main
from .experiment_cli import EXPERIMENT_COMMANDS
from .experiment_cli import main as experiment_main
from .frontier_cli import FRONTIER_COMMANDS
from .frontier_cli import main as frontier_main


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in FRONTIER_COMMANDS:
        return frontier_main(values)
    if values and values[0] in EXPERIMENT_COMMANDS:
        return experiment_main(values)
    return compatibility_main(values)
