from __future__ import annotations

import sys

from .cli_v2 import main as compatibility_main
from .experiment_cli import EXPERIMENT_COMMANDS
from .experiment_cli import main as experiment_main
from .frontier_cli import FRONTIER_COMMANDS
from .frontier_cli import main as frontier_main
from .research_cli import RESEARCH_COMMANDS
from .research_cli import main as research_main
from .run_cli import RUN_COMMANDS
from .run_cli import main as run_main


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] in RUN_COMMANDS:
        return run_main(values)
    if values and values[0] in FRONTIER_COMMANDS:
        return frontier_main(values)
    if values and values[0] in EXPERIMENT_COMMANDS:
        return experiment_main(values)
    if values and values[0] in RESEARCH_COMMANDS:
        return research_main(values)
    return compatibility_main(values)
