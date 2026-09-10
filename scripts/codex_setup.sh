#!/usr/bin/env bash
# Codex cloud setup phase only: package installation, no audio/model downloads.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
python -m pip install -e '.[dev]'
python -m pip check
python scripts/codex_verify.py --plan
# Environment locking remains #28; this installation is not a transitive lock.
