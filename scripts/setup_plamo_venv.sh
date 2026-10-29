#!/usr/bin/env bash
# Create .venv-plamo with transformers 4.57.x for PLaMo 3 (see model card).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${CREATE_EVAL_PLAMO_VENV:-$REPO_ROOT/.venv-plamo}"
REQ_FILE="$REPO_ROOT/scripts/requirements-plamo-venv.txt"

# PLaMo venv needs Python 3.10+ and transformers 4.57 (not available on system python3.8).
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON="python3.12"
  else
    PYTHON="python3"
  fi
fi

echo "Creating PLaMo venv at: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install -U pip wheel
pip install -r "$REQ_FILE"

# Import path only; do not pull main-project deps (transformers>=4.58).
pip install -e "$REPO_ROOT" --no-deps

echo ""
echo "Verifying benchmark.run import..."
python -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from CybersecurityBenchmarks.benchmark import run  # noqa: F401
import transformers
print('OK: transformers', transformers.__version__)
"

echo ""
echo "Done. PLaMo subprocess will use: $VENV_DIR/bin/python"
echo "Optional: export CREATE_EVAL_PLAMO_PYTHON=$VENV_DIR/bin/python"
