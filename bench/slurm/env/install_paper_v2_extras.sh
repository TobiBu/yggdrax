#!/usr/bin/env bash
# Add the paper v2 run list's packages to an environment built by build_env.sh.
#
#   bash bench/slurm/env/build_env.sh generic ~/envs/yggdrax-jax0112      # once
#   bash bench/slurm/env/install_paper_v2_extras.sh [VENV_DIR]           # this
#
# 1. requirements-paper-v2-extras.lock.txt (autocvd, matplotlib): required; fails hard.
# 2. requirements-paper-v2-baselines.lock.txt, one group at a time (jaxkd; JZ-Tree;
#    KeOps; TreeCorr): optional; a group that does not install is reported and
#    skipped, and its bench arm then records "not installed".
# 3. Checks: jax is still the pinned 0.11.2 (nothing upgraded it), and which
#    baselines import.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-${VENV:-${HOME}/envs/yggdrax-jax0112}}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
LOCK="${HERE}/requirements-paper-v2-baselines.lock.txt"

pin() {  # the pinned spec of each named package in the baselines lock
  local name
  for name in "$@"; do
    grep -iE "^${name}==" "${LOCK}" | sed 's/[[:space:]]*#.*//'
  done
}

echo "# extras (required)"
python -m pip install -r "${HERE}/requirements-paper-v2-extras.lock.txt"

FAILED=()
group() {  # group LABEL PACKAGE...: install the pinned packages, --no-deps (all pinned)
  local label="$1"; shift
  echo "# baseline group: ${label}"
  # shellcheck disable=SC2046
  if ! python -m pip install --no-deps $(pin "$@"); then
    echo "# WARNING: ${label} did not install; its arm will skip" >&2
    FAILED+=("${label}")
  fi
}
group jaxkd jaxkd
group jztree jztree jztree-cu12
group keops pybind11 keopscore pykeops
group treecorr LSSTDESC.Coord future PyYAML setuptools TreeCorr

python -m pip freeze > "${VENV}/pip-freeze.txt"
python - <<'PYCHECK'
import importlib
import jax
assert jax.__version__ == "0.11.2", f"jax was changed to {jax.__version__}; rebuild"
print("jax", jax.__version__)
for mod in ("autocvd", "matplotlib", "jaxkd", "jztree", "pykeops", "treecorr", "Corrfunc"):
    try:
        importlib.import_module(mod)
        print(f"  {mod:10s} ok")
    except Exception as exc:  # jztree raises here without a CUDA jax/driver
        print(f"  {mod:10s} unavailable: {type(exc).__name__}: {str(exc)[:120]}")
PYCHECK
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "# groups that did not install: ${FAILED[*]}"
fi
echo "# done; pip freeze in ${VENV}/pip-freeze.txt"
