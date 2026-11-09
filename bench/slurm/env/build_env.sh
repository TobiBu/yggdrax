#!/usr/bin/env bash
# Build the pinned GPU environment for the scaling campaign, on a LOGIN node.
#
#   bash bench/slurm/env/build_env.sh horeka   [VENV_DIR]   # default ~/envs/yggdrax-jax0112
#   bash bench/slurm/env/build_env.sh juwels   [VENV_DIR]
#   bash bench/slurm/env/build_env.sh generic  [VENV_DIR]   # any Linux box with python3.12
#
# What it makes: a plain venv on top of the site's Python 3.12, with the pinned
# wheels of requirements-jax0112-cuda12.lock.txt (jax 0.11.2 + CUDA 12 runtime as nvidia-*
# wheels) and this checkout of yggdrax installed editable (--no-deps). No CUDA
# module is needed at run time: the CUDA libraries come from the wheels; only the
# driver is the system's. Login nodes have internet; compute nodes may not, so
# build here, not in a job.
#
# Put the venv on a file system the compute nodes see and that is not purged:
# HoreKa: $HOME or a workspace (ws_allocate); JUWELS: $PROJECT_<project> (not
# $HOME, which is small). The job scripts activate $VENV, default ~/envs/yggdrax-jax0112:
# the same environment bench/slurm/differentiability_scaling.sbatch expects.
#
# The wheels need glibc >= 2.28 (manylinux_2_28): RHEL/Rocky 8 or newer, which
# both HoreKa and JUWELS run.
set -euo pipefail

SITE="${1:-generic}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
VENV="${2:-${VENV:-${HOME}/envs/yggdrax-jax0112}}"

case "${SITE}" in
  horeka)
    # Check `module avail devel/python` for the current name of a 3.12 module.
    module purge
    module load devel/python/3.12.3_gnu_14.2 || module load devel/python/3.12
    ;;
  juwels)
    # Check `module spider Python` for the current stage; 3.12 must be the
    # interpreter, the stage's numpy/scipy are NOT used (the venv pins them).
    module purge
    module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
    ;;
  generic) ;;
  *) echo "unknown site ${SITE} (horeka | juwels | generic)" >&2; exit 2 ;;
esac

PY="${PYTHON:-python3.12}"
command -v "${PY}" >/dev/null || PY=python3
"${PY}" - <<'PYCHECK'
import sys
if sys.version_info[:2] < (3, 12):
    raise SystemExit(f"need Python >= 3.12 for jax 0.11.x, got {sys.version}")
PYCHECK

echo "# building ${VENV} with $(${PY} --version) for ${SITE}"
"${PY}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "${HERE}/requirements-jax0112-cuda12.lock.txt"
python -m pip install --no-deps -e "${REPO}"

# record exactly what was installed, next to the venv
python -m pip freeze > "${VENV}/pip-freeze.txt"
git -C "${REPO}" rev-parse HEAD > "${VENV}/yggdrax-commit.txt" || true

python - <<'PYCHECK'
import jax, jaxlib, yggdrax
print("jax", jax.__version__, "jaxlib", jaxlib.__version__, "yggdrax", yggdrax.__file__)
from yggdrax.distributed.comm import RAGGED_NATIVE_FIXED_JAX, _jax_version_tuple
assert _jax_version_tuple() >= RAGGED_NATIVE_FIXED_JAX, "jax too old for the native exchange"
print("native ragged all-to-all allowed (jax >= %s)" % (RAGGED_NATIVE_FIXED_JAX,))
PYCHECK
echo "# done. Job scripts: export VENV=${VENV} (the default when it is ~/envs/yggdrax-jax0112)"
