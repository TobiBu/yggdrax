import os
import pathlib
import sys

# jax 0.11.2: XLA:CPU's LLVM loop vectorizer can fall into a recursion
# (`llvm::vputils::onlyFirstLaneUsed`) that stalls a compile for many minutes; this
# suite stopped at a quarter for over an hour. Turning the pass off costs these
# compile-bound tests nothing; it must be set before the first backend starts.
# YGGDRAX_TEST_KEEP_LOOP_VECTORIZER=1 leaves the flags alone.
if os.environ.get("YGGDRAX_TEST_KEEP_LOOP_VECTORIZER", "0") != "1" and (
    "xla_backend_extra_options" not in os.environ.get("XLA_FLAGS", "")
):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_backend_extra_options=-vectorize-loops=false"
    ).strip()

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
