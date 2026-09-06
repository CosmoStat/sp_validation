# Shared environment for the run_*.sh sweep drivers. Source, don't run:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/container_env.sh"
#
# Sets the checkout the drivers run out of, the container to run in, and
# `spv_python`, which is how every driver invokes python inside it.
WT=/n17data/cdaley/unions/code/sp_validation.worktrees/repro-paper-ii-astra
SRC=$WT/src
WSCRIPTS=$WT/workflow/scripts
PSCRIPTS=$WT/papers/bmodes/scripts

# CONTAINER and BIND are resolved exactly as `sp_validation/container.py` does:
# the writable sandbox if there is one, else the SIF.
_spv_cache=${XDG_CACHE_HOME:-$HOME/.cache}/sp_validation
_spv_sandbox=${SPV_SANDBOX:-$_spv_cache/sandbox}
if [ -d "$_spv_sandbox" ]; then
  CONTAINER=$_spv_sandbox
else
  CONTAINER=${SPV_CONTAINER:-$_spv_cache/sp_validation.sif}
fi
BIND=${SPV_APPTAINER_BINDS:-/home,/scratch,/automnt,/n17data,/n23data1,/n09data}
unset _spv_cache _spv_sandbox

# Every math library pinned to one thread -- pass as SPV_EXEC_EXTRA where the
# parallelism is by process, not by thread.
SINGLE_THREAD_ENV="--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1
  --env MKL_NUM_THREADS=1 --env NUMBA_NUM_THREADS=1 --env NUMEXPR_NUM_THREADS=1
  --env VECLIB_MAXIMUM_THREADS=1"

# Run python inside the container against the checkout's src. Extra
# `apptainer exec` flags go in SPV_EXEC_EXTRA (word-split on purpose).
spv_python() {
  apptainer exec --bind "$BIND" --env PYTHONPATH="$SRC" ${SPV_EXEC_EXTRA:-} \
    "$CONTAINER" /usr/local/bin/python "$@"
}

# Echo the version list a sweep runs over: $VERSIONS if the caller set one,
# else whatever sweep_versions.py resolves from $1 (a config path).
sweep_versions() {
  if [ -n "${VERSIONS:-}" ]; then
    echo "$VERSIONS"
  else
    spv_python "$PSCRIPTS/sweep_versions.py" --config "$1"
  fi
}
