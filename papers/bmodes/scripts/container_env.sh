# Shared container settings for the run_*.sh sweep drivers. Source, don't run:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/container_env.sh"
#
# Sets CONTAINER (this user's image) and BIND (the mounts to pass to
# `apptainer exec`), resolved exactly as `sp_validation/container.py` does:
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
