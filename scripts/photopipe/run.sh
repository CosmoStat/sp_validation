#!/bin/bash
# Rebuild the shape + PhotoPipe merge. Run from a Fir login node, under nohup or tmux.
#   1. extract PhotoPipe columns in a CANFAR headless session (unchanged tiles are skipped)
#   2. stage the extract to Fir scratch from this login node
#   3. merge and check on a Slurm allocation
# Paths are overridable from the environment; a new shape version changes SHAPES and PRODUCT.
set -euo pipefail

BASE=/project/rrg-kyi/cdaley/unions
SHAPES=${SHAPES:-$BASE/data/v2.x/run1-snap3-hats/catalogue}
PRODUCT=${PRODUCT:-$BASE/data/photopipe-dr6}
EXTRACT=${EXTRACT:-/scratch/cdaley/photopipe-dr6/extract}
NZ=${NZ:-$BASE/data/nz-v2-full-products}
SIF=${SIF:-$BASE/analyses/mock-ensemble-tomo/inputs/containers/sp_validation-ddd58fe0.sif}
ACCOUNT=${ACCOUNT:-rrg-kyi_cpu}
CANFAR_USER=${CANFAR_USER:-cailmdaley}
ARC=home/$CANFAR_USER/photopipe-dr6
STREAMS=${STREAMS:-8}
STAGES=${STAGES:-extract stage merge}

API=https://ws-uv.canfar.net
CERT=$HOME/.ssl/cadcproxy.pem
CODE=$(cd "$(dirname "$0")/../.." && pwd)

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

curl_api() { curl -sS --retry 5 --retry-all-errors -m 120 --cert "$CERT" "$@"; }

if has_stage extract; then
    vcp "$CODE/scripts/photopipe/extract_on_canfar.py" "arc:$ARC/extract_on_canfar.py"
    vrm "arc:$ARC/extract/COMPLETE" 2>/dev/null || true
    session=$(curl_api -X POST "$API/skaha/v1/session" \
        -d name=photopipe-extract -d image=images.canfar.net/skaha/astroflow:latest \
        -d cores=16 -d ram=64 -d kind=headless --data-urlencode cmd=python \
        --data-urlencode "args=/arc/$ARC/extract_on_canfar.py /arc/$ARC/extract --workers 16")
    echo "CANFAR session $session"
    while :; do
        state=$(curl_api "$API/skaha/v1/session/$session" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
        [[ $state == Completed ]] && break
        [[ $state == Failed || $state == Error ]] && { echo "session $session $state"; exit 1; }
        sleep 300
    done
    vls "arc:$ARC/extract/COMPLETE"
fi

if has_stage stage; then
    mkdir -p "$EXTRACT"
    cd "$EXTRACT"
    vls "arc:$ARC/extract" | grep -E '^ra=[0-9]+\.(parquet|json)$' > remote.txt
    fetch='curl -fsS --retry 8 --retry-all-errors --cert '"$CERT"' -o "$1.part" "'"$API/arc/files/$ARC/extract"'/$1" && mv "$1.part" "$1"'
    xargs -P "$STREAMS" -I{} bash -c "$fetch" _ {} < remote.txt
    [[ $(ls ra=*.parquet ra=*.json | wc -l) == $(wc -l < remote.txt) ]]
fi

if has_stage merge; then
    sbatch --wait -A "$ACCOUNT" -c 32 --mem=180G -t 3:00:00 -J photopipe-merge \
        -o "$PRODUCT/merge-%j.log" --wrap "module load apptainer; \
        apptainer exec -B /project,/scratch --env PYTHONPATH=$CODE/src $SIF python $CODE/scripts/photopipe/merge_shapes.py $SHAPES $EXTRACT $PRODUCT --workers 32 && \
        apptainer exec -B /project,/scratch --env PYTHONPATH=$CODE/src $SIF python $CODE/scripts/photopipe/check_tom_bins.py $PRODUCT --model $NZ/som_assignment_model.npz --reference $NZ/tom_bin_full.parquet --workers 32"
fi
