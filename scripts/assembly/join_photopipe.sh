#!/bin/bash
# Rebuild the PhotoPipe join from a login node on the cluster that holds the
# shapes, under nohup or tmux:
#   extract  step 1, extract_photopipe.py in a CANFAR headless session (unchanged tiles are skipped)
#   stage    copy the extract to $EXTRACT from this login node
#   join     step 2, join_photopipe.py on a Slurm job, inside the sp_validation image
# Set ACCOUNT (Slurm account) and CANFAR_USER; STAGES picks a subset.
# SHAPES and OUTPUT default to the data roots in join_photopipe.py.
set -euo pipefail

: "${ACCOUNT:?Slurm account for the join job}"
: "${CANFAR_USER:?CANFAR user whose arc home holds the extract}"
EXTRACT=${EXTRACT:-$SCRATCH/photopipe-dr6/extract}
SIF=${SIF:-$HOME/.cache/sp_validation/sp_validation.sif}
STREAMS=${STREAMS:-8}
STAGES=${STAGES:-extract stage join}
ARC=home/$CANFAR_USER/photopipe-dr6

API=https://ws-uv.canfar.net
CERT=$HOME/.ssl/cadcproxy.pem
CODE=$(cd "$(dirname "$0")/../.." && pwd)

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

curl_api() { curl -sS --retry 5 --retry-all-errors -m 120 --cert "$CERT" "$@"; }

if has_stage extract; then
    vcp "$CODE/scripts/assembly/extract_photopipe.py" "arc:$ARC/extract_photopipe.py"
    vrm "arc:$ARC/extract/COMPLETE" 2>/dev/null || true
    session=$(curl_api -X POST "$API/skaha/v1/session" \
        -d name=photopipe-extract -d image=images.canfar.net/skaha/astroflow:latest \
        -d cores=16 -d ram=64 -d kind=headless --data-urlencode cmd=python \
        --data-urlencode "args=/arc/$ARC/extract_photopipe.py /arc/$ARC/extract --workers 16")
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

if has_stage join; then
    sbatch --wait -A "$ACCOUNT" -c 32 --mem=180G -t 3:00:00 -J photopipe-join \
        -o "$EXTRACT/join-%j.log" --wrap "module load apptainer; \
        apptainer exec -B /project,/scratch --env PYTHONPATH=$CODE/src $SIF \
        python $CODE/scripts/assembly/join_photopipe.py $EXTRACT --workers 32"
fi
