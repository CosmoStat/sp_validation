#!/bin/bash
# Stage PhotoPipe's DR6 inputs of the v2 collection (config/assembly/v2.yaml)
# under DATA, the directory the recipe's ${unions_data} names. Run from a login
# node, under nohup or tmux:
#   extract  extract_photopipe.py in a CANFAR headless session, beside the tile
#            tree (unchanged tiles are skipped)
#   stage    copy the extract to DATA/photopipe-dr6-extract
#   specz    copy PhotoPipe's spec-z match over every tile and each tile's
#            redshift compilation to DATA/photopipe-dr6-specz-all
#
#   CANFAR_USER=<canfar user> scripts/assembly/fetch_photopipe.sh DATA
#
# CANFAR_USER's arc home holds the extract; STAGES picks a subset of stages,
# COMPILATION the compilation's date. Needs a CADC proxy at ~/.ssl/cadcproxy.pem.
set -euo pipefail

DATA=${1:?usage: fetch_photopipe.sh DATA}
STREAMS=${STREAMS:-8}
STAGES=${STAGES:-extract stage specz}
COMPILATION=${COMPILATION:-20261019}
DR6=projects/unions/catalogues/unions/GAaP_photometry/UNIONS_DR6
CODE=$(cd "$(dirname "$0")/../.." && pwd)

export API=https://ws-uv.canfar.net
export CERT=$HOME/.ssl/cadcproxy.pem

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

curl_api() { curl -sS --retry 5 --retry-all-errors -m 120 --cert "$CERT" "$@"; }

# Copy the arc file $1 (a path under arc/files) here, under its own name.
get() {
    local name=${1##*/}
    curl -fsS --retry 8 --cert "$CERT" -o "$name.part" "$API/arc/files/$1" &&
        mv "$name.part" "$name"
}
export -f get

if has_stage extract || has_stage stage; then
    : "${CANFAR_USER:?CANFAR user whose arc home holds the extract}"
    ARC=home/$CANFAR_USER/photopipe-dr6
fi

if has_stage extract; then
    vmkdir -p "arc:$ARC"
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
    mkdir -p "$DATA/photopipe-dr6-extract"
    cd "$DATA/photopipe-dr6-extract"
    vls "arc:$ARC/extract" | grep -E '^(ra=[0-9]+\.(parquet|json)|COMPLETE)$' |
        sed "s|^|$ARC/extract/|" > remote.txt
    xargs -P "$STREAMS" -I{} bash -c 'get "$1"' _ {} < remote.txt
    [[ $(ls ra=*.parquet ra=*.json COMPLETE | wc -l) == $(wc -l < remote.txt) ]]
fi

if has_stage specz; then
    mkdir -p "$DATA/photopipe-dr6-specz-all/asc"
    cd "$DATA/photopipe-dr6-specz-all"
    for name in UNIONS_DR6_ugriz_specz.cat UNIONS_DR6_ugriz_specz_tiles.txt; do
        get "${DR6}_QC/$name"
    done
    # The compilation of each tile in the match; missing.txt lists the tiles
    # without one on arc.
    cd asc
    : > missing.txt
    sed 's|.*/UNIONS\.\([0-9]*\.[0-9]*\)_ugriz.*|\1|' ../UNIONS_DR6_ugriz_specz_tiles.txt |
        sed "s|.*|$DR6/UNIONS.&/specz/UNIONS.&_redshifts-$COMPILATION.asc|" |
        xargs -P "$STREAMS" -I{} bash -c 'get "$1" || echo "$1" >> missing.txt' _ {}
    echo "$(ls | grep -c '\.asc$') compilations, $(wc -l < missing.txt) tiles without one"
fi
