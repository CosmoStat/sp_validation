#!/bin/bash
# Paint the three seed-1 twins (clean, PSF leak 0.1, additive B 2.6e-11) onto the v2 input,
# with the same rotation seed and injections as the v1.4.6.3 run.
set -euo pipefail
source "$(dirname "$0")/env.sh"
P="python $REPO/scripts/glass_mock/paint_catalogue.py paint --seed 1 --shear gamma --maps $CVPAINT_MOCK/maps --input $INPUT --w-col w"
cd "$REPO"
spv $P --output "$CVPAINT_MOCK/painted.h5" > "$CVPAINT_MOCK/paint.log" 2>&1 &
spv $P --alpha 0.1 --output "$CVPAINT_MOCK/painted_psfleak.h5" > "$CVPAINT_MOCK/psfleak.log" 2>&1 &
spv $P --b-amplitude 2.6e-11 --b-seed 401 --output "$CVPAINT_MOCK/painted_addB.h5" > "$CVPAINT_MOCK/addB.log" 2>&1 &
wait
echo "painted $(date -Is)"
