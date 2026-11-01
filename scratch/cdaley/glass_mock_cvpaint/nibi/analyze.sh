#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
spv python -u "$REPO/scratch/cdaley/glass_mock_cvpaint/validate.py" "$@"
