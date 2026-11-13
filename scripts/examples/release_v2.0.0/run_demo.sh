#!/bin/bash
# UNIONS v2.0.0 demo from nothing. Needs apptainer, membership of the CADC group
# cfis-read, and a proxy certificate (`cadc-get-cert -u <user>` writes ~/.ssl/cadcproxy.pem).
set -e
V=https://ws-cadc.canfar.net/vault/files/cfis/weak_lensing/ShapePipe/v2.x/v2.0.0
apptainer pull sp_validation.sif docker://ghcr.io/cosmostat/sp_validation:develop
curl -fsSLO -E ~/.ssl/cadcproxy.pem $V/nz/nz_tomo.txt
curl -fsSLO -E ~/.ssl/cadcproxy.pem $V/blind/unions.blind.json  # read by the code; never open it
apptainer exec sp_validation.sif python demo.py
