#!/bin/bash
# UNIONS v2.0.0 demo from nothing. Needs apptainer, membership of the CADC group
# cfis-read, and a proxy certificate (`cadc-get-cert -u <user>` writes
# ~/.ssl/cadcproxy.pem). The pseudo-Cℓ of all ten bin pairs wants a node with
# ~64 GB of memory; it reads the release from the vault on first run.
set -e
apptainer pull sp_validation.sif docker://ghcr.io/cosmostat/sp_validation:develop
curl -fsSLO -E ~/.ssl/cadcproxy.pem \
    https://ws-cadc.canfar.net/vault/files/cfis/weak_lensing/ShapePipe/v2.x/v2.0.0/demo/demo.py
apptainer exec sp_validation.sif python demo.py
