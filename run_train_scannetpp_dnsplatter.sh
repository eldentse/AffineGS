#!/bin/bash
# TRAIN SCRIPT — DN-Splatter with sensor depth on ScanNet++
# DN-Splatter's own ScanNet++ recipe (iPhone sensor depth, Omnidata normals),
# with the same frames, meshes (TSDF and Poisson) and evaluation as
# run_train_scannetpp.sh. Needs the Omnidata weights in omnidata_ckpt/
# (python dn_splatter/data/download_scripts/download_omnidata.py).
#
# Usage: bash run_train_scannetpp_dnsplatter.sh [scene ...]   (default: both scenes)
#        SCANNETPP_DATA=<path to scannetpp/data> bash run_train_scannetpp_dnsplatter.sh
export METHOD=dnsplatter
exec bash "$(dirname "$0")/run_train_scannetpp.sh" "$@"
