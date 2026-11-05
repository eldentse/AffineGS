#!/bin/bash
# TRAIN SCRIPT — DN-Splatter with sensor depth
# The original DN-Splatter: iPhone sensor depth (used as is, no alignment) plus
# monocular normal supervision. Everything else, including the Poisson
# (pred_mesh/) and TSDF (mesh_o3/) meshes, is as in run_train_dnsplatter_mono.sh.
#
# Usage: bash run_train_dnsplatter_sensor.sh [scene ...]   (default: all six scenes)
#        EXP_NAME=<name> bash run_train_dnsplatter_sensor.sh ...   to write elsewhere
export DEPTH_MODE=sensor
export EXP_NAME=${EXP_NAME:-dnsplatter_sensordepth/iphone_pcd/only_rgb}
exec bash "$(dirname "$0")/run_train_dnsplatter_mono.sh" "$@"
