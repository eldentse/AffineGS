#!/bin/bash
# TRAIN SCRIPT — paper settings
# Reproduces the MuSHRoom results of the paper (Table 14, Depth-Anything-V2):
# per-scene guided-filter radius/eps from
# depthanythingv2_disparity_normal_hyperparams.txt and per-scene loss weights
# (see below; both are listed in the paper's Table 11), with meshes from Open3D
# TSDF fusion, the mesher the paper's numbers come from.
#
# Usage: bash run_train_paper.sh [scene ...]   (default: all six scenes)
#        EXP_NAME=<name> bash run_train_paper.sh ...   to write elsewhere
# After a crash, rerun the same command: finished scenes are skipped, and a
# scene left half-done is moved aside to <scene>_incomplete_<time> and retrained.

# Pin to GPU 0 (PCI 0000:1a:00) — GPU 1 (0000:68:00) drives the display and has
# been dropping off the PCIe bus (Xid 79) under sustained training load
# (2026-10-15 18:08 and 22:27), blacking out the screen and forcing a reboot.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Log GPU temperature/power every 5s so a crash leaves evidence of what the
# card was doing right before it died.
mkdir -p ./outputs
GPU_LOG=./outputs/gpu_log_$(date +%Y%m%d_%H%M%S).csv
nvidia-smi --query-gpu=timestamp,index,pci.bus_id,temperature.gpu,fan.speed,power.draw,clocks.sm,clocks_event_reasons.active \
  --format=csv -l 5 > "$GPU_LOG" &
GPU_LOG_PID=$!
trap 'kill $GPU_LOG_PID 2>/dev/null' EXIT
echo "Logging GPU stats to $GPU_LOG"

# Stop the whole run on the first failure instead of moving on to the next scene.
fail() {
  echo "ERROR: $1 failed on scene '$item', stopping." >&2
  echo "If the GPU crashed, check: journalctl -k | grep Xid   and the tail of $GPU_LOG" >&2
  exit 1
}


DATA_DIR=./datasets/room_datasets
EXP_NAME=${EXP_NAME:-affinegs_paper/iphone_pcd/only_rgb}

# Per-scene guided-filter settings (depthanythingv2_disparity_normal_hyperparams.txt).
declare -A RADIUS=( [coffee_room]=600  [computer]=600 [honka]=600 [kokko]=550 [sauna]=550 [vr_room]=650 )
declare -A EPS=(    [coffee_room]=0.08 [computer]=0.5 [honka]=0.1 [kokko]=0.4 [sauna]=0.3 [vr_room]=0.2 )

# Per-scene loss weights: lambda_grad / lambda_normal / lambda_depth (L1).
# The hyperparameter file lists 0.1 / 0.08 / 0.4 for every scene, but with
# those kokko and sauna fill with floaters (kokko PSNR 19.9, sauna Abs Rel 0.19);
# half the weights reproduce kokko and a quarter reproduce sauna (2026-10-19/20 sweeps).
declare -A GRAD=(   [coffee_room]=0.1  [computer]=0.1  [honka]=0.1  [kokko]=0.05 [sauna]=0.025 [vr_room]=0.1  )
declare -A NORMAL=( [coffee_room]=0.08 [computer]=0.08 [honka]=0.08 [kokko]=0.04 [sauna]=0.02  [vr_room]=0.08 )
declare -A DEPTH=(  [coffee_room]=0.4  [computer]=0.4  [honka]=0.4  [kokko]=0.2  [sauna]=0.1   [vr_room]=0.4  )

if [ $# -gt 0 ]; then SCENES=("$@"); else SCENES=("coffee_room" "honka" "kokko" "sauna" "computer" "vr_room"); fi
for item in "${SCENES[@]}"; do
  [ -n "${RADIUS[$item]}" ] || { echo "ERROR: no hyperparameters for scene '$item'" >&2; exit 1; }
done

for item in "${SCENES[@]}"
do
  run_dir=outputs/$EXP_NAME/dn-splatter/$item
  if [ -f $run_dir/mesh_o3/mesh_metrics.json ]; then
    echo "=== $item already done, skipping ==="
    continue
  fi
  # A folder without final metrics is left over from a crashed run: keep it aside.
  [ -d $run_dir ] && mv $run_dir ${run_dir}_incomplete_$(date +%Y%m%d_%H%M%S)

  echo "=== $item: guided-blur-radius ${RADIUS[$item]}, guided-blur-eps ${EPS[$item]}, loss weights ${GRAD[$item]} / ${NORMAL[$item]} / ${DEPTH[$item]} ==="

  ns-train dn-splatter --experiment-name $EXP_NAME \
    --vis tensorboard \
    --timestamp $item \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.use-disparity-normal-loss True \
    --pipeline.model.disparity-depth-loss-lambda ${GRAD[$item]} \
    --pipeline.model.disparity-render-normal-loss-lambda ${NORMAL[$item]} \
    --pipeline.model.l1-weight ${DEPTH[$item]} \
    --pipeline.model.guided-blur-radius ${RADIUS[$item]} \
    --pipeline.model.guided-blur-eps ${EPS[$item]} \
    --pipeline.model.use_opacity_shrink False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.normal-lambda 0.1 \
    --pipeline.model.normal-loss-type L1 \
    --pipeline.model.opacity_shrink_lambda 1e-4 \
    mushroom \
    --data $DATA_DIR/$item \
    --depth-mode disparity \
    --load-normals False \
    || fail ns-train


  gs-mesh o3dtsdf \
    --load-config outputs/$EXP_NAME/dn-splatter/$item/config.yml \
    --output-dir outputs/$EXP_NAME/dn-splatter/$item/mesh_o3 \
    || fail gs-mesh

  ns-eval \
        --load-config outputs/$EXP_NAME/dn-splatter/$item/config.yml \
        --output-path outputs/$EXP_NAME/dn-splatter/$item/eval/results.json \
        --render-output-path outputs/$EXP_NAME/dn-splatter/$item/ \
        || fail ns-eval

  # Writes mesh_metrics.json next to the mesh, in mesh_o3/.
  python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
      --gt_mesh_path $DATA_DIR/$item \
      --pred_mesh_path outputs/$EXP_NAME/dn-splatter/$item/mesh_o3/Open3dTSDFfusion_mesh.ply \
      --device iphone \
      || fail eval_mesh

  # Let the GPU cool down before the next scene.
  sleep 120
done

# results.json stores the depth metrics under "faro" (the GT depth source), not "depth".
DESIRED_METRICS=(
    "within_rgb_psnr"
    "with_rgb_psnr"
    "within_rgb_ssim"
    "with_rgb_ssim"
    "within_rgb_lpips"
    "with_rgb_lpips"
    "within_faro_abs_rel"
    "with_faro_abs_rel"
    "within_faro_sq_rel"
    "with_faro_sq_rel"
    "within_faro_rmse"
    "with_faro_rmse"
    "within_faro_a1"
    "with_faro_a1"
)
EXP_DIR=./outputs/$EXP_NAME/dn-splatter
# Loop through all subdirectories (scenes) in the experiment folder
for scene_dir in "$EXP_DIR"/*/; do
    # Skip if it's not actually a directory
    [ -d "$scene_dir" ] || continue

    scene_name=$(basename "$scene_dir")
    # Leftovers from crashed runs are not results.
    case "$scene_name" in *_incomplete_*) continue ;; esac
    echo "Scene: $scene_name"
    echo "---------------------------------------------------------"

    # --- 1. Extract from eval/results.json ---
    RESULTS_FILE="$scene_dir/eval/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  [eval/results.json]"

        # Loop through our list of desired metrics
        for key in "${DESIRED_METRICS[@]}"; do
            # Extract the value, default to "N/A" if it doesn't exist
            val=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['results'].get(sys.argv[2], 'N/A'))" "$RESULTS_FILE" "$key")

            # Print with aligned columns for better readability (22 characters wide)
            printf "    %-22s : %s\n" "$key" "$val"
        done
    else
        echo "  [eval/results.json] -> Not found"
    fi

    echo "" # Add a blank line for readability

    # --- 2. Extract from mesh_o3/mesh_metrics.json ---
    MESH_FILE="$scene_dir/mesh_o3/mesh_metrics.json"
    if [ -f "$MESH_FILE" ]; then
        echo "  [mesh_o3/mesh_metrics.json]"

        # Dynamically loop through ALL fields and print them with aligned columns
        python -c "import json,sys; [print(f'    {k:<22} : {v}') for k, v in json.load(open(sys.argv[1])).items()]" "$MESH_FILE"
    else
        echo "  [mesh_o3/mesh_metrics.json] -> Not found"
    fi

    echo "========================================================="
done
