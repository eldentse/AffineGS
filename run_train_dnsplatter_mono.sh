#!/bin/bash
# TRAIN SCRIPT — DN-Splatter (Monodepth) baseline
# DN-Splatter with the sensor depth replaced by monocular depth (mono_depth/),
# re-aligned to the rendered depth with a per-image scale and shift at every
# step, plus monocular normal supervision (normals_from_pretrain/). Meshes are
# extracted with both Poisson (pred_mesh/) and Open3D TSDF fusion (mesh_o3/).
#
# Usage: bash run_train_dnsplatter_mono.sh [scene ...]   (default: all six scenes)
#        EXP_NAME=<name> bash run_train_dnsplatter_mono.sh ...   to write elsewhere
# DEPTH_MODE=sensor uses the iPhone sensor depth instead of mono_depth/; see
# run_train_dnsplatter_sensor.sh.
# After a crash, rerun the same command: finished scenes are skipped, and a
# scene left half-done is moved aside to <scene>_incomplete_<time> and retrained.

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
DEPTH_MODE=${DEPTH_MODE:-mono}
EXP_NAME=${EXP_NAME:-dnsplatter_monodepth/iphone_pcd/only_rgb}

if [ $# -gt 0 ]; then SCENES=("$@"); else SCENES=("coffee_room" "honka" "kokko" "sauna" "computer" "vr_room"); fi

for item in "${SCENES[@]}"
do
  run_dir=outputs/$EXP_NAME/dn-splatter/$item
  if [ -f $run_dir/mesh_o3/mesh_metrics.json ] && [ -f $run_dir/pred_mesh/mesh_metrics.json ]; then
    echo "=== $item already done, skipping ==="
    continue
  fi
  # A folder without final metrics is left over from a crashed run: keep it aside.
  [ -d $run_dir ] && mv $run_dir ${run_dir}_incomplete_$(date +%Y%m%d_%H%M%S)

  echo "=== $item: DN-Splatter, depth mode $DEPTH_MODE ==="

  ns-train dn-splatter --experiment-name $EXP_NAME \
    --vis tensorboard \
    --timestamp $item \
    --pipeline.model.use-depth-loss True \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss True \
    --pipeline.model.use-normal-tv-loss True \
    --pipeline.model.use-disparity-normal-loss False \
    --pipeline.model.use_opacity_shrink False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.normal-lambda 0.1 \
    --pipeline.model.normal-loss-type L1 \
    mushroom \
    --data $DATA_DIR/$item \
    --depth-mode $DEPTH_MODE \
    --load-normals True \
    || fail ns-train

  gs-mesh dn \
    --load-config $run_dir/config.yml \
    --output-dir $run_dir/pred_mesh \
    || fail "gs-mesh dn"

  gs-mesh o3dtsdf \
    --load-config $run_dir/config.yml \
    --output-dir $run_dir/mesh_o3 \
    || fail "gs-mesh o3dtsdf"

  ns-eval \
        --load-config $run_dir/config.yml \
        --output-path $run_dir/eval/results.json \
        --render-output-path $run_dir/ \
        || fail ns-eval

  # Each writes mesh_metrics.json next to its mesh.
  python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
      --gt_mesh_path $DATA_DIR/$item \
      --pred_mesh_path $run_dir/pred_mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply \
      --device iphone \
      || fail "eval_mesh (Poisson)"
  python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
      --gt_mesh_path $DATA_DIR/$item \
      --pred_mesh_path $run_dir/mesh_o3/Open3dTSDFfusion_mesh.ply \
      --device iphone \
      || fail "eval_mesh (TSDF)"

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
for scene_dir in "$EXP_DIR"/*/; do
    [ -d "$scene_dir" ] || continue
    scene_name=$(basename "$scene_dir")
    # Leftovers from crashed runs are not results.
    case "$scene_name" in *_incomplete_*) continue ;; esac
    echo "Scene: $scene_name"
    echo "---------------------------------------------------------"

    RESULTS_FILE="$scene_dir/eval/results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "  [eval/results.json]"
        for key in "${DESIRED_METRICS[@]}"; do
            val=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['results'].get(sys.argv[2], 'N/A'))" "$RESULTS_FILE" "$key")
            printf "    %-22s : %s\n" "$key" "$val"
        done
    else
        echo "  [eval/results.json] -> Not found"
    fi
    echo ""

    for mesh in pred_mesh mesh_o3; do
        MESH_FILE="$scene_dir/$mesh/mesh_metrics.json"
        if [ -f "$MESH_FILE" ]; then
            echo "  [$mesh/mesh_metrics.json]"
            python -c "import json,sys; [print(f'    {k:<22} : {v}') for k, v in json.load(open(sys.argv[1])).items()]" "$MESH_FILE"
        else
            echo "  [$mesh/mesh_metrics.json] -> Not found"
        fi
    done
    echo "========================================================="
done
