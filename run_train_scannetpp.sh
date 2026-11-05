#!/bin/bash
# TRAIN SCRIPT — ScanNet++
# Trains AffineGS on the two ScanNet++ scenes of the paper (iPhone sequences),
# extracts meshes with Open3D TSDF fusion (mesh_o3/) and Poisson (pred_mesh/),
# and evaluates them against the laser-scan mesh.
#
# Needs, per scene, the iPhone frames extracted with the ScanNet++ toolkit
# (iphone/rgb, iphone/depth) and disparity maps from generate_disparsity.py
# (iphone/disparity).
#
# Usage: bash run_train_scannetpp.sh [scene ...]   (default: both scenes)
#        SCANNETPP_DATA=<path to scannetpp/data> bash run_train_scannetpp.sh
#        EXP_NAME=<name> bash run_train_scannetpp.sh ...   to write elsewhere
# METHOD=dnsplatter trains DN-Splatter with sensor depth and Omnidata normals
# instead; see run_train_scannetpp_dnsplatter.sh. METHOD=rgb_only trains plain
# 3DGS with no geometry losses, as a control.
# LOAD_EVERY=<n> keeps every n-th training frame (dataparser default: 5).
# RES_SCALE=<f> trains at f times the image resolution (e.g. 0.5 for 960x720).
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


DATA_DIR=${SCANNETPP_DATA:-./datasets/scannetpp/data}
METHOD=${METHOD:-affinegs}

if [ "$METHOD" = "dnsplatter" ]; then
  # DN-Splatter's own ScanNet++ recipe: sensor depth (lambda 0.2) and monocular
  # normals, which the loader generates with Omnidata into iphone/normals_from_pretrain.
  EXP_NAME=${EXP_NAME:-dnsplatter_sensordepth_scannetpp/iphone/only_rgb}
  SETTINGS="DN-Splatter, sensor depth"
  MODEL_ARGS="--pipeline.model.use-depth-loss True
    --pipeline.model.depth-lambda 0.2
    --pipeline.model.use-normal-loss True
    --pipeline.model.use-normal-tv-loss True
    --pipeline.model.use-disparity-normal-loss False
    --pipeline.model.normal-supervision mono
    --pipeline.model.normal-lambda 0.1
    --pipeline.model.normal-loss-type L1"
  DATA_ARGS="--depth-mode sensor --load-normals True"
elif [ "$METHOD" = "rgb_only" ]; then
  EXP_NAME=${EXP_NAME:-rgbonly_scannetpp/iphone/only_rgb}
  SETTINGS="no geometry losses (control)"
  MODEL_ARGS="--pipeline.model.use-depth-loss False
    --pipeline.model.use-normal-loss False
    --pipeline.model.use-normal-tv-loss False
    --pipeline.model.use-disparity-normal-loss False"
  DATA_ARGS="--depth-mode none --load-normals False"
else
  # The paper's ScanNet++ settings (Table 11). They were chosen on b20a261fdf and
  # confirmed on 8b5caf3398: half resolution (960x720, close to the
  # MuSHRoom frames the other settings were tuned on) and a quarter of the MuSHRoom
  # loss weights. Against full resolution with the MuSHRoom weights (0.1 / 0.08 / 0.4),
  # the two-scene average went from Chamfer-L1 0.157 / F-score 0.268 to 0.107 / 0.321.
  # Each can be overridden from the environment (RADIUS, EPS, GRAD, NORMAL, DEPTH,
  # RES_SCALE=1 for full resolution).
  EXP_NAME=${EXP_NAME:-affinegs_scannetpp/iphone/half_res}
  RADIUS=${RADIUS:-600}
  EPS=${EPS:-0.1}
  GRAD=${GRAD:-0.025}
  NORMAL=${NORMAL:-0.02}
  DEPTH=${DEPTH:-0.1}
  RES_SCALE=${RES_SCALE:-0.5}
  SETTINGS="AffineGS, guided-blur-radius $RADIUS, guided-blur-eps $EPS, loss weights $GRAD / $NORMAL / $DEPTH"
  MODEL_ARGS="--pipeline.model.use-depth-loss False
    --pipeline.model.use-normal-loss False
    --pipeline.model.use-normal-tv-loss False
    --pipeline.model.use-disparity-normal-loss True
    --pipeline.model.disparity-depth-loss-lambda $GRAD
    --pipeline.model.disparity-render-normal-loss-lambda $NORMAL
    --pipeline.model.l1-weight $DEPTH
    --pipeline.model.guided-blur-radius $RADIUS
    --pipeline.model.guided-blur-eps $EPS"
  DATA_ARGS="--depth-mode disparity --load-normals False"
fi
[ -n "$LOAD_EVERY" ] && DATA_ARGS="$DATA_ARGS --load-every $LOAD_EVERY" && SETTINGS="$SETTINGS, load every $LOAD_EVERY"
[ -n "$RES_SCALE" ] && MODEL_ARGS="$MODEL_ARGS --pipeline.datamanager.camera-res-scale-factor $RES_SCALE" && SETTINGS="$SETTINGS, resolution x$RES_SCALE"

if [ $# -gt 0 ]; then SCENES=("$@"); else SCENES=("b20a261fdf" "8b5caf3398"); fi

for item in "${SCENES[@]}"
do
  run_dir=outputs/$EXP_NAME/dn-splatter/$item
  if [ -f $run_dir/mesh_o3/mesh_metrics.json ] && [ -f $run_dir/pred_mesh/mesh_metrics.json ]; then
    echo "=== $item already done, skipping ==="
    continue
  fi
  # A folder without final metrics is left over from a crashed run: keep it aside.
  [ -d $run_dir ] && mv $run_dir ${run_dir}_incomplete_$(date +%Y%m%d_%H%M%S)

  echo "=== $item: $SETTINGS ==="

  # MODEL_ARGS and DATA_ARGS are deliberately unquoted so they split into flags.
  ns-train dn-splatter --experiment-name $EXP_NAME \
    --vis tensorboard \
    --timestamp $item \
    $MODEL_ARGS \
    --pipeline.model.use_opacity_shrink False \
    scannetpp \
    --data $DATA_DIR \
    --sequence $item \
    $DATA_ARGS \
    || fail ns-train

  gs-mesh o3dtsdf \
    --load-config $run_dir/config.yml \
    --output-dir $run_dir/mesh_o3 \
    || fail "gs-mesh o3dtsdf"

  gs-mesh dn \
    --load-config $run_dir/config.yml \
    --output-dir $run_dir/pred_mesh \
    || fail "gs-mesh dn"

  ns-eval \
        --load-config $run_dir/config.yml \
        --output-path $run_dir/eval/results.json \
        --render-output-path $run_dir/ \
        || fail ns-eval

  # Each writes mesh_metrics.json next to its mesh. The loader writes the
  # training poses to iphone/transforms.json, which the culling step uses.
  for mesh in mesh_o3/Open3dTSDFfusion_mesh.ply pred_mesh/DepthAndNormalMapsPoisson_poisson_mesh.ply; do
    python dn_splatter/eval/eval_mesh_vis_cull.py \
        --gt-mesh-path $DATA_DIR/$item/scans/mesh_aligned_0.05.ply \
        --pred-mesh-path $run_dir/$mesh \
        --transformation-file $DATA_DIR/$item/iphone/transforms.json \
        --dataset-path $DATA_DIR/$item/iphone \
        || fail "eval_mesh ($mesh)"
  done

  # Let the GPU cool down before the next scene.
  sleep 120
done

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
        python -c "import json,sys; r=json.load(open(sys.argv[1]))['results']; [print(f'    {k:<22} : {v}') for k, v in r.items() if not k.endswith(('_std', 'fps', 'rays_per_sec'))]" "$RESULTS_FILE"
    else
        echo "  [eval/results.json] -> Not found"
    fi
    for mesh in mesh_o3 pred_mesh; do
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
