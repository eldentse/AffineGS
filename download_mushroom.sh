DATA_DIR=./datasets/room_datasets
SCENES=("coffee_room" "honka" "kokko" "sauna" "computer" "vr_room")

# True when every image of the scene's iPhone captures has a disparity map.
disparity_done() {
    for cap in long_capture short_capture; do
        d=$DATA_DIR/$1/iphone/$cap
        n_img=$(ls "$d/images" 2>/dev/null | grep -ciE '\.(jpg|jpeg|png)$')
        n_disp=$(ls "$d/disparity" 2>/dev/null | grep -c '\.npy$')
        [ "$n_img" -gt 0 ] && [ "$n_img" -eq "$n_disp" ] || return 1
    done
}

# Reference depth maps used by the depth evaluation (one archive for all rooms).
if [ ! -d "$DATA_DIR/${SCENES[0]}/iphone/long_capture/reference_depth" ]; then
    python dn_splatter/data/mushroom_utils/reference_depth_download.py
fi

# Steps that already finished are skipped, so the script can be rerun.
for item in "${SCENES[@]}"
do
    if [ ! -f "$DATA_DIR/$item/gt_mesh.ply" ] || [ ! -f "$DATA_DIR/$item/iphone/long_capture/transformations_colmap.json" ]; then
        python dn_splatter/data/download_scripts/mushroom_download.py --room-name $item
    fi

    if ! disparity_done $item; then
        python dn_splatter/generate_disparsity.py \
        --data $DATA_DIR/$item
    fi
done

# The downloaders print errors but carry on, so check that every scene has
# what training and evaluation need.
missing=0
for item in "${SCENES[@]}"
do
    for f in gt_mesh.ply icp_iphone.json \
             iphone/long_capture/transformations_colmap.json \
             iphone/long_capture/reference_depth
    do
        if [ ! -e "$DATA_DIR/$item/$f" ]; then
            echo "MISSING: $DATA_DIR/$item/$f"
            missing=1
        fi
    done
    if ! disparity_done $item; then
        echo "MISSING: disparity maps for some $item images"
        missing=1
    fi
done
if [ $missing -eq 1 ]; then
    echo "Some files are missing. Delete the affected archives in datasets/ (e.g. datasets/computer_mesh_pd.tar.gz) and run this script again." >&2
    exit 1
fi
echo "All ${#SCENES[@]} scenes are ready."
