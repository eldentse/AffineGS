"""Generate disparity maps for MuSHRoom or ScanNet++ using Depth Anything V2.

Usage:
    python dn_splatter/scripts/generate_disparity.py \
        --data /path/to/room_datasets/coffee_room \
        --mode iphone \
        --checkpoint /path/to/depth_anything_v2_vitl.pth

For ScanNet++, pass the scene folder (e.g. datasets/scannetpp/data/b20a261fdf);
disparity maps are written to <scene>/iphone/disparity/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import tyro


@dataclass
class GenerateDisparity:
    data: Path
    """Path to the room dataset folder (e.g. datasets/room_datasets/coffee_room)"""
    mode: str = "iphone"
    """Device mode: iphone or kinect"""
    checkpoint: Path = Path(
        "./depth_anything_ckpts/depth_anything_v2_vitl.pth"
    )
    """Path to Depth Anything V2 ViT-L checkpoint"""
    input_size: int = 518
    """Input resolution for Depth Anything V2"""

    def main(self) -> None:
        # Add parent so that depth_anything_v2 can be imported
        dn_splatter_root = Path(__file__).resolve().parent.parent
        if str(dn_splatter_root) not in sys.path:
            sys.path.insert(0, str(dn_splatter_root))

        from depth_anything_v2.dpt import DepthAnythingV2

        # Load model
        print("Loading Depth Anything V2 (ViT-L)...")
        model = DepthAnythingV2(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        state = torch.load(str(self.checkpoint), map_location="cuda")
        model.load_state_dict(state)
        model.eval().half().to("cuda")
        print("Model loaded.")

        # MuSHRoom: <scene>/<mode>/{long,short}_capture/images/
        # ScanNet++: <scene>/<mode>/rgb/
        base = self.data / self.mode
        captures = []
        for name in ["long_capture", "short_capture"]:
            cap_dir = base / name
            if cap_dir.exists():
                captures.append(cap_dir)
        if not captures and (base / "rgb").exists():
            captures.append(base)

        if not captures:
            raise FileNotFoundError(
                f"No capture directories or rgb/ folder found under {base}"
            )

        for cap_dir in captures:
            images_dir = cap_dir / "images" if (cap_dir / "images").exists() else cap_dir / "rgb"
            if not images_dir.exists():
                print(f"Skipping {cap_dir.name}: no images/ or rgb/ folder")
                continue

            disparity_dir = cap_dir / "disparity"
            disparity_dir.mkdir(exist_ok=True, parents=True)

            image_paths = sorted(images_dir.iterdir())
            image_paths = [
                p for p in image_paths if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]

            print(f"\nProcessing {cap_dir.name}: {len(image_paths)} images -> {disparity_dir}")
            for img_path in tqdm(image_paths, desc=cap_dir.name):
                raw_image = cv2.imread(str(img_path))
                if raw_image is None:
                    print(f"  Warning: could not read {img_path.name}, skipping")
                    continue

                disparity = model.infer_image(raw_image, input_size=self.input_size)
                disparity = np.nan_to_num(
                    disparity, nan=0.0, posinf=0.0, neginf=0.0
                )

                # Clip to the 1st/99th percentiles, then min-max normalize to [0, 1].
                # disparity_normal_loss expects disparity in [0, 1].
                disp_min = np.percentile(disparity[disparity > 0], 1) if (disparity > 0).any() else 1.0
                disp_max = np.percentile(disparity[disparity > 0], 99) if (disparity > 0).any() else 1.0
                disparity = np.clip(disparity, disp_min, disp_max)
                disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)

                out_path = disparity_dir / (img_path.stem + ".npy")
                np.save(str(out_path), disparity.astype(np.float32))

                # Save JPEG visualization (0-255)
                cv2.imwrite(
                    str(disparity_dir / (img_path.stem + ".jpeg")),
                    (disparity * 255).astype(np.uint8),
                )

        print("\nDone.")


if __name__ == "__main__":
    tyro.cli(GenerateDisparity).main()
