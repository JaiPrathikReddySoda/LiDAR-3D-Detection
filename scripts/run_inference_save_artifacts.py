import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.structures import Det3DDataSample


def read_lidar_xyz(file_path: Path, dataset_name: str) -> np.ndarray:
    """Read a lidar file and return only XYZ coordinates."""
    buffer = np.fromfile(str(file_path), dtype=np.float32)
    ds = dataset_name.lower()

    if ds == "kitti":
        pts = buffer.reshape(-1, 4)  # x, y, z, intensity
    elif ds == "nuscenes":
        pts = buffer.reshape(-1, 5)  # x, y, z, intensity, ring
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    return pts[:, :3]


def dump_ply_cloud(xyz_points: np.ndarray, target_path: Path) -> None:
    """Write XYZ points to a PLY file using Open3D."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz_points)
    # use a neutral gray for all points
    cloud.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.io.write_point_cloud(str(target_path), cloud)


def predictions_to_dict(sample: Det3DDataSample) -> dict:
    """Convert Det3DDataSample into plain Python types for JSON."""
    pred = sample.pred_instances_3d
    return {
        "bboxes_3d": pred.bboxes_3d.tensor.cpu().numpy().tolist(),
        "scores_3d": pred.scores_3d.cpu().numpy().tolist(),
        "labels_3d": pred.labels_3d.cpu().numpy().tolist(),
    }


def collect_lidar_files(root: Path, dataset_name: str) -> list[Path]:
    """Return sorted lidar file list for the chosen dataset."""
    ds = dataset_name.lower()
    if ds == "kitti":
        pattern = "*.bin"
    elif ds == "nuscenes":
        pattern = "*.pcd.bin"
    else:
        raise ValueError("dataset must be either 'kitti' or 'nuscenes'")

    return sorted(root.glob(pattern))


def build_detector_model(cfg_path: str, ckpt_path: str, device: str, frame_dir: Path):
    """Initialize the mmdet3d model and tweak visualizer save dir if possible."""
    print("Initializing 3D detector model...")
    model = init_model(cfg_path, ckpt_path, device=device)

    # Try to redirect visualizer output to frames folder (if config supports it)
    try:
        vis_cfg = model.cfg.visualizer
        backends = vis_cfg.get("vis_backends", [])
        if backends and isinstance(backends[0], dict):
            init_kwargs = backends[0].get("init_kwargs", {})
            init_kwargs["save_dir"] = str(frame_dir)
            backends[0]["init_kwargs"] = init_kwargs
            vis_cfg["vis_backends"] = backends
    except Exception:
        # Non-fatal; the later show_results(out_dir=...) still works if enabled.
        pass

    return model


def process_lidar_sequence(
    config_path: str,
    checkpoint_path: str,
    lidar_dir: str,
    output_dir: str,
    device: str = "cuda:0",
    max_samples: int | None = 50,
    dataset: str = "kitti",
) -> None:
    """
    Iterate over lidar frames and run 3D detection.

    For each frame:
      - run inference
      - save predictions to JSON (preds/)
      - save raw point cloud to PLY (ply/)
      - optionally store visualization frames (frames/)
    """
    lidar_root = Path(lidar_dir)
    out_root = Path(output_dir)

    frames_root = out_root / "frames"
    ply_root = out_root / "ply"
    preds_root = out_root / "preds"

    for sub in (frames_root, ply_root, preds_root):
        sub.mkdir(parents=True, exist_ok=True)

    lidar_files = collect_lidar_files(lidar_root, dataset)
    if max_samples is not None:
        lidar_files = lidar_files[:max_samples]

    print(f"Found {len(lidar_files)} lidar files in {lidar_root}")

    model = build_detector_model(config_path, checkpoint_path, device, frames_root)

    t0 = time.time()
    total = len(lidar_files)

    for idx, lidar_path in enumerate(lidar_files, start=1):
        frame_id = lidar_path.stem
        print(f"[{idx}/{total}] Processing frame: {frame_id}")

        # mmdet3d >=1.4 returns (result, data)
        result, input_data = inference_detector(model, str(lidar_path))

        if not isinstance(result, Det3DDataSample):
            raise TypeError(f"Expected Det3DDataSample, got {type(result)}")

        # --- write prediction JSON ---
        json_out = preds_root / f"{frame_id}.json"
        with json_out.open("w") as fp:
            json.dump(predictions_to_dict(result), fp, indent=2)

        # --- write PLY with raw points ---
        xyz = read_lidar_xyz(lidar_path, dataset)
        ply_out = ply_root / f"{frame_id}.ply"
        dump_ply_cloud(xyz, ply_out)

        # --- optional visualization (kept commented, same behavior as original) ---
        # model.show_results(
        #     input_data,
        #     result,
        #     out_dir=str(frames_root),
        #     show=False,
        #     pred_score_thr=0.3,
        # )

    elapsed = time.time() - t0
    fps = (total / elapsed) if elapsed > 0 else 0.0
    print(f"\nFinished {total} frames in {elapsed:.2f} s  ({fps:.2f} FPS)")


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run mmdet3d inference on lidar data.")
    parser.add_argument("--config", required=True, help="Path to model config (.py)")
    parser.add_argument("--checkpoint", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--pcd-dir", required=True, help="Directory containing lidar files")
    parser.add_argument("--out-dir", required=True, help="Directory to store outputs")
    parser.add_argument("--device", default="cuda:0", help="Device spec, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of frames to process (None = all)",
    )
    parser.add_argument(
        "--dataset",
        default="kitti",
        help="Dataset type: 'kitti' or 'nuscenes'",
    )
    return parser.parse_args()


def cli_entry() -> None:
    args = parse_args()
    process_lidar_sequence(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        lidar_dir=args.pcd_dir,
        output_dir=args.out_dir,
        device=args.device,
        max_samples=args.max_samples,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    cli_entry()
