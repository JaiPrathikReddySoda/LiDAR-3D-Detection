import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def build_wireframe_box(center_xyz, size_xyz, yaw_rad, rgb=(0.0, 1.0, 0.0)):
    """
    Create an Open3D LineSet representing a 3D bounding box.

    center_xyz: (x, y, z)
    size_xyz:   (dx, dy, dz)
    yaw_rad:    rotation around Z axis
    """
    cx, cy, cz = center_xyz
    sx, sy, sz = size_xyz

    # half-dimensions
    hx = sx * 0.5
    hy = sy * 0.5
    hz = sz * 0.5

    # corners in local box coordinates (Z up)
    local_corners = np.array(
        [
            [hx,  hy,  hz],  # 0
            [hx, -hy,  hz],  # 1
            [-hx, -hy, hz],  # 2
            [-hx,  hy,  hz], # 3
            [hx,  hy, -hz],  # 4
            [hx, -hy, -hz],  # 5
            [-hx, -hy, -hz], # 6
            [-hx,  hy, -hz], # 7
        ],
        dtype=np.float32,
    )

    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    rot_z = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rotated = (rot_z @ local_corners.T).T
    world_corners = rotated + np.array([cx, cy, cz], dtype=np.float32)

    # edges of the box
    line_indices = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],  # top face
            [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
            [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
        ],
        dtype=np.int32,
    )

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_corners)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)

    colors = np.tile(np.asarray(rgb, dtype=np.float32), (line_indices.shape[0], 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def parse_detection_json(json_file: Path, score_threshold: float = 0.3):
    """Load 3D bounding boxes, filter by score, and split into components."""
    with json_file.open("r") as f:
        raw = json.load(f)

    boxes = np.asarray(raw["bboxes_3d"], dtype=np.float32)
    scores = np.asarray(raw["scores_3d"], dtype=np.float32)

    valid_mask = scores >= score_threshold
    boxes = boxes[valid_mask]

    centers = boxes[:, 0:3]
    dims = boxes[:, 3:6]
    yaws = boxes[:, 6]
    return centers, dims, yaws


def apply_height_coloring(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Assign colors to points based on their Z coordinate."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd

    z = pts[:, 2]

    z_min = np.percentile(z, 5)
    z_max = np.percentile(z, 95)

    # normalize into [0, 1]
    norm = (z - z_min) / (z_max - z_min + 1e-6)
    norm = np.clip(norm, 0.0, 1.0)

    colors = np.zeros((pts.shape[0], 3), dtype=np.float32)
    # R grows with height
    colors[:, 0] = norm
    # G is strongest around the middle
    colors[:, 1] = 1.0 - np.abs(norm - 0.5) * 2.0
    # B decreases with height
    colors[:, 2] = 1.0 - norm

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def resolve_camera_preset(view_name: str = "birds_eye"):
    """
    Map a string preset to camera parameters.

    Returns:
        front (3,), lookat (3,), up (3,), zoom (float)
    """
    presets = {
        "birds_eye": ([0.3, 0.0, -0.95], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0.35),
        "top_down": ([0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0.5),
        "front": ([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.4),
        "side": ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.4),
        "diagonal": ([0.5, 0.5, -0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.4),
        "chase": ([0.8, 0.0, -0.6], [10.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.3),
        "high_angle": ([0.2, 0.0, -0.98], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0.4),
    }

    if view_name in presets:
        return presets[view_name]

    # custom numeric configuration "f1,f2,f3,up1,up2,up3,zoom"
    try:
        values = [float(v) for v in view_name.split(",")]
        front = values[0:3] if len(values) >= 3 else [0.3, 0.0, -0.95]
        lookat = [0.0, 0.0, 0.0]
        up = values[3:6] if len(values) >= 6 else [0.0, -1.0, 0.0]
        zoom = values[6] if len(values) >= 7 else 0.35
        return front, lookat, up, zoom
    except Exception:
        print(f"Invalid view preset: {view_name!r}, falling back to 'birds_eye'")
        return presets["birds_eye"]


def configure_camera(ctrl, front, lookat, up, zoom):
    """Apply the chosen camera parameters to an Open3D view controller."""
    ctrl.set_front(front)
    ctrl.set_lookat(lookat)
    ctrl.set_up(up)
    ctrl.set_zoom(zoom)


def init_visualizer():
    """Create and configure an Open3D visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Rendering", width=1920, height=1080, visible=True)

    render_opts = vis.get_render_option()
    render_opts.background_color = np.asarray([0.0, 0.0, 0.0])
    render_opts.point_size = 2.0
    render_opts.line_width = 3.0

    return vis


def encode_video_from_frames(frame_dir: Path, fps: int, video_path: str):
    """Use ffmpeg to turn saved PNG frames into a video file."""
    cmd = (
        f"ffmpeg -y -r {fps} "
        f"-i {frame_dir}/frame_%06d.png "
        f"-c:v libx264 -pix_fmt yuv420p -preset slow -crf 18 "
        f"{video_path}"
    )
    os.system(cmd)


def run_visualization(
    ply_dir: Path,
    json_dir: Path,
    frame_output_dir: Path,
    video_name: str,
    fps: int,
    score_thr: float,
    use_height_color: bool,
    view_preset: str,
):
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        print("No .ply files found in", ply_dir)
        return

    print(f"Found {len(ply_files)} frames.")
    print(f"Camera preset: {view_preset}")

    cam_front, cam_lookat, cam_up, cam_zoom = resolve_camera_preset(view_preset)
    print(f"Camera params -> front={cam_front}, up={cam_up}, zoom={cam_zoom}")

    # set up Open3D visualizer
    vis = init_visualizer()

    # load first frame
    first_cloud = o3d.io.read_point_cloud(str(ply_files[0]))
    if use_height_color:
        first_cloud = apply_height_coloring(first_cloud)
    elif not first_cloud.has_colors():
        first_cloud.paint_uniform_color([0.9, 0.9, 0.9])

    vis.add_geometry(first_cloud)

    cam_ctrl = vis.get_view_control()
    configure_camera(cam_ctrl, cam_front, cam_lookat, cam_up, cam_zoom)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.0)

    # references for updating
    pcd_geom = first_cloud
    active_boxes = []

    for frame_idx, ply_path in enumerate(ply_files):
        # read current point cloud
        current_cloud = o3d.io.read_point_cloud(str(ply_path))

        if use_height_color:
            current_cloud = apply_height_coloring(current_cloud)
        elif not current_cloud.has_colors():
            current_cloud.paint_uniform_color([0.9, 0.9, 0.9])

        pcd_geom.points = current_cloud.points
        pcd_geom.colors = current_cloud.colors
        vis.update_geometry(pcd_geom)

        # clear previous boxes
        for box in active_boxes:
            vis.remove_geometry(box, reset_bounding_box=False)
        active_boxes.clear()

        # add new boxes from detection JSON, if present
        det_json = json_dir / f"{ply_path.stem}.json"
        if det_json.exists():
            centers, dims, yaws = parse_detection_json(det_json, score_thr)
            for center, dim, yaw in zip(centers, dims, yaws):
                box_lines = build_wireframe_box(center, dim, yaw, rgb=(0.0, 1.0, 0.0))
                vis.add_geometry(box_lines, reset_bounding_box=False)
                active_boxes.append(box_lines)

        # camera sometimes drifts after geometry updates; re-apply params
        configure_camera(cam_ctrl, cam_front, cam_lookat, cam_up, cam_zoom)

        # let Open3D render a few iterations
        for _ in range(3):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)

        # save frame as PNG
        frame_path = frame_output_dir / f"frame_{frame_idx:06d}.png"
        vis.capture_screen_image(str(frame_path), do_render=True)
        print(f"Saved {frame_idx + 1}/{len(ply_files)} -> {frame_path.name}")

    vis.destroy_window()

    print("\nEncoding video...")
    encode_video_from_frames(frame_output_dir, fps, video_name)
    print(f"✓ Video created: {video_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render PLY frames with JSON 3D detections and export a video."
    )
    parser.add_argument("--ply_dir", required=True, help="Directory containing .ply point clouds")
    parser.add_argument("--json_dir", required=True, help="Directory containing detection JSON files")
    parser.add_argument("--output_dir", default="frames", help="Directory to store rendered frames")
    parser.add_argument("--video_name", default="detection.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the video")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Score threshold for boxes")
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Color point cloud according to point height (Z axis)",
    )
    parser.add_argument(
        "--view",
        default="birds_eye",
        help=(
            "Camera preset: birds_eye, top_down, front, side, diagonal, "
            "chase, high_angle, or a custom 'f1,f2,f3,up1,up2,up3,zoom' string"
        ),
    )
    return parser


def cli_main():
    parser = build_arg_parser()
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    json_dir = Path(args.json_dir)
    frame_dir = Path(args.output_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    run_visualization(
        ply_dir=ply_dir,
        json_dir=json_dir,
        frame_output_dir=frame_dir,
        video_name=args.video_name,
        fps=args.fps,
        score_thr=args.score_thr,
        use_height_color=args.colorize,
        view_preset=args.view,
    )


if __name__ == "__main__":
    cli_main()