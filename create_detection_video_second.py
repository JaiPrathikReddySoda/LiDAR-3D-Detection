import argparse
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import time
import os


def make_box_lines(center, dims, yaw, color=(0, 1, 0)):
    """Create 3D bounding box as LineSet"""
    cx, cy, cz = center
    dx, dy, dz = dims
    x = dx / 2.0
    y = dy / 2.0
    z = dz / 2.0
    
    corners = np.array([
        [ x,  y,  z], [ x, -y,  z], [-x, -y,  z], [-x,  y,  z],
        [ x,  y, -z], [ x, -y, -z], [-x, -y, -z], [-x,  y, -z],
    ], dtype=np.float32)
    
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw,  cos_yaw, 0.0],
        [0.0,     0.0,     1.0],
    ], dtype=np.float32)
    
    corners_rot = (R @ corners.T).T
    corners_world = corners_rot + np.array([cx, cy, cz], dtype=np.float32)
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(np.array(edges, dtype=np.int32))
    colors = np.tile(np.array(color, dtype=np.float32), (len(edges), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def load_prediction_boxes(json_path, score_thr=0.3):
    """Load 3D bounding boxes from JSON file"""
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        bboxes = np.asarray(data["bboxes_3d"], dtype=np.float32)
        scores = np.asarray(data["scores_3d"], dtype=np.float32)
        
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        keep = scores >= score_thr
        bboxes = bboxes[keep]
        
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        if bboxes.ndim == 1:
            bboxes = bboxes.reshape(1, -1)
        
        centers = bboxes[:, 0:3]
        dims = bboxes[:, 3:6]
        yaws = bboxes[:, 6]
        
        return centers, dims, yaws
        
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return np.array([]), np.array([]), np.array([])


def colorize_by_height(pcd):
    """Color point cloud by height (Z coordinate)"""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd
    
    z_values = points[:, 2]
    z_min = np.percentile(z_values, 5)
    z_max = np.percentile(z_values, 95)
    
    z_normalized = np.clip((z_values - z_min) / (z_max - z_min + 1e-6), 0, 1)
    
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized
    colors[:, 1] = 1.0 - np.abs(z_normalized - 0.5) * 2
    colors[:, 2] = 1.0 - z_normalized
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="Create smooth video from point cloud sequence with 3D detections"
    )
    
    parser.add_argument("--ply_dir", required=True, help="Directory containing .ply files")
    parser.add_argument("--json_dir", required=True, help="Directory containing .json files")
    parser.add_argument("--output_dir", default="frames", help="Output frames directory")
    parser.add_argument("--video_name", default="detection.mp4", help="Output video filename")
    parser.add_argument("--fps", type=int, default=5, help="Input frame rate (lower = slower video)")
    parser.add_argument("--output_fps", type=int, default=30, help="Output video frame rate (for smoothness)")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--colorize", action="store_true", help="Color by height")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point size")
    parser.add_argument("--interpolate", action="store_true", help="Add motion interpolation for smoother video")
    parser.add_argument("--slowdown", type=float, default=1.0, help="Slow down factor (2.0 = half speed)")
    
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        print(f"ERROR: No .ply files found in {ply_dir}")
        return
    
    print(f"=" * 60)
    print(f"Found {len(ply_files)} PLY files")
    print(f"Input FPS: {args.fps} (lower = slower)")
    print(f"Output FPS: {args.output_fps}")
    print(f"Slowdown factor: {args.slowdown}x")
    print(f"Interpolation: {'ON' if args.interpolate else 'OFF'}")
    print(f"=" * 60)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Rendering", width=1920, height=1080, visible=True)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = args.point_size
    opt.line_width = 3.0

    first_pcd = o3d.io.read_point_cloud(str(ply_files[0]))
    
    if args.colorize:
        first_pcd = colorize_by_height(first_pcd)
    elif not first_pcd.has_colors():
        first_pcd.paint_uniform_color([0.9, 0.9, 0.9])
    
    vis.add_geometry(first_pcd)
    
    ctr = vis.get_view_control()
    ctr.set_front([0.3, 0, -0.95])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.35)
    
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1.0)

    pcd_geometry = first_pcd
    box_geometries = []

    print("\nRendering frames...")
    for idx, ply_path in enumerate(ply_files):
        current_pcd = o3d.io.read_point_cloud(str(ply_path))
        
        if args.colorize:
            current_pcd = colorize_by_height(current_pcd)
        elif not current_pcd.has_colors():
            current_pcd.paint_uniform_color([0.9, 0.9, 0.9])
        
        pcd_geometry.points = current_pcd.points
        pcd_geometry.colors = current_pcd.colors
        vis.update_geometry(pcd_geometry)
        
        for box in box_geometries:
            vis.remove_geometry(box, reset_bounding_box=False)
        box_geometries.clear()
        
        json_path = json_dir / f"{ply_path.stem}.json"
        num_boxes = 0
        
        if json_path.exists():
            centers, dims, yaws = load_prediction_boxes(json_path, args.score_thr)
            
            if len(centers) > 0:
                for c, d, y in zip(centers, dims, yaws):
                    box = make_box_lines(c, d, y, color=(0, 1, 0))
                    vis.add_geometry(box, reset_bounding_box=False)
                    box_geometries.append(box)
                num_boxes = len(centers)
        
        for _ in range(3):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05)
        
        frame_path = output_dir / f"frame_{idx:06d}.png"
        vis.capture_screen_image(str(frame_path), do_render=True)
        
        print(f"[{idx+1}/{len(ply_files)}] {ply_path.name} -> {frame_path.name} ({num_boxes} boxes)")

    vis.destroy_window()
    print(f"\n✓ All frames saved to '{output_dir}'")
    
    # Create video with smooth options
    print("\nCreating video...")
    
    if args.interpolate:
        # Motion interpolation for ultra-smooth video
        ffmpeg_cmd = (
            f"ffmpeg -y -r {args.fps} "
            f"-i {output_dir}/frame_%06d.png "
            f"-vf 'minterpolate=fps={args.output_fps}:mi_mode=mci,setpts={args.slowdown}*PTS' "
            f"-c:v libx264 -pix_fmt yuv420p "
            f"-preset slow -crf 18 "
            f"{args.video_name}"
        )
    else:
        # Simple slowdown without interpolation
        ffmpeg_cmd = (
            f"ffmpeg -y -r {args.fps} "
            f"-i {output_dir}/frame_%06d.png "
            f"-vf 'setpts={args.slowdown}*PTS' "
            f"-c:v libx264 -pix_fmt yuv420p -r {args.output_fps} "
            f"-preset slow -crf 18 "
            f"{args.video_name}"
        )
    
    ret = os.system(ffmpeg_cmd)
    
    if ret == 0:
        print(f"\n{'=' * 60}")
        print(f"✓ SUCCESS: Video created -> {args.video_name}")
        print(f"{'=' * 60}")
    else:
        print(f"\n✗ ERROR: ffmpeg failed")


if __name__ == "__main__":
    main()