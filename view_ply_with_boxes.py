import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d


def make_box_lines(center, dims, yaw, color=(1, 0, 0)):
    """
    Build an oriented 3D bounding box as an Open3D LineSet.
    center: (x, y, z)
    dims: (dx, dy, dz) # length along x, y, z in the same coord frame
    yaw: rotation around z axis (in radians)
    color: RGB tuple
    """
    cx, cy, cz = center
    dx, dy, dz = dims

    # 8 corners in box local coordinates (centered at origin)
    x = dx / 2.0
    y = dy / 2.0
    z = dz / 2.0
    corners = np.array([
        [ x,  y,  z],
        [ x, -y,  z],
        [-x, -y,  z],
        [-x,  y,  z],
        [ x,  y, -z],
        [ x, -y, -z],
        [-x, -y, -z],
        [-x,  y, -z],
    ], dtype=np.float32)

    # Rotation around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw,  cos_yaw, 0.0],
        [0.0,     0.0,     1.0],
    ], dtype=np.float32)

    # Rotate + translate to world coordinates
    corners_rot = (R @ corners.T).T
    corners_world = corners_rot + np.array([cx, cy, cz], dtype=np.float32)

    # Define edges between corners (12 edges of a box)
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # top face
        [4, 5], [5, 6], [6, 7], [7, 4],  # bottom face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(np.array(edges, dtype=np.int32))
    colors = np.tile(np.array(color, dtype=np.float32), (len(edges), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def load_prediction_boxes(json_path, score_thr=0.3):
    """
    Load predicted boxes from our JSON format:
      {
        "bboxes_3d": [[x, y, z, dx, dy, dz, yaw], ...],
        "scores_3d": [...],
        "labels_3d": [...]
      }
    Returns list of (center, dims, yaw) filtered by score_thr.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    bboxes = np.asarray(data["bboxes_3d"], dtype=np.float32)
    scores = np.asarray(data["scores_3d"], dtype=np.float32)

    keep = scores >= score_thr
    bboxes = bboxes[keep]

    centers = bboxes[:, 0:3]
    dims = bboxes[:, 3:6]
    yaws = bboxes[:, 6]

    return centers, dims, yaws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True, help="Path to .ply file")
    parser.add_argument("--json", required=True, help="Path to matching .json file")
    parser.add_argument("--score-thr", type=float, default=0.3,
                        help="Only show boxes with score >= this")
    args = parser.parse_args()

    ply_path = Path(args.ply)
    json_path = Path(args.json)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(ply_path))

    # Load prediction boxes
    centers, dims, yaws = load_prediction_boxes(json_path, score_thr=args.score_thr)
    print(f"Loaded {len(centers)} boxes from {json_path}")

    # Build LineSets for all boxes
    geometries = [pcd]
    for center, dim, yaw in zip(centers, dims, yaws):
        box_lines = make_box_lines(center, dim, yaw, color=(1, 0, 0))  # red boxes
        geometries.append(box_lines)

    # Visualize
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()