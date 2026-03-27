#!/usr/bin/env python3
"""
Quick debug: show camera image vs LiDAR projection vs overlay for one frame.

Usage:
  python3 debug_frame.py <scan_dir> [--frame N]

Displays a 4-panel window:
  Panel 1: raw RGB camera frame
  Panel 2: LiDAR points projected onto black (depth-colored)
  Panel 3: overlay (LiDAR depth-colored dots on camera background)
  Panel 4: image projected onto LiDAR (LiDAR dots colored with actual RGB pixels, on black)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_color import load_scan_data, compute_camera_pose_in_lidar_frame, load_d455_image, load_d455_depth
from utils.projection import project_points_to_image, color_points_from_frame

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_DIST = CFG["projection"]["max_distance_m"]
ZBUF_TOL = CFG["projection"]["zbuffer_tolerance"]
DEPTH_THRESHOLD = CFG["projection"].get("depth_consistency_threshold", 0.10)


def depth_colormap(depths, max_dist=MAX_DIST):
    """Map depths to BGR colors using a jet colormap."""
    norm = np.clip(depths / max_dist, 0, 1)
    # jet: 0=blue, 0.5=green, 1=red
    colors = np.zeros((len(depths), 3), dtype=np.uint8)
    colors[:, 0] = (np.clip(1.5 - abs(norm * 4 - 3), 0, 1) * 255).astype(np.uint8)  # B
    colors[:, 1] = (np.clip(1.5 - abs(norm * 4 - 2), 0, 1) * 255).astype(np.uint8)  # G
    colors[:, 2] = (np.clip(1.5 - abs(norm * 4 - 1), 0, 1) * 255).astype(np.uint8)  # R
    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan_dir", type=Path)
    parser.add_argument("--frame", type=int, nargs="+", default=[0],
                        help="Frame index/indices (0-based) to visualize/combine")
    parser.add_argument("--dot-radius", type=int, default=2,
                        help="Dot radius for projected LiDAR points")
    parser.add_argument("--sharpness", type=float, default=6.0,
                        help="Weight curve sharpness (1=smooth/blurry, 6=balanced, 20=first-wins-like)")
    args = parser.parse_args()

    scan_dir = args.scan_dir.expanduser()
    data = load_scan_data(scan_dir)

    frame_indices = args.frame
    multi = len(frame_indices) > 1

    # Pre-allocate color buffer — center-weighted blending for multi-frame
    n_pts = len(data["points"])
    color_sum = np.zeros((n_pts, 3), dtype=np.float64)
    weight_sum = np.zeros(n_pts, dtype=np.float64)

    for frame_idx in frame_indices:
        frame_num = data["frame_numbers"][frame_idx]
        print(f"Processing frame index {frame_idx} (frame_{frame_num:03d})")

        rgb = load_d455_image(scan_dir, frame_num)
        if rgb is None:
            print(f"  WARNING: could not load frame_{frame_num:03d}_rgb.png, skipping")
            continue

        cam_mat = data["camera_matrix"]
        if data["dist_coeffs"] is not None:
            rgb = cv2.undistort(rgb, cam_mat, data["dist_coeffs"])

        H, W = rgb.shape[:2]

        T_cam = compute_camera_pose_in_lidar_frame(frame_idx, data)
        depth_img = load_d455_depth(scan_dir, frame_num)

        # Use proper z-buffer filtering (no depth consistency — too aggressive)
        pt_indices, colors, depths, pixels = color_points_from_frame(
            data["points"], T_cam,
            data["fx"], data["fy"], data["cx"], data["cy"],
            rgb, ZBUF_TOL,
        )

        if len(pt_indices) == 0:
            print(f"  0 LiDAR points visible")
            continue

        # Filter by max distance
        dist_mask = depths < MAX_DIST
        pt_indices = pt_indices[dist_mask]
        colors = colors[dist_mask]
        pixels = pixels[dist_mask]
        depths = depths[dist_mask]

        print(f"  {len(pt_indices):,} LiDAR points visible (z-buffered + depth-checked)")

        colors_f = colors.astype(np.float64) / 255.0

        # Center-weighted blending
        dx = (pixels[:, 0] - data["cx"]) / (W / 2)
        dy = (pixels[:, 1] - data["cy"]) / (H / 2)
        dist = np.clip(np.sqrt(dx**2 + dy**2), 0, 1)
        weights = (0.5 * (1.0 + np.cos(np.pi * dist))) ** args.sharpness

        np.add.at(color_sum, (pt_indices, slice(None)),
                  colors_f * weights[:, np.newaxis])
        np.add.at(weight_sum, pt_indices, weights)

        # 2D panels — only for single frame
        if not multi:
            dot_colors = depth_colormap(depths)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            colors_bgr = colors[:, ::-1]  # RGB→BGR

            lidar_panel = np.zeros((H, W, 3), dtype=np.uint8)
            overlay_panel = rgb_bgr.copy()
            projected_img_panel = np.zeros((H, W, 3), dtype=np.uint8)

            r = args.dot_radius
            for (u, v), depth_color, rgb_color in zip(pixels, dot_colors, colors_bgr):
                dc = tuple(int(x) for x in depth_color)
                rc = tuple(int(x) for x in rgb_color)
                cv2.circle(lidar_panel, (u, v), r, dc, -1)
                cv2.circle(overlay_panel, (u, v), r, dc, -1)
                cv2.circle(projected_img_panel, (u, v), r, rc, -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for panel, label in [(rgb_bgr, "Camera RGB"),
                                 (lidar_panel, f"LiDAR projection ({len(depths):,} pts, z-buffered)"),
                                 (overlay_panel, "Overlay"),
                                 (projected_img_panel, "Image projected onto LiDAR")]:
                cv2.putText(panel, label, (10, 30), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(panel, label, (10, 30), font, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

            combined = np.hstack([rgb_bgr, lidar_panel, overlay_panel, projected_img_panel])
            max_w = 3840
            if combined.shape[1] > max_w:
                scale = max_w / combined.shape[1]
                combined = cv2.resize(combined, None, fx=scale, fy=scale)

            cv2.imshow(f"Debug frame {frame_idx} | frame_{frame_num:03d}", combined)
            print("Press any key to close 2D view.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # --- Build and save combined 3D cloud ---
    colored_mask = weight_sum > 0
    combined_pts = data["points"][colored_mask]
    combined_rgb = color_sum[colored_mask] / weight_sum[colored_mask, np.newaxis]
    combined_rgb = np.clip(combined_rgb, 0, 1)
    print(f"Total unique colored points: {len(combined_pts):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    pcd.colors = o3d.utility.Vector3dVector(combined_rgb)

    if len(frame_indices) == 1:
        label = str(frame_indices[0])
    else:
        label = f"frames{frame_indices[0]}-{frame_indices[-1]}"
    out_path = scan_dir / f"debug_{label}.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved: {out_path}")
    import subprocess
    subprocess.Popen(["meshlab", str(out_path)])


if __name__ == "__main__":
    main()
