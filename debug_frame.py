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
from project_color import load_scan_data, compute_camera_pose_in_lidar_frame, load_d455_image
from utils.projection import project_points_to_image

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_DIST = CFG["projection"]["max_distance_m"]


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
    args = parser.parse_args()

    scan_dir = args.scan_dir.expanduser()
    data = load_scan_data(scan_dir)

    frame_indices = args.frame
    multi = len(frame_indices) > 1

    # Pre-allocate color buffer over the full cloud — each point colored once
    n_pts = len(data["points"])
    color_buf = np.full((n_pts, 3), -1.0)  # -1 = uncolored

    for frame_idx in frame_indices:
        frame_num = data["frame_numbers"][frame_idx]
        print(f"Processing frame index {frame_idx} (frame_{frame_num:03d})")

        rgb = load_d455_image(scan_dir, frame_num)
        if rgb is None:
            print(f"  WARNING: could not load frame_{frame_num:03d}_rgb.png, skipping")
            continue

        cam_mat = data["camera_matrix"]
        if data["dist_coeffs"] is not None:
            rgb = cv2.undistort(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                cam_mat, data["dist_coeffs"]
            )
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        H, W = rgb.shape[:2]

        T_cam = compute_camera_pose_in_lidar_frame(frame_idx, data)
        pixels, pt_indices, depths = project_points_to_image(
            data["points"], T_cam,
            data["fx"], data["fy"], data["cx"], data["cy"],
            H, W
        )

        if len(depths) > 0:
            mask = depths < MAX_DIST
            pixels = pixels[mask]
            pt_indices = pt_indices[mask]
            depths = depths[mask]

        print(f"  {len(depths):,} LiDAR points visible")

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        sampled_rgb = rgb_bgr[pixels[:, 1], pixels[:, 0]]  # BGR

        # Only color uncolored points (first frame to see a point wins)
        uncolored = color_buf[pt_indices, 0] < 0
        color_buf[pt_indices[uncolored]] = sampled_rgb[uncolored, ::-1] / 255.0

        # 2D panels — only for single frame
        if not multi:
            dot_colors = depth_colormap(depths)

            lidar_panel = np.zeros((H, W, 3), dtype=np.uint8)
            overlay_panel = rgb_bgr.copy()
            projected_img_panel = np.zeros((H, W, 3), dtype=np.uint8)

            r = args.dot_radius
            for (u, v), depth_color, rgb_color in zip(pixels, dot_colors, sampled_rgb):
                dc = tuple(int(x) for x in depth_color)
                rc = tuple(int(x) for x in rgb_color)
                cv2.circle(lidar_panel, (u, v), r, dc, -1)
                cv2.circle(overlay_panel, (u, v), r, dc, -1)
                cv2.circle(projected_img_panel, (u, v), r, rc, -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for panel, label in [(rgb_bgr, "Camera RGB"),
                                 (lidar_panel, f"LiDAR projection ({len(depths):,} pts)"),
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
    colored_mask = color_buf[:, 0] >= 0
    combined_pts = data["points"][colored_mask]
    combined_rgb = color_buf[colored_mask]
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
