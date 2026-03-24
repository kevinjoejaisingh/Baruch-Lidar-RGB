#!/usr/bin/env python3
"""
Project color from D455 RGB frames onto LiDAR point cloud.

Uses the extrinsic calibration (T_lidar_from_d455) and LiDAR trajectory
to determine camera poses in the LiDAR world frame, then projects each
frame onto the point cloud with Z-buffer filtering.

Usage:
  python project_color.py <scan_dir>                    # Full projection
  python project_color.py <scan_dir> --frame 50 --visualize  # Single-frame debug
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from utils.projection import (color_points_from_frame, color_points_multi_frame,
                               project_points_to_image)
from utils.timestamps import interpolate_pose_at_timestamp

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

ZBUF_TOL = CFG["projection"]["zbuffer_tolerance"]
MAX_DIST = CFG["projection"]["max_distance_m"]
DEFAULT_COLOR = tuple(CFG["projection"]["default_color"])
DEPTH_THRESHOLD = CFG["projection"].get("depth_consistency_threshold", 0.10)


def load_scan_data(scan_dir):
    """Load all required data for projection."""
    # LiDAR cloud
    cloud_path = scan_dir / "lidar_cloud.ply"
    pcd = o3d.io.read_point_cloud(str(cloud_path))
    points = np.asarray(pcd.points)
    print(f"LiDAR cloud: {len(points):,} points")

    # LiDAR trajectory
    lidar_traj = np.load(scan_dir / "lidar_trajectory.npz")
    lidar_poses = lidar_traj["poses"]
    lidar_ts = lidar_traj["timestamps_ns"]
    print(f"LiDAR trajectory: {len(lidar_poses)} poses")

    # D455 — load intrinsics and frame timestamps directly from captured files
    d455_dir = scan_dir / "d455"
    with open(d455_dir / "intrinsics.json") as f:
        intr = json.load(f)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    dist_coeffs = np.array(intr["coeffs"]) if "coeffs" in intr else None

    frame_numbers_all = sorted(
        int(p.stem.split("_")[1]) for p in d455_dir.glob("frame_*_rgb.png")
    )
    d455_ts = []
    valid_frames = []
    for num in frame_numbers_all:
        imu_path = d455_dir / f"frame_{num:03d}_imu.json"
        if imu_path.exists():
            with open(imu_path) as f:
                imu = json.load(f)
            ts = imu.get("capture_timestamp_ns", int(imu["timestamp_ms"] * 1e6))
            d455_ts.append(ts)
            valid_frames.append(num)
    d455_ts = np.array(d455_ts, dtype=np.int64)
    frame_numbers = np.array(valid_frames)
    print(f"D455: {len(frame_numbers)} frames, intrinsics: fx={fx:.1f} fy={fy:.1f}")

    # Extrinsic — load from permanent calibrated path (not per-scan)
    extrinsic_path = Path(CFG["paths"]["permanent_extrinsic"]).expanduser()
    with open(extrinsic_path) as f:
        ext = json.load(f)
    T_lidar_from_d455 = np.array(ext["transform"])
    print(f"Extrinsic loaded: translation={np.linalg.norm(T_lidar_from_d455[:3, 3])*100:.2f}cm")

    return {
        "points": points,
        "pcd": pcd,
        "lidar_poses": lidar_poses,
        "lidar_ts": lidar_ts,
        "d455_ts": d455_ts,
        "frame_numbers": frame_numbers,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "T_lidar_from_d455": T_lidar_from_d455,
        "dist_coeffs": dist_coeffs,
        "camera_matrix": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
    }


def compute_camera_pose_in_lidar_frame(frame_idx, data):
    """
    Compute the camera-from-world transform for a given D455 frame.

    T_lidar_from_d455 transforms a point FROM D455 camera frame TO LiDAR local frame.
    T_world_lidar transforms a point FROM LiDAR local frame TO LiDAR world frame.

    So the full chain for a D455 camera point to world:
      P_world = T_world_lidar @ T_lidar_from_d455 @ P_d455

    Therefore:
      T_world_camera = T_world_lidar @ T_lidar_from_d455
      T_cam_from_world = inv(T_world_camera)
    """
    frame_ts = data["d455_ts"][frame_idx]

    # Get LiDAR pose at this timestamp
    T_world_lidar = interpolate_pose_at_timestamp(
        frame_ts, data["lidar_ts"], data["lidar_poses"]
    )

    # Camera pose in world: chain LiDAR pose with bracket extrinsic
    T_world_camera = T_world_lidar @ data["T_lidar_from_d455"]

    # Camera-from-world for projection
    T_cam_from_world = np.linalg.inv(T_world_camera)
    return T_cam_from_world


def load_d455_image(scan_dir, frame_num):
    """Load an RGB image for a D455 frame."""
    rgb_path = scan_dir / "d455" / f"frame_{frame_num:03d}_rgb.png"
    if not rgb_path.exists():
        return None
    img = cv2.imread(str(rgb_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def project_single_frame(scan_dir, data, frame_idx):
    """Project a single frame and return colored indices + colors."""
    frame_num = data["frame_numbers"][frame_idx]
    rgb = load_d455_image(scan_dir, frame_num)
    if rgb is None:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.uint8)

    if data["dist_coeffs"] is not None:
        rgb = cv2.undistort(rgb, data["camera_matrix"], data["dist_coeffs"])

    T_cam = compute_camera_pose_in_lidar_frame(frame_idx, data)

    indices, colors, depths = color_points_from_frame(
        data["points"], T_cam,
        data["fx"], data["fy"], data["cx"], data["cy"],
        rgb, ZBUF_TOL,
    )

    # Filter by max distance
    if len(indices) > 0:
        mask = depths < MAX_DIST
        indices = indices[mask]
        colors = colors[mask]

    return indices, colors


def visualize_single_frame(scan_dir, data, frame_idx):
    """Visualize a single frame's projection with camera frustum."""
    indices, colors = project_single_frame(scan_dir, data, frame_idx)

    # Create colored version of the cloud
    n = len(data["points"])
    all_colors = np.full((n, 3), 0.5)  # gray
    if len(indices) > 0:
        all_colors[indices] = colors.astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"])
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Camera frustum wireframe
    T_cam = compute_camera_pose_in_lidar_frame(frame_idx, data)
    T_world_cam = np.linalg.inv(T_cam)
    cam_pos = T_world_cam[:3, 3]

    # Frustum corners in camera frame (approximate)
    fov_scale = 0.5
    hw = data["cx"] / data["fx"] * fov_scale
    hh = data["cy"] / data["fy"] * fov_scale
    corners_cam = np.array([
        [0, 0, 0],
        [-hw, -hh, fov_scale],
        [hw, -hh, fov_scale],
        [hw, hh, fov_scale],
        [-hw, hh, fov_scale],
    ])
    # Transform to world
    corners_world = (T_world_cam[:3, :3] @ corners_cam.T).T + cam_pos
    corners_world[0] = cam_pos  # origin is camera position

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(corners_world)
    frustum.lines = o3d.utility.Vector2iVector([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ])
    frustum.paint_uniform_color([1, 1, 0])

    frame_num = data["frame_numbers"][frame_idx]
    print(f"Frame {frame_num}: {len(indices)} points colored ({len(indices)/n*100:.1f}%)")
    o3d.visualization.draw_geometries(
        [pcd, frustum],
        window_name=f"Frame {frame_num} Projection",
        width=1280, height=720,
    )


def debug_overlay_frame(scan_dir, data, frame_idx, dot_radius=3):
    """
    Project LiDAR points into the D455 image and save a 2D alignment overlay.

    The overlay shows the D455 RGB image with LiDAR points drawn on top,
    colored by depth (blue=close, red=far). Use this to check if LiDAR
    edges (doorframes, wall corners) land on the matching image edges.
    If they're offset, the extrinsic rotation needs adjustment.

    Also reports an edge alignment score (median px distance from each LiDAR
    point to the nearest Canny edge). Lower = better extrinsic calibration.
    """
    frame_num = data["frame_numbers"][frame_idx]
    rgb = load_d455_image(scan_dir, frame_num)
    if rgb is None:
        print(f"No image for frame {frame_num}")
        return

    if data["dist_coeffs"] is not None:
        rgb = cv2.undistort(rgb, data["camera_matrix"], data["dist_coeffs"])

    T_cam = compute_camera_pose_in_lidar_frame(frame_idx, data)
    h, w = rgb.shape[:2]

    pixel_coords, point_indices, depths = project_points_to_image(
        data["points"], T_cam,
        data["fx"], data["fy"], data["cx"], data["cy"],
        h, w,
    )

    # Filter by max distance
    dist_mask = depths < MAX_DIST
    pixel_coords = pixel_coords[dist_mask]
    depths       = depths[dist_mask]

    print(f"Frame {frame_num}: {len(pixel_coords):,} LiDAR points projected")

    # Sort far → near so that near points (blue) draw on top
    order = np.argsort(depths)[::-1]
    pixel_coords = pixel_coords[order]
    depths       = depths[order]

    # Depth → jet colormap (BGR: blue=close, red=far)
    d_min = depths.min()
    d_max = np.percentile(depths, 98)
    d_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    jet_colors = cv2.applyColorMap(d_uint8.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # --- Draw LiDAR dots: vectorized write + dilation for configurable radius ---
    lidar_canvas = np.zeros_like(bgr)
    lidar_canvas[pixel_coords[:, 1], pixel_coords[:, 0]] = jet_colors  # near overwrites far
    if dot_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dot_radius + 1, 2 * dot_radius + 1)
        )
        lidar_canvas = cv2.dilate(lidar_canvas, kernel)
    lidar_mask = lidar_canvas.any(axis=2)

    # --- Panel 1: RGB + semi-transparent LiDAR dots ---
    alpha = 0.70
    panel1 = bgr.copy()
    panel1[lidar_mask] = (
        (1 - alpha) * bgr[lidar_mask].astype(np.float32) +
        alpha * lidar_canvas[lidar_mask].astype(np.float32)
    ).astype(np.uint8)

    # --- Canny edges ---
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    # --- Edge alignment score via distance transform ---
    # dist_to_edge[y, x] = pixel distance from (x,y) to nearest edge
    dist_to_edge = cv2.distanceTransform(
        (edges == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    sampled_dists = dist_to_edge[pixel_coords[:, 1], pixel_coords[:, 0]]
    median_px = float(np.median(sampled_dists))
    mean_px   = float(np.mean(sampled_dists))
    print(f"  Edge alignment: median={median_px:.1f}px  mean={mean_px:.1f}px  (lower=better)")

    # --- Panel 2: white edges on black + LiDAR dots ---
    panel2 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    panel2[lidar_mask] = lidar_canvas[lidar_mask]

    # --- Save ---
    combined = np.vstack([panel1, panel2])
    out_path = scan_dir / f"debug_overlay_frame{frame_num:03d}.png"
    cv2.imwrite(str(out_path), combined)
    print(f"Saved: {out_path}")
    print()
    print("How to read the overlay:")
    print("  Top panel : RGB + LiDAR depth dots (alpha-blended)")
    print("  Bot panel : RGB edges + LiDAR depth dots")
    print("  Color     : blue=close  red=far")
    print("  Alignment : LiDAR dots should sit ON the matching image edge")
    print("  If dots are offset from edges → adjust extrinsic rotation")


def load_d455_depth(scan_dir, frame_num):
    """Load a depth image for a D455 frame."""
    depth_path = scan_dir / "d455" / f"frame_{frame_num:03d}_depth.png"
    if not depth_path.exists():
        return None
    return cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)


def project_all_frames(scan_dir, data):
    """Project all frames onto the point cloud."""
    n_frames = len(data["frame_numbers"])
    print(f"\nProjecting {n_frames} frames onto {len(data['points']):,} points...")

    frames = []
    for i in range(n_frames):
        frame_num = data["frame_numbers"][i]
        rgb = load_d455_image(scan_dir, frame_num)
        if rgb is None:
            continue

        if data["dist_coeffs"] is not None:
            rgb = cv2.undistort(rgb, data["camera_matrix"], data["dist_coeffs"])

        T_cam = compute_camera_pose_in_lidar_frame(i, data)
        depth = load_d455_depth(scan_dir, frame_num)

        frames.append({
            "T_cam_from_world": T_cam,
            "rgb_image": rgb,
            "depth_image": depth,
            "fx": data["fx"], "fy": data["fy"],
            "cx": data["cx"], "cy": data["cy"],
        })

    print(f"  {len(frames)} frames with images loaded")

    colors, stats = color_points_multi_frame(
        data["points"], frames,
        zbuffer_tolerance=ZBUF_TOL,
        max_distance=MAX_DIST,
        default_color=DEFAULT_COLOR,
        depth_threshold=DEPTH_THRESHOLD,
    )

    print(f"\nProjection complete:")
    print(f"  Colored: {stats['colored_points']:,} / {stats['total_points']:,} "
          f"({stats['coverage_pct']:.1f}%)")
    print(f"  Avg frames/colored point: {stats['avg_frames_per_colored_point']:.1f}")

    return colors, stats


def main():
    parser = argparse.ArgumentParser(description="Project D455 color onto LiDAR cloud")
    parser.add_argument("scan_dir", type=str, help="Path to scan directory")
    parser.add_argument("--frame", type=int, default=None,
                        help="Single frame index to project (for debugging)")
    parser.add_argument("--visualize", action="store_true",
                        help="Open 3D visualization (single-frame mode)")
    parser.add_argument("--all", action="store_true",
                        help="Project all frames (default if --frame not given)")
    parser.add_argument("--overlay", action="store_true",
                        help="Save 2D alignment overlay image (use with --frame)")
    parser.add_argument("--overlay-stride", type=int, default=None, metavar="N",
                        help="Save alignment overlay for every Nth frame across the whole scan")
    parser.add_argument("--dot-radius", type=int, default=3, metavar="R",
                        help="Radius of LiDAR dots in overlay images (default: 3)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    output_path = scan_dir / "colored_cloud.ply"

    print("Loading scan data...")
    data = load_scan_data(scan_dir)

    if args.overlay_stride is not None:
        # Sweep mode: generate overlays for every Nth frame
        n_frames = len(data["frame_numbers"])
        frame_indices = range(0, n_frames, args.overlay_stride)
        print(f"Generating overlays for {len(list(frame_indices))} / {n_frames} frames "
              f"(stride={args.overlay_stride})...")
        for i in frame_indices:
            debug_overlay_frame(scan_dir, data, i, dot_radius=args.dot_radius)
        return 0

    if args.frame is not None:
        # Single-frame mode
        if args.frame >= len(data["frame_numbers"]):
            print(f"ERROR: Frame index {args.frame} out of range (0..{len(data['frame_numbers'])-1})")
            return 1

        if args.overlay:
            debug_overlay_frame(scan_dir, data, args.frame, dot_radius=args.dot_radius)
        elif args.visualize:
            visualize_single_frame(scan_dir, data, args.frame)
        else:
            indices, colors = project_single_frame(scan_dir, data, args.frame)
            n = len(data["points"])
            print(f"Frame {data['frame_numbers'][args.frame]}: "
                  f"{len(indices)} points colored ({len(indices)/n*100:.1f}%)")
        return 0

    # Full projection
    colors, stats = project_all_frames(scan_dir, data)

    # Fill uncolored points with height-based rainbow colormap
    colored_mask = ~np.all(colors == np.array(DEFAULT_COLOR, dtype=np.uint8), axis=1)
    uncolored = ~colored_mask
    n_uncolored = np.sum(uncolored)
    if n_uncolored > 0:
        z = data["points"][uncolored, 2]
        z_min, z_max = np.percentile(z, [2, 98])
        if z_max - z_min < 0.01:
            z_max = z_min + 1.0
        t = np.clip((z - z_min) / (z_max - z_min), 0, 1)

        # Full rainbow: blue at bottom → cyan → green → yellow → red at top
        hsv = np.zeros((n_uncolored, 3), dtype=np.float32)
        hsv[:, 0] = (1.0 - t) * 240.0  # Hue: blue(240°) at bottom → red(0°) at top
        hsv[:, 1] = 0.6                # Saturation (muted so it's clearly not real color)
        hsv[:, 2] = 0.5                # Value (dimmed to distinguish from real RGB)
        rgb_float = cv2.cvtColor(hsv[np.newaxis, :, :], cv2.COLOR_HSV2RGB)[0]
        colors[uncolored] = (rgb_float * 255).astype(np.uint8)
        print(f"  Height-colored {n_uncolored:,} uncolored points")

    # Save colored PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["points"])
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    o3d.io.write_point_cloud(str(output_path), pcd)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")

    if args.visualize:
        print("Opening viewer...")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Colored LiDAR Cloud",
            width=1280, height=720,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
