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
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from utils.projection import (color_points_from_frame, _center_weight,
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
EDGE_EROSION_PX = CFG["projection"].get("edge_erosion_px", 2)
EDGE_DEPTH_THRESHOLD_MM = CFG["projection"].get("edge_depth_threshold_mm", 300)


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

    # Extrinsic — prefer per-scan calibration, fall back to permanent
    per_scan_extrinsic = scan_dir / "extrinsic.json"
    if per_scan_extrinsic.exists():
        extrinsic_path = per_scan_extrinsic
        print(f"Using per-scan extrinsic: {extrinsic_path}")
    else:
        extrinsic_path = Path(CFG["paths"]["permanent_extrinsic"]).expanduser()
        print(f"Using permanent extrinsic: {extrinsic_path}")
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

    indices, colors, depths, _pixels = color_points_from_frame(
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


def refine_pose_icp(T_cam_from_world, depth_image, points_world,
                     fx, fy, cx, cy, camera_matrix, dist_coeffs,
                     max_depth=3.0, icp_voxel=0.02, icp_threshold=0.05):
    """
    Refine camera pose using ICP between D455 depth cloud and local LiDAR crop.

    1. Unproject D455 depth pixels into camera-frame 3D points
    2. Crop LiDAR points near the camera and transform to camera frame
    3. Run Point-to-Plane ICP to refine the relative alignment
    4. Return the corrected T_cam_from_world

    This typically reduces alignment error from ~5px to ~1-2px.
    """
    if depth_image is None:
        return T_cam_from_world

    h, w = depth_image.shape[:2]

    # 1. Unproject D455 depth to 3D (camera frame)
    depth_m = depth_image.astype(np.float32) / 1000.0
    valid = (depth_m > 0.1) & (depth_m < max_depth)

    # Subsample for speed — every 4th pixel
    ys, xs = np.where(valid)
    step = 4
    ys, xs = ys[::step], xs[::step]
    zs = depth_m[ys, xs]

    d455_pts_cam = np.zeros((len(xs), 3), dtype=np.float32)
    d455_pts_cam[:, 0] = (xs - cx) * zs / fx
    d455_pts_cam[:, 1] = (ys - cy) * zs / fy
    d455_pts_cam[:, 2] = zs

    if len(d455_pts_cam) < 500:
        return T_cam_from_world

    # 2. Crop LiDAR points near camera and transform to camera frame
    T_world_from_cam = np.linalg.inv(T_cam_from_world)
    cam_pos = T_world_from_cam[:3, 3]
    dists = np.linalg.norm(points_world - cam_pos, axis=1)
    near_mask = dists < max_depth * 1.5
    lidar_near_world = points_world[near_mask]

    if len(lidar_near_world) < 500:
        return T_cam_from_world

    # Transform LiDAR to camera frame using current estimate
    ones = np.ones((len(lidar_near_world), 1), dtype=lidar_near_world.dtype)
    lidar_hom = np.hstack([lidar_near_world, ones])
    lidar_pts_cam = (T_cam_from_world @ lidar_hom.T).T[:, :3]

    # Keep only points in front of camera
    front = lidar_pts_cam[:, 2] > 0
    lidar_pts_cam = lidar_pts_cam[front]

    if len(lidar_pts_cam) < 500:
        return T_cam_from_world

    # 3. Build Open3D point clouds and run ICP
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(d455_pts_cam.astype(np.float64))
    src = src.voxel_down_sample(icp_voxel)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(lidar_pts_cam.astype(np.float64))
    tgt = tgt.voxel_down_sample(icp_voxel)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        src, tgt, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
    )

    if result.fitness < 0.3:
        # ICP failed to converge — keep original pose
        return T_cam_from_world

    # 4. Apply correction: T_correction aligns D455 cloud to LiDAR cloud in camera frame
    # The refined camera-from-world = T_correction @ T_cam_from_world
    # But T_correction was computed in camera frame, so:
    # New T_cam = T_correction_inv @ old_T_cam  (we aligned src→tgt, so src needs correction)
    # Actually: ICP gives T that transforms src to align with tgt.
    # src = D455 in cam frame, tgt = LiDAR in cam frame (using old T_cam)
    # So corrected D455 = T_icp @ D455_cam should align with LiDAR_cam
    # This means the D455 points were slightly off — the camera was actually at a slightly
    # different pose. The corrected T_cam = T_cam @ inv(T_icp) ... but we need to think carefully.
    #
    # Simpler: the ICP correction is small. Apply it as:
    # T_cam_refined = inv(T_icp) @ T_cam_from_world
    # Because: LiDAR_cam = T_cam @ P_world (correct reference)
    #          D455_cam_wrong = points from depth (slightly off)
    #          T_icp @ D455_cam_wrong = LiDAR_cam
    #          So D455 in TRUE cam frame = T_icp @ D455_cam
    #          Which means true_cam_from_old_cam = T_icp
    #          T_true_cam_from_world = T_icp @ T_cam_from_world
    T_refined = result.transformation @ T_cam_from_world

    return T_refined


def _check_gyro_motion(scan_dir, frame_num, max_gyro_dps):
    """Check if a frame was captured during fast rotation. Returns True if OK to use."""
    if max_gyro_dps is None:
        return True
    imu_path = scan_dir / "d455" / f"frame_{frame_num:03d}_imu.json"
    if not imu_path.exists():
        return True
    with open(imu_path) as f:
        imu = json.load(f)
    gyro = imu.get("gyroscope")
    if gyro is None:
        return True
    # Angular velocity magnitude in deg/s
    omega = np.degrees(np.sqrt(gyro["x"]**2 + gyro["y"]**2 + gyro["z"]**2))
    return omega <= max_gyro_dps


def _load_frame(args):
    """Load and undistort a single frame (worker for parallel loading)."""
    i, scan_dir, frame_num, data = args
    # Skip frames captured during fast rotation (blurry + unreliable timestamps)
    if not _check_gyro_motion(scan_dir, frame_num, data.get("max_gyro_dps")):
        return None
    rgb = load_d455_image(scan_dir, frame_num)
    if rgb is None:
        return None
    if data["dist_coeffs"] is not None:
        rgb = cv2.undistort(rgb, data["camera_matrix"], data["dist_coeffs"])
    T_cam = compute_camera_pose_in_lidar_frame(i, data)
    depth = load_d455_depth(scan_dir, frame_num)
    return {
        "T_cam_from_world": T_cam,
        "rgb_image": rgb,
        "depth_image": depth,
        "fx": data["fx"], "fy": data["fy"],
        "cx": data["cx"], "cy": data["cy"],
    }


def project_all_frames(scan_dir, data, blend_mode="mean", refine_icp=False):
    """Project all frames onto the point cloud, processing in batches to limit memory."""
    n_frames = len(data["frame_numbers"])
    n_points = len(data["points"])
    print(f"\nProjecting {n_frames} frames onto {n_points:,} points...")
    print(f"  Blend mode: {blend_mode}")
    if data.get("max_gyro_dps") is not None:
        print(f"  Motion filter: skip frames > {data['max_gyro_dps']:.0f} deg/s")
    if refine_icp:
        print(f"  ICP pose refinement: ENABLED")

    # Batch size: limit concurrent frames in memory to control RAM usage.
    # Each frame is ~5MB (RGB+depth) and each projection thread needs ~800MB
    # of temporaries for large clouds, so keep batches small.
    BATCH_SIZE = 8
    N_PROJ_WORKERS = min(4, os.cpu_count() or 2)

    # Accumulators for center_weighted and mean blending
    if blend_mode == "center_weighted":
        color_sum = np.zeros((n_points, 3), dtype=np.float64)
        weight_sum = np.zeros(n_points, dtype=np.float64)
    elif blend_mode == "mean":
        color_sum = np.zeros((n_points, 3), dtype=np.float64)
        color_count = np.zeros(n_points, dtype=np.int32)
    elif blend_mode == "closest":
        best_colors = np.full((n_points, 3), DEFAULT_COLOR, dtype=np.uint8)
        best_depth = np.full(n_points, np.inf, dtype=np.float64)
    elif blend_mode == "first":
        best_colors = np.full((n_points, 3), DEFAULT_COLOR, dtype=np.uint8)
        colored = np.zeros(n_points, dtype=bool)

    total_colored = 0

    for batch_start in range(0, n_frames, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_frames)
        batch_indices = range(batch_start, batch_end)

        # Load this batch of frames
        load_args = [(i, scan_dir, data["frame_numbers"][i], data) for i in batch_indices]
        with ThreadPoolExecutor(max_workers=N_PROJ_WORKERS) as executor:
            loaded = list(executor.map(_load_frame, load_args))

        # Project each frame in this batch
        for j, frame in enumerate(loaded):
            if frame is None:
                continue
            frame_idx = batch_start + j

            # Per-frame ICP pose refinement
            if refine_icp and frame.get("depth_image") is not None:
                frame["T_cam_from_world"] = refine_pose_icp(
                    frame["T_cam_from_world"],
                    frame["depth_image"],
                    data["points"],
                    frame["fx"], frame["fy"], frame["cx"], frame["cy"],
                    data.get("camera_matrix"), data.get("dist_coeffs"),
                )

            indices, colors, depths, pixels = color_points_from_frame(
                data["points"],
                frame["T_cam_from_world"],
                frame["fx"], frame["fy"], frame["cx"], frame["cy"],
                frame["rgb_image"],
                ZBUF_TOL,
                depth_image=frame.get("depth_image"),
                depth_threshold=DEPTH_THRESHOLD,
                erosion_px=EDGE_EROSION_PX,
                edge_threshold_mm=EDGE_DEPTH_THRESHOLD_MM,
            )

            if len(indices) == 0:
                continue

            # Filter by max distance
            dist_mask = depths < MAX_DIST
            indices = indices[dist_mask]
            colors = colors[dist_mask]
            depths = depths[dist_mask]
            pixels = pixels[dist_mask]

            # Accumulate per blend mode
            if blend_mode == "center_weighted":
                h, w = frame["rgb_image"].shape[:2]
                weights = _center_weight(pixels, frame["cx"], frame["cy"], w / 2, h / 2)
                np.add.at(color_sum, (indices, slice(None)),
                          colors.astype(np.float64) * weights[:, np.newaxis])
                np.add.at(weight_sum, indices, weights)
            elif blend_mode == "mean":
                np.add.at(color_sum, (indices, slice(None)), colors.astype(np.float64))
                np.add.at(color_count, indices, 1)
            elif blend_mode == "closest":
                closer_mask = depths < best_depth[indices]
                update_idx = indices[closer_mask]
                best_colors[update_idx] = colors[closer_mask]
                best_depth[update_idx] = depths[closer_mask]
            elif blend_mode == "first":
                new_mask = ~colored[indices]
                new_idx = indices[new_mask]
                best_colors[new_idx] = colors[new_mask]
                colored[new_idx] = True

        # Free batch memory
        del loaded

        # Progress report
        frames_done = batch_end
        if blend_mode == "center_weighted":
            pct = np.sum(weight_sum > 0) / n_points * 100
        elif blend_mode == "mean":
            pct = np.sum(color_count > 0) / n_points * 100
        elif blend_mode == "closest":
            pct = np.sum(best_depth < np.inf) / n_points * 100
        elif blend_mode == "first":
            pct = np.sum(colored) / n_points * 100
        print(f"  Frame {frames_done}/{n_frames}: {pct:.1f}% colored")

    # Finalize colors
    if blend_mode == "center_weighted":
        colored_mask = weight_sum > 0
        final_colors = np.full((n_points, 3), DEFAULT_COLOR, dtype=np.uint8)
        if np.any(colored_mask):
            avg = color_sum[colored_mask] / weight_sum[colored_mask, np.newaxis]
            final_colors[colored_mask] = np.clip(avg, 0, 255).astype(np.uint8)
    elif blend_mode == "mean":
        colored_mask = color_count > 0
        final_colors = np.full((n_points, 3), DEFAULT_COLOR, dtype=np.uint8)
        if np.any(colored_mask):
            mean_colors = color_sum[colored_mask] / color_count[colored_mask, np.newaxis]
            final_colors[colored_mask] = np.clip(mean_colors, 0, 255).astype(np.uint8)
    elif blend_mode == "closest":
        final_colors = best_colors
        colored_mask = best_depth < np.inf
    elif blend_mode == "first":
        final_colors = best_colors
        colored_mask = colored

    n_colored = int(np.sum(colored_mask))
    stats = {
        "total_points": n_points,
        "colored_points": n_colored,
        "coverage_pct": float(n_colored / n_points * 100),
        "avg_frames_per_colored_point": float(
            np.mean(color_count[colored_mask]) if blend_mode == "mean" and np.any(colored_mask)
            else np.mean(weight_sum[colored_mask]) if blend_mode == "center_weighted" and np.any(colored_mask)
            else 1.0
        ),
    }

    print(f"\nProjection complete:")
    print(f"  Colored: {stats['colored_points']:,} / {stats['total_points']:,} "
          f"({stats['coverage_pct']:.1f}%)")
    print(f"  Avg frames/colored point: {stats['avg_frames_per_colored_point']:.1f}")

    return final_colors, stats


def debug_frame_panels(scan_dir, data, frame_idx):
    """
    Generate a 4-panel debug image for a single frame:
      Top-left:     Raw RGB frame
      Top-right:    RGB masked to only pixels that hit LiDAR points
      Bottom-left:  Camera-view render of colored LiDAR points (color from image)
      Bottom-right: Camera-view render of LiDAR geometry only (white dots, same subset)
    Saved as debug_panels_frameNNN.png
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

    # Project all LiDAR points into this camera
    pixel_coords, point_indices, depths = project_points_to_image(
        data["points"], T_cam,
        data["fx"], data["fy"], data["cx"], data["cy"],
        h, w,
    )

    # Filter by max distance
    dist_mask = depths < MAX_DIST
    pixel_coords = pixel_coords[dist_mask]
    point_indices = point_indices[dist_mask]
    depths = depths[dist_mask]

    print(f"Frame {frame_num}: {len(pixel_coords):,} LiDAR points projected into image")

    # --- Panel 1: Raw RGB frame ---
    panel1 = rgb.copy()

    # --- Panel 2: RGB but only pixels that have LiDAR points, rest dimmed ---
    panel2 = (rgb.astype(np.float32) * 0.15).astype(np.uint8)  # dim everything
    # Create a mask of pixels that have LiDAR hits
    hit_mask = np.zeros((h, w), dtype=bool)
    hit_mask[pixel_coords[:, 1], pixel_coords[:, 0]] = True
    # Dilate slightly so individual pixels are visible
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hit_mask_dilated = cv2.dilate(hit_mask.astype(np.uint8), kernel).astype(bool)
    panel2[hit_mask_dilated] = rgb[hit_mask_dilated]

    # --- Panel 3: Black canvas, draw colored LiDAR points using image colors ---
    panel3 = np.zeros((h, w, 3), dtype=np.uint8)
    # Sample colors from RGB image at projected pixel locations
    colors_sampled = rgb[pixel_coords[:, 1], pixel_coords[:, 0]]
    # Sort far-to-near so near points draw on top
    order = np.argsort(depths)[::-1]
    px_sorted = pixel_coords[order]
    col_sorted = colors_sampled[order]
    panel3[px_sorted[:, 1], px_sorted[:, 0]] = col_sorted
    # Dilate to make points visible
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    panel3 = cv2.dilate(panel3, kernel3)

    # --- Panel 4: Black canvas, draw LiDAR points as white (geometry only) ---
    panel4 = np.zeros((h, w, 3), dtype=np.uint8)
    # Depth-shade: closer = brighter white
    d_min = depths.min()
    d_max = np.percentile(depths, 98)
    d_norm = np.clip((depths - d_min) / (d_max - d_min + 1e-6), 0, 1)
    brightness = ((1.0 - d_norm) * 200 + 55).astype(np.uint8)  # 55-255
    white_colors = np.stack([brightness, brightness, brightness], axis=1)
    px_sorted4 = pixel_coords[order]
    wc_sorted = white_colors[order]
    panel4[px_sorted4[:, 1], px_sorted4[:, 0]] = wc_sorted
    panel4 = cv2.dilate(panel4, kernel3)

    # --- Labels ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    for panel, label in [(panel1, "1: Raw RGB"),
                         (panel2, "2: RGB pixels hitting LiDAR"),
                         (panel3, "3: LiDAR colored by image"),
                         (panel4, "4: LiDAR geometry (depth-shaded)")]:
        cv2.putText(panel, label, (10, 30), font, 0.8, (0, 255, 0), 2)

    # --- Combine 2x2 ---
    top = np.hstack([panel1, panel2])
    bot = np.hstack([panel3, panel4])
    combined = np.vstack([top, bot])

    # Convert RGB to BGR for saving
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    out_path = scan_dir / f"debug_panels_frame{frame_num:03d}.png"
    cv2.imwrite(str(out_path), combined_bgr)
    print(f"Saved: {out_path} ({combined.shape[1]}x{combined.shape[0]})")


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
    parser.add_argument("--blend-mode", choices=["mean", "closest", "first", "center_weighted"],
                        default="center_weighted",
                        help="Color blending: 'center_weighted' weights by image center distance (default), "
                             "'mean' averages all frames uniformly, "
                             "'closest' keeps color from nearest camera, "
                             "'first' keeps first valid color")
    parser.add_argument("--refine-icp", action="store_true",
                        help="Refine each frame's pose via ICP before projection (sharper colors)")
    parser.add_argument("--max-gyro", type=float, default=60.0, metavar="DPS",
                        help="Skip frames where gyro angular velocity exceeds this (deg/s). "
                             "Filters motion-blurred frames with unreliable timestamps. Default: 60")
    parser.add_argument("--debug-frame", type=int, default=None, metavar="F",
                        help="Generate 4-panel debug image for frame F")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    output_path = scan_dir / "colored_cloud.ply"

    print("Loading scan data...")
    data = load_scan_data(scan_dir)
    data["max_gyro_dps"] = args.max_gyro

    if args.debug_frame is not None:
        if args.debug_frame >= len(data["frame_numbers"]):
            print(f"ERROR: Frame index {args.debug_frame} out of range "
                  f"(0..{len(data['frame_numbers'])-1})")
            return 1
        debug_frame_panels(scan_dir, data, args.debug_frame)
        return 0

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
    colors, stats = project_all_frames(scan_dir, data, blend_mode=args.blend_mode,
                                       refine_icp=args.refine_icp)

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
