#!/usr/bin/env python3
"""
Calibrate D455-to-LiDAR extrinsic for a scan directory.

Adapted from fusion_pipeline/align_trajectories.py to work with
this repo's data layout (no d455_trajectory.npz needed).

Usage:
  python3 calibrate_extrinsic.py <scan_dir>
  python3 calibrate_extrinsic.py folsom_test --visualize
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

from utils.timestamps import interpolate_pose_at_timestamp

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_DEPTH = CFG["d455"]["max_depth_m"]


def depth_to_pointcloud(depth_image, fx, fy, cx, cy, max_depth_m=3.0):
    """Vectorized depth unprojection -> Open3D PointCloud (in camera frame)."""
    h, w = depth_image.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float64).ravel()
    v = v.astype(np.float64).ravel()
    z = depth_image.ravel().astype(np.float64) / 1000.0

    valid = (z > 0.1) & (z < max_depth_m)
    u, v, z = u[valid], v[valid], z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def load_d455_timestamps(d455_dir):
    """Load frame numbers and timestamps from per-frame IMU JSONs."""
    frame_numbers_all = sorted(
        int(p.stem.split("_")[1]) for p in d455_dir.glob("frame_*_rgb.png")
    )
    timestamps = []
    valid_frames = []
    for num in frame_numbers_all:
        imu_path = d455_dir / f"frame_{num:03d}_imu.json"
        if imu_path.exists():
            with open(imu_path) as f:
                imu = json.load(f)
            ts = imu.get("capture_timestamp_ns", int(imu["timestamp_ms"] * 1e6))
            timestamps.append(ts)
            valid_frames.append(num)
    return np.array(valid_frames), np.array(timestamps, dtype=np.int64)


def prepare_lidar_local(lidar_world_pts, T_world_lidar, max_range):
    """Crop LiDAR cloud near rig and transform to local frame."""
    rig_pos = T_world_lidar[:3, 3]
    dists = np.linalg.norm(lidar_world_pts - rig_pos, axis=1)
    nearby = lidar_world_pts[dists < max_range + 0.5]

    if len(nearby) < 500:
        return None

    T_local_world = np.linalg.inv(T_world_lidar)
    ones = np.ones((len(nearby), 1))
    local_pts = (T_local_world @ np.hstack([nearby, ones]).T).T[:, :3]

    local_dists = np.linalg.norm(local_pts, axis=1)
    mask = (local_dists > 0.1) & (local_dists < max_range + 0.5)
    local_pts = local_pts[mask]

    if len(local_pts) < 500:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_pts)
    return pcd


def quick_icp(source, target, init_T, max_dist, max_iter=15):
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return result.fitness, result.transformation


MAX_BRACKET_OFFSET_M = 0.20  # bracket offset can't exceed 20cm


def translation_ok(T):
    """Reject ICP results with unrealistic bracket translation."""
    return np.linalg.norm(T[:3, 3]) < MAX_BRACKET_OFFSET_M


def coarse_rotation_search(d455_down, lidar_down, voxel_size):
    step = 60
    angles = np.arange(0, 360, step)
    max_dist = voxel_size * 4

    best_fitness = 0
    best_T = np.eye(4)

    for ax in angles:
        for ay in angles:
            for az in angles:
                R = Rotation.from_euler('xyz', [ax, ay, az], degrees=True).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                fitness, T_result = quick_icp(d455_down, lidar_down, T, max_dist, 15)
                if fitness > best_fitness and translation_ok(T_result):
                    best_fitness = fitness
                    best_T = T_result

    return best_T, best_fitness


def fine_rotation_search(d455_down, lidar_down, voxel_size, coarse_T):
    coarse_euler = Rotation.from_matrix(coarse_T[:3, :3]).as_euler('xyz', degrees=True)
    coarse_trans = coarse_T[:3, 3]
    max_dist = voxel_size * 3

    offsets = np.arange(-25, 30, 10)
    best_fitness = 0
    best_T = coarse_T.copy()

    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                euler = coarse_euler + np.array([dx, dy, dz])
                R = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = coarse_trans
                fitness, T_result = quick_icp(d455_down, lidar_down, T, max_dist, 30)
                if fitness > best_fitness and translation_ok(T_result):
                    best_fitness = fitness
                    best_T = T_result

    return best_T, best_fitness


def full_icp_refinement(d455_down, lidar_down, voxel_size, init_T):
    lidar_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    result = o3d.pipelines.registration.registration_icp(
        d455_down, lidar_down,
        voxel_size * 2,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=200,
        ),
    )
    return result.transformation, result.fitness


def align_single_frame(d455_pcd, lidar_local_pcd, voxel_size=0.05):
    search_voxel = max(voxel_size, 0.05)
    d455_search = d455_pcd.voxel_down_sample(search_voxel)
    lidar_search = lidar_local_pcd.voxel_down_sample(search_voxel)

    print(f"    Search clouds: D455={len(d455_search.points)}, "
          f"LiDAR={len(lidar_search.points)}")

    t0 = time.time()
    coarse_T, coarse_fitness = coarse_rotation_search(d455_search, lidar_search, search_voxel)
    euler = Rotation.from_matrix(coarse_T[:3, :3]).as_euler('xyz', degrees=True)
    print(f"    Coarse: fitness={coarse_fitness:.4f}, "
          f"euler=[{euler[0]:.0f}, {euler[1]:.0f}, {euler[2]:.0f}] "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    fine_T, fine_fitness = fine_rotation_search(d455_search, lidar_search, search_voxel, coarse_T)
    euler = Rotation.from_matrix(fine_T[:3, :3]).as_euler('xyz', degrees=True)
    print(f"    Fine:   fitness={fine_fitness:.4f}, "
          f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] "
          f"({time.time()-t0:.1f}s)")

    d455_down = d455_pcd.voxel_down_sample(voxel_size)
    lidar_down = lidar_local_pcd.voxel_down_sample(voxel_size)
    final_T, final_fitness = full_icp_refinement(d455_down, lidar_down, voxel_size, fine_T)
    euler = Rotation.from_matrix(final_T[:3, :3]).as_euler('xyz', degrees=True)
    trans = final_T[:3, 3]
    print(f"    Final:  fitness={final_fitness:.4f}, "
          f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}], "
          f"t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")

    return final_T, final_fitness


def validate_on_frame(d455_pcd, lidar_local_pcd, init_T, voxel_size=0.03):
    d455_down = d455_pcd.voxel_down_sample(voxel_size)
    lidar_down = lidar_local_pcd.voxel_down_sample(voxel_size)
    lidar_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    result = o3d.pipelines.registration.registration_icp(
        d455_down, lidar_down,
        voxel_size * 2,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100,
        ),
    )
    return result.transformation, result.fitness


def main():
    parser = argparse.ArgumentParser(description="Calibrate D455-to-LiDAR extrinsic")
    parser.add_argument("scan_dir", type=str)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    d455_dir = scan_dir / "d455"
    output_path = scan_dir / "extrinsic.json"

    # Load LiDAR
    print("Loading LiDAR data...")
    lidar_cloud = o3d.io.read_point_cloud(str(scan_dir / "lidar_cloud.ply"))
    lidar_pts = np.asarray(lidar_cloud.points)
    lidar_traj = np.load(scan_dir / "lidar_trajectory.npz")
    lidar_poses = lidar_traj["poses"]
    lidar_ts = lidar_traj["timestamps_ns"]
    print(f"  Cloud: {len(lidar_pts):,} points, Trajectory: {len(lidar_poses)} poses")

    # Load D455 intrinsics + timestamps
    print("Loading D455 data...")
    with open(d455_dir / "intrinsics.json") as f:
        intr = json.load(f)
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])

    frame_numbers, d455_ts = load_d455_timestamps(d455_dir)
    print(f"  {len(frame_numbers)} frames, intrinsics: fx={fx:.1f} fy={fy:.1f}")

    # Pick primary frame near middle
    primary_idx = len(frame_numbers) // 2
    primary_frame_num = int(frame_numbers[primary_idx])
    primary_ts = d455_ts[primary_idx]

    depth_path = d455_dir / f"frame_{primary_frame_num:03d}_depth.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or np.all(depth == 0):
        print(f"ERROR: Cannot load depth for frame {primary_frame_num}")
        return 1

    d455_pcd = depth_to_pointcloud(depth, fx, fy, cx, cy, MAX_DEPTH)
    print(f"\nPrimary frame: {primary_frame_num} ({len(d455_pcd.points):,} depth points)")

    T_world_lidar = interpolate_pose_at_timestamp(primary_ts, lidar_ts, lidar_poses)
    lidar_local = prepare_lidar_local(lidar_pts, T_world_lidar, MAX_DEPTH)
    if lidar_local is None:
        print("ERROR: Not enough LiDAR points near the rig position")
        return 1
    print(f"LiDAR local crop: {len(lidar_local.points):,} points")

    # Grid search + ICP
    print(f"\nRunning rotation grid search + ICP (voxel={args.voxel_size}m)...")
    t_start = time.time()
    T_best, fitness_best = align_single_frame(d455_pcd, lidar_local, args.voxel_size)
    print(f"Grid search completed in {time.time() - t_start:.1f}s")

    if fitness_best < 0.1:
        print("ERROR: Alignment failed (fitness too low)")
        return 1

    # Validate on additional frames
    n_total = len(frame_numbers)
    n_validate = min(8, n_total)
    val_indices = np.linspace(0, n_total - 1, n_validate + 2, dtype=int)[1:-1]

    print(f"\nValidating on {n_validate} additional frames...")
    val_results = []

    for idx in val_indices:
        if idx == primary_idx:
            continue

        frame_num = int(frame_numbers[idx])
        dp = d455_dir / f"frame_{frame_num:03d}_depth.png"
        if not dp.exists():
            continue

        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if d is None or np.all(d == 0):
            continue

        pcd = depth_to_pointcloud(d, fx, fy, cx, cy, MAX_DEPTH)
        if len(pcd.points) < 500:
            continue

        T_wl = interpolate_pose_at_timestamp(d455_ts[idx], lidar_ts, lidar_poses)
        ll = prepare_lidar_local(lidar_pts, T_wl, MAX_DEPTH)
        if ll is None:
            continue

        T_val, fit_val = validate_on_frame(pcd, ll, T_best, args.voxel_size)
        trans = T_val[:3, 3]
        val_results.append((T_val, fit_val, frame_num))
        print(f"  Frame {frame_num:3d}: fitness={fit_val:.4f}, "
              f"t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")

    # Aggregate results
    all_results = [(T_best, fitness_best, primary_frame_num)] + val_results

    translations = np.array([r[0][:3, 3] for r in all_results])
    median_t = np.median(translations, axis=0)
    t_dists = np.linalg.norm(translations - median_t, axis=1)
    consistent = t_dists < 0.10

    good_results = [r for r, c in zip(all_results, consistent) if c]
    if not good_results:
        good_results = all_results

    print(f"\n{len(good_results)} / {len(all_results)} results consistent")

    # Weighted average
    if len(good_results) >= 3:
        weights = np.array([r[1] for r in good_results])
        weights /= weights.sum()

        avg_t = sum(w * r[0][:3, 3] for w, (r, _) in
                     zip(weights, [(r, None) for r in good_results]))

        quats = [Rotation.from_matrix(r[0][:3, :3]).as_quat() for r in good_results]
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[0]) < 0:
                quats[i] = -quats[i]
        avg_q = sum(w * q for w, q in zip(weights, quats))
        avg_q /= np.linalg.norm(avg_q)
        avg_R = Rotation.from_quat(avg_q).as_matrix()

        T_final = np.eye(4)
        T_final[:3, :3] = avg_R
        T_final[:3, 3] = avg_t
    else:
        T_final = good_results[0][0]

    euler = Rotation.from_matrix(T_final[:3, :3]).as_euler("xyz", degrees=True)
    trans = T_final[:3, 3]

    print(f"\nT_lidar_from_d455:")
    print(f"  Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]m "
          f"(magnitude: {np.linalg.norm(trans) * 100:.2f}cm)")
    print(f"  Rotation (euler XYZ): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

    # Save
    extrinsic_data = {
        "description": "T_lidar_from_d455: transforms points from D455 camera frame to LiDAR local frame",
        "transform": T_final.tolist(),
        "translation_m": trans.tolist(),
        "rotation_matrix": T_final[:3, :3].tolist(),
        "method": "depth_icp_grid_search",
        "alignment_info": {
            "n_frames_used": len(good_results),
            "primary_frame": primary_frame_num,
            "best_fitness": float(max(r[1] for r in good_results)),
            "mean_fitness": float(np.mean([r[1] for r in good_results])),
            "voxel_size": args.voxel_size,
        },
    }

    with open(output_path, "w") as f:
        json.dump(extrinsic_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Also install as the permanent extrinsic
    perm_path = Path(CFG["paths"]["permanent_extrinsic"]).expanduser()
    perm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(perm_path, "w") as f:
        json.dump(extrinsic_data, f, indent=2)
    print(f"Installed as permanent extrinsic: {perm_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
