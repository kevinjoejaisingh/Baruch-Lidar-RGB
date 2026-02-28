#!/usr/bin/env python3
"""
Find the D455-to-LiDAR bracket extrinsic via rotation grid search + ICP.

Both sensors are rigidly bracket-mounted, so the extrinsic T_lidar_from_d455
is a fixed rotation + small translation. The main unknown is the rotation
between D455 camera frame and LiDAR sensor frame.

Algorithm:
  1. Load LiDAR world cloud + trajectory, pick one D455 frame
  2. Crop LiDAR cloud near the rig, transform to LiDAR local frame
  3. Coarse rotation grid search (60-degree steps) with quick ICP
  4. Fine rotation search around the best (10-degree steps)
  5. Full ICP refinement for final T_lidar_from_d455
  6. Validate on additional frames

Usage:
  python align_trajectories.py <scan_dir>
  python align_trajectories.py scans/test_real --visualize
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

from utils.projection import depth_to_pointcloud
from utils.timestamps import interpolate_pose_at_timestamp

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

MAX_DEPTH = CFG["d455"]["max_depth_m"]


# ---------------------------------------------------------------------------
# Prepare LiDAR local cloud for a given timestamp
# ---------------------------------------------------------------------------

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

    # Restrict to similar depth range as D455 (both measured from origin)
    local_dists = np.linalg.norm(local_pts, axis=1)
    mask = (local_dists > 0.1) & (local_dists < max_range + 0.5)
    local_pts = local_pts[mask]

    if len(local_pts) < 500:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_pts)
    return pcd


# ---------------------------------------------------------------------------
# Rotation grid search + ICP
# ---------------------------------------------------------------------------

def quick_icp(source, target, init_T, max_dist, max_iter=15):
    """Run fast Point-to-Point ICP, return (fitness, transformation)."""
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
    )
    return result.fitness, result.transformation


def coarse_rotation_search(d455_down, lidar_down, voxel_size):
    """
    Stage 1: Search over Euler angle grid at 60-degree steps.
    216 candidates, ~15 ICP iterations each.
    """
    step = 60
    angles = np.arange(0, 360, step)
    max_dist = voxel_size * 4

    best_fitness = 0
    best_T = np.eye(4)
    count = 0
    total = len(angles) ** 3

    for ax in angles:
        for ay in angles:
            for az in angles:
                R = Rotation.from_euler('xyz', [ax, ay, az], degrees=True).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R

                fitness, T_result = quick_icp(d455_down, lidar_down, T, max_dist, 15)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_T = T_result

                count += 1

    return best_T, best_fitness


def fine_rotation_search(d455_down, lidar_down, voxel_size, coarse_T):
    """
    Stage 2: Search ±25 degrees around the coarse best in 10-degree steps.
    ~216 candidates, 30 ICP iterations each.
    """
    coarse_euler = Rotation.from_matrix(coarse_T[:3, :3]).as_euler('xyz', degrees=True)
    coarse_trans = coarse_T[:3, 3]
    max_dist = voxel_size * 3

    offsets = np.arange(-25, 30, 10)  # [-25, -15, -5, 5, 15, 25]
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
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_T = T_result

    return best_T, best_fitness


def full_icp_refinement(d455_down, lidar_down, voxel_size, init_T):
    """Stage 3: Full Point-to-Plane ICP refinement."""
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


# ---------------------------------------------------------------------------
# Main alignment for one frame
# ---------------------------------------------------------------------------

def align_single_frame(d455_pcd, lidar_local_pcd, voxel_size=0.05, verbose=True):
    """
    Find T_lidar_from_d455 by grid-search rotation + ICP.
    Both clouds must be in the right frames (D455 in camera, LiDAR in local).
    """
    # Aggressive downsampling for search phase
    search_voxel = max(voxel_size, 0.05)
    d455_search = d455_pcd.voxel_down_sample(search_voxel)
    lidar_search = lidar_local_pcd.voxel_down_sample(search_voxel)

    if verbose:
        print(f"    Search clouds: D455={len(d455_search.points)}, "
              f"LiDAR={len(lidar_search.points)}")

    # Stage 1: Coarse rotation search
    t0 = time.time()
    coarse_T, coarse_fitness = coarse_rotation_search(
        d455_search, lidar_search, search_voxel
    )
    if verbose:
        euler = Rotation.from_matrix(coarse_T[:3, :3]).as_euler('xyz', degrees=True)
        print(f"    Coarse: fitness={coarse_fitness:.4f}, "
              f"euler=[{euler[0]:.0f}, {euler[1]:.0f}, {euler[2]:.0f}] "
              f"({time.time()-t0:.1f}s)")

    # Stage 2: Fine rotation search
    t0 = time.time()
    fine_T, fine_fitness = fine_rotation_search(
        d455_search, lidar_search, search_voxel, coarse_T
    )
    if verbose:
        euler = Rotation.from_matrix(fine_T[:3, :3]).as_euler('xyz', degrees=True)
        print(f"    Fine:   fitness={fine_fitness:.4f}, "
              f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] "
              f"({time.time()-t0:.1f}s)")

    # Stage 3: Full ICP refinement at original resolution
    d455_down = d455_pcd.voxel_down_sample(voxel_size)
    lidar_down = lidar_local_pcd.voxel_down_sample(voxel_size)

    final_T, final_fitness = full_icp_refinement(
        d455_down, lidar_down, voxel_size, fine_T
    )
    if verbose:
        euler = Rotation.from_matrix(final_T[:3, :3]).as_euler('xyz', degrees=True)
        trans = final_T[:3, 3]
        print(f"    Final:  fitness={final_fitness:.4f}, "
              f"euler=[{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}], "
              f"t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]")

    return final_T, final_fitness


# ---------------------------------------------------------------------------
# Validate on additional frames (using the found rotation as initial guess)
# ---------------------------------------------------------------------------

def validate_on_frame(d455_pcd, lidar_local_pcd, init_T, voxel_size=0.03):
    """Quick ICP validation with a known initial guess."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find D455-to-LiDAR extrinsic via depth ICP")
    parser.add_argument("scan_dir", type=str, help="Path to scan directory")
    parser.add_argument("--visualize", action="store_true", help="Show 3D visualization")
    parser.add_argument("--voxel-size", type=float, default=0.03,
                        help="Voxel size for final ICP (default: 0.03m)")
    parser.add_argument("--hardcode", action="store_true",
                        help="Use hardcoded bracket measurements")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    output_path = scan_dir / "extrinsic.json"
    d455_dir = scan_dir / "d455"

    if args.hardcode:
        # Hardcoded: D455→LiDAR based on sensor convention analysis
        # D455 (X-right, Y-down, Z-fwd) → LiDAR (X-down, Y-left, Z-fwd)
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [0.015, 0, 0]  # D455 ~1.5cm below in LiDAR X-down frame
        euler = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
        print(f"Using hardcoded extrinsic:")
        print(f"  Rotation (euler XYZ): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}] deg")
        print(f"  Translation: {T[:3, 3].tolist()}")

        extrinsic_data = {
            "description": "T_lidar_from_d455: transforms points from D455 camera frame to LiDAR local frame",
            "transform": T.tolist(),
            "translation_m": T[:3, 3].tolist(),
            "rotation_matrix": T[:3, :3].tolist(),
            "method": "hardcoded_bracket",
        }
        with open(output_path, "w") as f:
            json.dump(extrinsic_data, f, indent=2)
        print(f"Saved: {output_path}")
        return 0

    # ---- Load data ----
    print("Loading LiDAR data...")
    lidar_cloud = o3d.io.read_point_cloud(str(scan_dir / "lidar_cloud.ply"))
    lidar_pts = np.asarray(lidar_cloud.points)
    lidar_traj = np.load(scan_dir / "lidar_trajectory.npz")
    lidar_poses = lidar_traj["poses"]
    lidar_ts = lidar_traj["timestamps_ns"]
    print(f"  Cloud: {len(lidar_pts):,} points, Trajectory: {len(lidar_poses)} poses")

    print("Loading D455 data...")
    d455_traj = np.load(scan_dir / "d455_trajectory.npz")
    d455_ts = d455_traj["timestamps_ns"]
    frame_numbers = d455_traj["frame_numbers"]
    fx = float(d455_traj["intrinsics_fx"])
    fy = float(d455_traj["intrinsics_fy"])
    cx = float(d455_traj["intrinsics_cx"])
    cy = float(d455_traj["intrinsics_cy"])
    print(f"  {len(frame_numbers)} frames, intrinsics: fx={fx:.1f} fy={fy:.1f}")

    # ---- Pick the primary frame for grid search ----
    # Use a frame near the middle of the scan (most stable trajectory region)
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

    # ---- Grid search + ICP on primary frame ----
    print(f"\nRunning rotation grid search + ICP (voxel={args.voxel_size}m)...")
    t_start = time.time()
    T_best, fitness_best = align_single_frame(
        d455_pcd, lidar_local, args.voxel_size
    )
    print(f"Grid search completed in {time.time() - t_start:.1f}s")

    if fitness_best < 0.1:
        print("ERROR: Alignment failed (fitness too low)")
        return 1

    # ---- Validate on additional frames ----
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

    # ---- Aggregate ----
    all_results = [(T_best, fitness_best, primary_frame_num)] + val_results

    # Filter by translation consistency (bracket offset should be small and consistent)
    translations = np.array([r[0][:3, 3] for r in all_results])
    median_t = np.median(translations, axis=0)
    t_dists = np.linalg.norm(translations - median_t, axis=1)
    consistent = t_dists < 0.10  # within 10cm of median

    good_results = [r for r, c in zip(all_results, consistent) if c]
    if not good_results:
        good_results = all_results  # fall back to all

    print(f"\n{len(good_results)} / {len(all_results)} results consistent "
          f"(within 10cm of median)")

    # Weighted average of good results
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

    # ---- Report ----
    euler = Rotation.from_matrix(T_final[:3, :3]).as_euler("xyz", degrees=True)
    trans = T_final[:3, 3]

    print(f"\nT_lidar_from_d455:")
    print(f"  Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]m "
          f"(magnitude: {np.linalg.norm(trans) * 100:.2f}cm)")
    print(f"  Rotation (euler XYZ): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

    if len(good_results) >= 2:
        trans_std = np.std([r[0][:3, 3] for r in good_results], axis=0)
        print(f"  Translation std: [{trans_std[0]:.4f}, {trans_std[1]:.4f}, "
              f"{trans_std[2]:.4f}]m")
        print(f"  Mean fitness: {np.mean([r[1] for r in good_results]):.4f}")

    # ---- Visualize ----
    if args.visualize:
        visualize_alignment(d455_dir, primary_frame_num, primary_ts,
                            lidar_pts, lidar_ts, lidar_poses,
                            T_final, fx, fy, cx, cy)

    # ---- Save ----
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

    return 0


def visualize_alignment(d455_dir, frame_num, frame_ts,
                         lidar_pts, lidar_ts, lidar_poses,
                         T_lidar_from_d455, fx, fy, cx, cy):
    """Show D455 cloud overlaid on LiDAR cloud using the computed extrinsic."""
    depth_path = d455_dir / f"frame_{frame_num:03d}_depth.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    d455_pcd = depth_to_pointcloud(depth, fx, fy, cx, cy, MAX_DEPTH)

    # Color D455 cloud red
    d455_colors = np.zeros((len(d455_pcd.points), 3))
    d455_colors[:, 0] = 1.0
    d455_pcd.colors = o3d.utility.Vector3dVector(d455_colors)

    # Transform D455 to world frame
    T_world_lidar = interpolate_pose_at_timestamp(frame_ts, lidar_ts, lidar_poses)
    T_world_d455 = T_world_lidar @ T_lidar_from_d455
    d455_pcd.transform(T_world_d455)

    # Subsample LiDAR for visualization
    lidar_vis = o3d.geometry.PointCloud()
    lidar_vis.points = o3d.utility.Vector3dVector(lidar_pts)
    lidar_vis = lidar_vis.voxel_down_sample(0.02)
    lidar_vis.paint_uniform_color([0.5, 0.5, 0.5])

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    frame.transform(T_world_lidar)

    print(f"Visualization: Red=D455 frame {frame_num}, Gray=LiDAR cloud")
    o3d.visualization.draw_geometries(
        [lidar_vis, d455_pcd, frame],
        window_name=f"Alignment (frame {frame_num})",
        width=1280, height=720,
    )


if __name__ == "__main__":
    sys.exit(main())
