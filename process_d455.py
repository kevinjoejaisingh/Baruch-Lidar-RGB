#!/usr/bin/env python3
"""
Process D455 frames → trajectory NPZ via ICP alignment + pose graph optimization.

Pipeline:
  1. Convert RGB-D frames to point clouds (vectorized depth unprojection)
  2. Align consecutive frames with Point-to-Plane ICP (IMU initial guess)
  3. Detect loop closure (first/last frame ICP)
  4. Pose graph optimization (scipy least_squares)

Reuses the pose_graph module from ~/3dRoom/src/ via sys.path.

Usage:
  python process_d455.py <scan_dir>
  python process_d455.py scans/scan_20260227_140000
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

VOXEL_SIZE = CFG["processing"]["d455_voxel_size"]
ICP_MAX_DIST = CFG["processing"]["icp_max_distance"]
ICP_MAX_ITER = CFG["processing"]["icp_max_iterations"]
NORMAL_RADIUS = CFG["processing"]["normal_radius"]
NORMAL_MAX_NN = CFG["processing"]["normal_max_nn"]
LC_MAX_DIST = CFG["processing"]["loop_closure_max_distance"]
LC_MAX_ITER = CFG["processing"]["loop_closure_max_iterations"]
PG_METHOD = CFG["processing"]["pose_graph_method"]
PG_MAX_ITER = CFG["processing"]["pose_graph_max_iterations"]
PG_TOL = CFG["processing"]["pose_graph_tolerance"]
MAX_DEPTH = CFG["d455"]["max_depth_m"]

# Import pose_graph from 3dRoom
THREED_ROOM_SRC = Path(CFG["paths"]["threed_room_src"]).expanduser()
sys.path.insert(0, str(THREED_ROOM_SRC))
from pose_graph import PoseGraphOptimizer, build_pose_graph_from_transforms


# ---------------------------------------------------------------------------
# Point cloud creation (from frame_to_pointcloud.py, vectorized)
# ---------------------------------------------------------------------------

def create_point_cloud(rgb_image, depth_image, fx, fy, cx, cy, max_depth_m=3.0):
    """Convert RGB-D to Open3D point cloud using vectorized unprojection."""
    height, width = depth_image.shape

    u_coords = np.arange(width)
    v_coords = np.arange(height)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    u_flat = u_grid.flatten().astype(np.float64)
    v_flat = v_grid.flatten().astype(np.float64)
    z_flat = depth_image.flatten().astype(np.float64) / 1000.0

    valid = (z_flat > 0) & (z_flat < max_depth_m)
    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    z_valid = z_flat[valid]

    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    points = np.stack([x_valid, y_valid, z_valid], axis=1)
    colors = rgb_image.reshape(-1, 3)[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_d455_frame(d455_dir, frame_num, fx, fy, cx, cy):
    """Load RGB, depth, IMU and create point cloud for a D455 frame."""
    prefix = f"frame_{frame_num:03d}"
    rgb_path = d455_dir / f"{prefix}_rgb.png"
    depth_path = d455_dir / f"{prefix}_depth.png"
    imu_path = d455_dir / f"{prefix}_imu.json"

    if not rgb_path.exists():
        return None, None, None

    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    with open(imu_path) as f:
        imu = json.load(f)

    if np.all(depth == 0):
        return None, imu, None

    pcd = create_point_cloud(rgb, depth, fx, fy, cx, cy, MAX_DEPTH)
    return pcd, imu, None


def list_d455_frames(d455_dir):
    """List available frame numbers."""
    frames = []
    for f in d455_dir.glob("frame_*_rgb.png"):
        try:
            num = int(f.stem.split("_")[1])
            frames.append(num)
        except (ValueError, IndexError):
            pass
    return sorted(frames)


# ---------------------------------------------------------------------------
# ICP helpers (from align_frames.py)
# ---------------------------------------------------------------------------

def preprocess_for_icp(pcd, voxel_size):
    """Downsample and compute normals."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    return pcd_down


def rotation_matrix_from_pitch_roll(pitch_deg, roll_deg):
    """Create rotation matrix from pitch and roll (degrees)."""
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    return Rz @ Rx


def imu_initial_guess(imu_source, imu_target):
    """Compute initial rotation guess from IMU pitch/roll data."""
    R_src = rotation_matrix_from_pitch_roll(
        imu_source["orientation"]["pitch_deg"],
        imu_source["orientation"]["roll_deg"],
    )
    R_tgt = rotation_matrix_from_pitch_roll(
        imu_target["orientation"]["pitch_deg"],
        imu_target["orientation"]["roll_deg"],
    )
    T = np.eye(4)
    T[:3, :3] = R_tgt @ R_src.T
    return T


def run_icp(source, target, initial_transform, max_dist, max_iter):
    """Run Point-to-Plane ICP."""
    return o3d.pipelines.registration.registration_icp(
        source, target,
        max_dist,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=max_iter,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_d455.py <scan_dir>")
        return 1

    scan_dir = Path(sys.argv[1]).expanduser()
    d455_dir = scan_dir / "d455"

    if not d455_dir.exists():
        print(f"ERROR: D455 directory not found: {d455_dir}")
        return 1

    output_traj = scan_dir / "d455_trajectory.npz"

    # Load intrinsics
    intrinsics_path = d455_dir / "intrinsics.json"
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            intr = json.load(f)
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        w, h = intr["width"], intr["height"]
        print(f"Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    else:
        print("WARNING: No intrinsics.json found, using D455 defaults for 1280x720")
        fx, fy, cx, cy = 640.0, 640.0, 640.0, 360.0
        w, h = 1280, 720

    # Discover frames
    frame_nums = list_d455_frames(d455_dir)
    if len(frame_nums) < 2:
        print(f"ERROR: Need at least 2 frames, found {len(frame_nums)}")
        return 1
    print(f"Found {len(frame_nums)} frames: {frame_nums[0]}..{frame_nums[-1]}")

    # ---- Step 1: Load point clouds and IMU data ----
    print("\n[1/4] Loading frames and creating point clouds...")
    pcds = {}
    imus = {}
    timestamps = {}
    for num in frame_nums:
        pcd, imu, _ = load_d455_frame(d455_dir, num, fx, fy, cx, cy)
        if pcd is None or len(pcd.points) == 0:
            print(f"  Frame {num}: skipped (no valid points)")
            continue
        pcds[num] = pcd
        imus[num] = imu
        timestamps[num] = imu.get("capture_timestamp_ns", int(imu["timestamp_ms"] * 1e6))
        if len(pcds) % 10 == 0:
            print(f"  Loaded {len(pcds)} frames...")
    print(f"  {len(pcds)} valid frames loaded")

    valid_nums = sorted(pcds.keys())
    if len(valid_nums) < 2:
        print("ERROR: Not enough valid frames for alignment")
        return 1

    # ---- Step 2: Consecutive frame alignment ----
    print(f"\n[2/4] Aligning {len(valid_nums) - 1} consecutive frame pairs...")
    print(f"  Voxel: {VOXEL_SIZE}m, Max ICP dist: {ICP_MAX_DIST}m")

    transforms = []
    fitnesses = []

    for i in range(len(valid_nums) - 1):
        src_num = valid_nums[i + 1]
        tgt_num = valid_nums[i]

        src_down = preprocess_for_icp(pcds[src_num], VOXEL_SIZE)
        tgt_down = preprocess_for_icp(pcds[tgt_num], VOXEL_SIZE)

        initial = imu_initial_guess(imus[src_num], imus[tgt_num])
        result = run_icp(src_down, tgt_down, initial, ICP_MAX_DIST, ICP_MAX_ITER)

        transforms.append(result.transformation)
        fitnesses.append(result.fitness)

        if (i + 1) % 10 == 0 or i == len(valid_nums) - 2:
            print(f"  {i + 1}/{len(valid_nums) - 1}: "
                  f"frame {src_num}->{tgt_num} fitness={result.fitness:.4f} RMSE={result.inlier_rmse:.6f}")

    avg_fitness = np.mean(fitnesses)
    min_fitness = np.min(fitnesses)
    print(f"  Average fitness: {avg_fitness:.4f}, Min: {min_fitness:.4f}")

    bad = [(i, f) for i, f in enumerate(fitnesses) if f < 0.3]
    if bad:
        print(f"  WARNING: {len(bad)} pairs with fitness < 0.3")

    # ---- Step 3: Loop closure ----
    print(f"\n[3/4] Computing loop closure (frame {valid_nums[-1]} -> frame {valid_nums[0]})...")

    first_down = preprocess_for_icp(pcds[valid_nums[0]], VOXEL_SIZE)
    last_down = preprocess_for_icp(pcds[valid_nums[-1]], VOXEL_SIZE)

    # Try multiple max distances
    best_result = None
    for max_dist in [0.05, 0.10, 0.15]:
        lc_result = run_icp(last_down, first_down, np.eye(4), max_dist, LC_MAX_ITER)
        print(f"  max_dist={max_dist}: fitness={lc_result.fitness:.4f} RMSE={lc_result.inlier_rmse:.6f}")
        if best_result is None or lc_result.fitness > best_result.fitness:
            best_result = lc_result

    # Use the configured distance for final result
    lc_result = run_icp(last_down, first_down, np.eye(4), LC_MAX_DIST, LC_MAX_ITER)
    lc_transform = lc_result.transformation
    lc_fitness = lc_result.fitness

    rot_angle = np.degrees(np.arccos(np.clip((np.trace(lc_transform[:3, :3]) - 1) / 2, -1, 1)))
    trans_mag = np.linalg.norm(lc_transform[:3, 3])
    print(f"  Final: fitness={lc_fitness:.4f}, rotation={rot_angle:.2f}deg, translation={trans_mag*100:.1f}cm")

    # ---- Step 4: Pose graph optimization ----
    print(f"\n[4/4] Pose graph optimization ({PG_METHOD}, max_iter={PG_MAX_ITER})...")

    graph = build_pose_graph_from_transforms(transforms, fitnesses, lc_transform, lc_fitness)
    optimizer = PoseGraphOptimizer(graph, verbose=True)
    success, final_error = optimizer.optimize(
        method=PG_METHOD, max_iterations=PG_MAX_ITER, tolerance=PG_TOL
    )

    optimized_poses = optimizer.get_optimized_poses()

    # ---- Save trajectory ----
    poses_array = np.array(optimized_poses)  # (N, 4, 4)
    timestamps_array = np.array([timestamps[n] for n in valid_nums], dtype=np.int64)

    np.savez(
        output_traj,
        poses=poses_array,
        timestamps_ns=timestamps_array,
        intrinsics_fx=fx, intrinsics_fy=fy,
        intrinsics_cx=cx, intrinsics_cy=cy,
        intrinsics_w=w, intrinsics_h=h,
        frame_numbers=np.array(valid_nums),
    )
    print(f"\nSaved trajectory: {output_traj} ({len(poses_array)} poses)")

    # Verify timestamps
    diffs = np.diff(timestamps_array)
    if np.all(diffs >= 0):
        print("  Timestamps: monotonic OK")
    else:
        print(f"  WARNING: {np.sum(diffs < 0)} non-monotonic timestamps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
