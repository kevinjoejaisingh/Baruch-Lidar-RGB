#!/usr/bin/env python3
"""
Process LiDAR bag → drift-corrected PLY + trajectory NPZ.

Pipeline (based on export_refined.py):
  1. Read /rko_lio/odometry and /rko_lio/frame from ROS2 bag
  2. Match frames to nearest odometry, transform to world frame
  3. Group frames into chunks, build per-chunk point clouds
  4. Pairwise ICP between consecutive chunks + loop closure (first↔last)
  5. Open3D pose graph optimization to distribute drift
  6. Apply chunk corrections to all points and per-frame poses
  7. Voxel downsample + outlier removal → PLY

Outputs:
  - lidar_cloud.ply         (corrected, downsampled, cleaned)
  - lidar_trajectory.npz    (corrected per-frame 4x4 poses + timestamps)

Usage:
  python process_lidar.py <scan_dir>
  python process_lidar.py scans/scan_20260227_140000
  python process_lidar.py scans/scan_20260227_140000 --chunk-size 10 --icp-voxel 0.05
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

VOXEL_SIZE = CFG["processing"]["lidar_voxel_size"]
OUTLIER_NB = CFG["processing"]["lidar_outlier_neighbors"]
OUTLIER_STD = CFG["processing"]["lidar_outlier_std"]
MIN_RANGE_M = CFG["processing"]["lidar_min_range_m"]

FOV_FILTER = CFG["processing"]["lidar_fov_filter"]
H_TAN = np.tan(np.radians(CFG["processing"]["lidar_fov_h_deg"] / 2))
V_TAN = np.tan(np.radians(CFG["processing"]["lidar_fov_v_deg"] / 2))

if FOV_FILTER:
    _ext_path = Path(CFG["paths"]["permanent_extrinsic"]).expanduser()
    with open(_ext_path) as _f:
        _ext = json.load(_f)
    _T_lidar_from_d455 = np.array(_ext["transform"], dtype=np.float64)
    _T_d455_from_lidar = np.linalg.inv(_T_lidar_from_d455)
    R_D455 = _T_d455_from_lidar[:3, :3].astype(np.float32)
    t_D455 = _T_d455_from_lidar[:3, 3].astype(np.float32)


# ---------------------------------------------------------------------------
# Vectorized PointCloud2 parser
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


def parse_pc2_vectorized(raw_data, typestore):
    """Parse PointCloud2 using numpy structured arrays. Returns (N,3) float32."""
    msg = typestore.deserialize_cdr(raw_data, "sensor_msgs/msg/PointCloud2")

    fields = sorted(msg.fields, key=lambda f: f.offset)
    dt_fields = []
    prev_end = 0
    for field in fields:
        if field.offset > prev_end:
            dt_fields.append((f"_pad{prev_end}", f"V{field.offset - prev_end}"))
        np_dtype = _DTYPE_MAP.get(field.datatype, np.float32)
        dt_fields.append((field.name, np_dtype))
        prev_end = field.offset + np.dtype(np_dtype).itemsize
    if prev_end < msg.point_step:
        dt_fields.append((f"_pad{prev_end}", f"V{msg.point_step - prev_end}"))

    dt = np.dtype(dt_fields)
    n_points = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n_points)

    x = arr["x"].astype(np.float32)
    y = arr["y"].astype(np.float32)
    z = arr["z"].astype(np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid &= ~((x == 0) & (y == 0) & (z == 0))
    return np.column_stack([x[valid], y[valid], z[valid]])


def parse_odom(raw_data, typestore):
    """Parse Odometry → (position, rotation_matrix, 4x4 pose)."""
    msg = typestore.deserialize_cdr(raw_data, "nav_msgs/msg/Odometry")
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    pos = np.array([p.x, p.y, p.z])
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    return pos, rot, pose


# ---------------------------------------------------------------------------
# ICP registration (from export_refined.py)
# ---------------------------------------------------------------------------

def pairwise_icp(source, target, voxel_size, max_dist_multiplier=1.5):
    """Point-to-plane ICP between two clouds already in global frame."""
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    radius = voxel_size * 2
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )

    distance_threshold = voxel_size * max_dist_multiplier
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, distance_threshold, result.transformation
    )
    return result.transformation, information, result.fitness


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process LiDAR bag with drift correction")
    parser.add_argument("scan_dir", help="Path to scan directory")
    parser.add_argument("--chunk-size", type=int, default=10,
                        help="Frames per chunk for ICP (default: 10)")
    parser.add_argument("--icp-voxel", type=float, default=0.05,
                        help="Voxel size for ICP in meters (default: 0.05)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    bag_path = scan_dir / "bag"
    chunk_size = args.chunk_size
    icp_voxel = args.icp_voxel

    if not bag_path.exists():
        if (scan_dir / "metadata.yaml").exists():
            bag_path = scan_dir
        else:
            print(f"ERROR: No bag found at {bag_path}")
            return 1

    output_ply = scan_dir / "lidar_cloud.ply"
    output_traj = scan_dir / "lidar_trajectory.npz"
    typestore = get_typestore(Stores.ROS2_JAZZY)

    # ==== 1. Read bag ====
    print(f"Reading bag: {bag_path}")
    odom_msgs = []
    with Reader(bag_path) as reader:
        odom_conns = [c for c in reader.connections if c.topic == "/rko_lio/odometry"]
        for conn, timestamp, raw_data in reader.messages(connections=odom_conns):
            pos, rot, pose = parse_odom(raw_data, typestore)
            odom_msgs.append((timestamp, pos, rot, pose))

    frame_msgs = []
    with Reader(bag_path) as reader:
        frame_conns = [c for c in reader.connections if c.topic == "/rko_lio/frame"]
        for conn, timestamp, raw_data in reader.messages(connections=frame_conns):
            frame_msgs.append((timestamp, raw_data))

    print(f"Read {len(odom_msgs)} odom, {len(frame_msgs)} frames")
    if len(odom_msgs) == 0 or len(frame_msgs) == 0:
        print("ERROR: No odometry or frame messages found")
        return 1

    # ==== 2. Parse frames + match to nearest odom ====
    print("\nParsing frames...")
    odom_times = np.array([m[0] for m in odom_msgs])
    frames = []  # (points_local, pose_4x4, timestamp)

    for i, (ft, raw_data) in enumerate(frame_msgs):
        points = parse_pc2_vectorized(raw_data, typestore)
        if len(points) == 0:
            continue
        range_sq = (points ** 2).sum(axis=1)
        points = points[range_sq >= MIN_RANGE_M ** 2]
        if len(points) == 0:
            continue
        if FOV_FILTER:
            pts_cam = (R_D455 @ points.T).T + t_D455
            z = pts_cam[:, 2]
            fov_mask = (z > 0) & \
                       (np.abs(pts_cam[:, 0]) <= H_TAN * z) & \
                       (np.abs(pts_cam[:, 1]) <= V_TAN * z)
            points = points[fov_mask]
            if len(points) == 0:
                continue
        idx = np.argmin(np.abs(odom_times - ft))
        pose = odom_msgs[idx][3]
        frames.append((points, pose, odom_msgs[idx][0]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(frame_msgs)} frames...")

    print(f"Parsed {len(frames)} valid frames")

    # ==== 3. Group into chunks ====
    num_chunks = (len(frames) + chunk_size - 1) // chunk_size
    print(f"\nGrouping into {num_chunks} chunks of ~{chunk_size} frames")

    chunk_frame_lists = []
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, len(frames))
        chunk_frame_lists.append(frames[start:end])

    # ==== 4. Build per-chunk point clouds (using raw odom) ====
    print("Building chunk point clouds...")
    chunk_clouds = []
    for chunk in chunk_frame_lists:
        all_pts = []
        for points, pose, _ in chunk:
            transformed = (pose[:3, :3] @ points.T).T + pose[:3, 3]
            all_pts.append(transformed)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
        chunk_clouds.append(pcd)

    # ==== 5. Pairwise ICP + loop closure ====
    print(f"\nRunning pairwise ICP (voxel={icp_voxel}m)...")
    t0 = time.time()
    icp_edges = []

    for i in range(num_chunks - 1):
        T, info, fitness = pairwise_icp(chunk_clouds[i], chunk_clouds[i + 1], icp_voxel)
        icp_edges.append((i, i + 1, T, info, fitness, False))
        if (i + 1) % 5 == 0 or i == num_chunks - 2:
            print(f"  Chunks {i:3d} -> {i+1:3d}: fitness={fitness:.3f}")

    if num_chunks > 2:
        T, info, fitness = pairwise_icp(
            chunk_clouds[-1], chunk_clouds[0], icp_voxel, max_dist_multiplier=3.0
        )
        icp_edges.append((num_chunks - 1, 0, T, info, fitness, True))
        print(f"  Loop closure {num_chunks-1:3d} ->   0: fitness={fitness:.3f}")

    print(f"ICP completed in {time.time() - t0:.1f}s")

    # ==== 6. Build Open3D pose graph + optimize ====
    print("\nBuilding pose graph...")
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for _ in range(num_chunks):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))

    successful, failed = 0, 0
    for (i, j, T, info, fitness, is_loop) in icp_edges:
        min_fitness = 0.05 if is_loop else 0.1
        if fitness > min_fitness:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i, j, T, info, uncertain=is_loop)
            )
            successful += 1
        else:
            print(f"  WARNING: Dropping edge {i}->{j} (fitness={fitness:.3f})")
            failed += 1
    print(f"  Edges: {successful} successful, {failed} dropped")

    print("Optimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=icp_voxel * 1.5,
        edge_prune_threshold=0.25,
        preference_loop_closure=2.0,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    chunk_corrections = [pose_graph.nodes[i].pose for i in range(num_chunks)]
    max_shift = 0
    for i, corr in enumerate(chunk_corrections):
        t_mag = np.linalg.norm(corr[:3, 3])
        angle = np.degrees(np.arccos(np.clip((np.trace(corr[:3, :3]) - 1) / 2, -1, 1)))
        max_shift = max(max_shift, t_mag)
        if i % 10 == 0 or i == num_chunks - 1:
            print(f"  Chunk {i:3d}: shift={t_mag:.4f}m  rot={angle:.2f}deg")
    print(f"  Max correction shift: {max_shift:.4f}m")

    # ==== 7. Apply corrections → merged cloud + corrected trajectory ====
    print("\nApplying corrections and merging...")
    all_points = []
    corrected_poses = []
    corrected_timestamps = []

    for c_idx, chunk in enumerate(chunk_frame_lists):
        correction = chunk_corrections[c_idx]
        for points, pose, timestamp in chunk:
            # Transform to global with raw odometry
            global_pts = (pose[:3, :3] @ points.T).T + pose[:3, 3]
            # Apply chunk correction
            ones = np.ones((len(global_pts), 1), dtype=np.float32)
            global_h = np.hstack([global_pts, ones])
            corrected = (correction @ global_h.T).T[:, :3]
            all_points.append(corrected)

            # Corrected pose = correction @ raw_pose
            corrected_pose = correction @ pose
            corrected_poses.append(corrected_pose)
            corrected_timestamps.append(timestamp)

        if (c_idx + 1) % 10 == 0:
            total = sum(len(p) for p in all_points)
            print(f"  {c_idx + 1}/{num_chunks} chunks, {total:,} points...")

    combined = np.vstack(all_points)
    print(f"Total: {len(frames)} frames, {len(combined):,} raw points")

    # ==== 8. Voxel downsample + outlier removal ====
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined)

    print(f"Downsampling (voxel {VOXEL_SIZE}m)...")
    pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"After downsample: {len(pcd_down.points):,}")

    print(f"Removing outliers (neighbors={OUTLIER_NB}, std={OUTLIER_STD})...")
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD
    )
    print(f"After cleanup: {len(pcd_clean.points):,}")

    # ==== 9. Save outputs ====
    o3d.io.write_point_cloud(str(output_ply), pcd_clean)
    print(f"\nSaved cloud: {output_ply} ({output_ply.stat().st_size / 1024 / 1024:.1f} MB)")

    poses_array = np.array(corrected_poses)
    timestamps_array = np.array(corrected_timestamps, dtype=np.int64)
    np.savez(output_traj, poses=poses_array, timestamps_ns=timestamps_array)
    print(f"Saved trajectory: {output_traj} ({len(poses_array)} corrected poses)")

    diffs = np.diff(timestamps_array)
    if np.all(diffs >= 0):
        print("  Timestamps: monotonic OK")
    else:
        print(f"  WARNING: {np.sum(diffs < 0)} non-monotonic timestamps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
