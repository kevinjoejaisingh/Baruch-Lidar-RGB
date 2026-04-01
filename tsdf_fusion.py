#!/usr/bin/env python3
"""
TSDF Fusion: Fuse D455 RGB-D frames into a solid colored mesh.

Instead of projecting color onto sparse LiDAR points, this integrates
all D455 depth+RGB frames into a Truncated Signed Distance Function
voxel grid, then extracts a watertight colored mesh via marching cubes.

Colors are pixel-perfect (same sensor) and the mesh has no gaps.
LiDAR trajectory provides the camera poses via the extrinsic calibration.

Usage:
  python tsdf_fusion.py <scan_dir>                      # Default 5mm voxels
  python tsdf_fusion.py <scan_dir> --voxel-size 0.008   # 8mm voxels (faster, less detail)
  python tsdf_fusion.py <scan_dir> --voxel-size 0.003   # 3mm voxels (more detail, more RAM)
  python tsdf_fusion.py <scan_dir> --max-depth 2.5      # Limit depth range
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from utils.timestamps import interpolate_pose_at_timestamp

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


def load_data(scan_dir):
    """Load trajectory, intrinsics, extrinsic, and frame list."""
    # LiDAR trajectory (provides poses)
    lidar_traj = np.load(scan_dir / "lidar_trajectory.npz")
    lidar_poses = lidar_traj["poses"]
    lidar_ts = lidar_traj["timestamps_ns"]
    print(f"LiDAR trajectory: {len(lidar_poses)} poses")

    # D455 intrinsics
    d455_dir = scan_dir / "d455"
    with open(d455_dir / "intrinsics.json") as f:
        intr = json.load(f)

    width = int(intr.get("width", 1280))
    height = int(intr.get("height", 720))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        float(intr["fx"]), float(intr["fy"]),
        float(intr["cx"]), float(intr["cy"]),
    )
    dist_coeffs = np.array(intr["coeffs"]) if "coeffs" in intr else None
    camera_matrix = np.array([
        [float(intr["fx"]), 0, float(intr["cx"])],
        [0, float(intr["fy"]), 0],
        [0, 0, 1],
    ])
    camera_matrix[1, 2] = float(intr["cy"])

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
    print(f"Extrinsic: translation={np.linalg.norm(T_lidar_from_d455[:3, 3])*100:.2f}cm")

    # Frame list with timestamps
    frame_numbers = sorted(
        int(p.stem.split("_")[1]) for p in d455_dir.glob("frame_*_rgb.png")
    )
    frames = []
    for num in frame_numbers:
        imu_path = d455_dir / f"frame_{num:03d}_imu.json"
        if not imu_path.exists():
            continue
        with open(imu_path) as f:
            imu = json.load(f)
        ts = imu.get("capture_timestamp_ns", int(imu["timestamp_ms"] * 1e6))
        frames.append({"num": num, "ts": ts})

    print(f"D455: {len(frames)} frames, {width}x{height}")

    return {
        "lidar_poses": lidar_poses,
        "lidar_ts": lidar_ts,
        "intrinsic": intrinsic,
        "dist_coeffs": dist_coeffs,
        "camera_matrix": camera_matrix,
        "T_lidar_from_d455": T_lidar_from_d455,
        "frames": frames,
        "width": width,
        "height": height,
    }


def get_camera_pose(frame_ts, data):
    """Get camera-to-world transform for a D455 frame timestamp."""
    T_world_lidar = interpolate_pose_at_timestamp(
        frame_ts, data["lidar_ts"], data["lidar_poses"]
    )
    T_world_camera = T_world_lidar @ data["T_lidar_from_d455"]
    return T_world_camera


def main():
    parser = argparse.ArgumentParser(description="TSDF Fusion from D455 RGB-D frames")
    parser.add_argument("scan_dir", type=str)
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="TSDF voxel size in meters (default: 0.005 = 5mm)")
    parser.add_argument("--sdf-trunc", type=float, default=None,
                        help="SDF truncation distance (default: 5x voxel size)")
    parser.add_argument("--max-depth", type=float, default=3.0,
                        help="Max depth to integrate in meters (default: 3.0)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <scan_dir>/tsdf_mesh.ply)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    sdf_trunc = args.sdf_trunc or (args.voxel_size * 5)
    output_path = Path(args.output) if args.output else scan_dir / "tsdf_mesh.ply"

    print(f"TSDF Fusion: voxel={args.voxel_size*1000:.1f}mm, "
          f"trunc={sdf_trunc*1000:.1f}mm, max_depth={args.max_depth}m")
    print(f"Loading data...")

    data = load_data(scan_dir)

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    d455_dir = scan_dir / "d455"
    frames_to_process = data["frames"][::args.skip]
    n_frames = len(frames_to_process)
    print(f"Integrating {n_frames} frames...")

    for i, frame_info in enumerate(frames_to_process):
        num = frame_info["num"]
        ts = frame_info["ts"]

        # Load RGB and depth
        rgb_path = d455_dir / f"frame_{num:03d}_rgb.png"
        depth_path = d455_dir / f"frame_{num:03d}_depth.png"
        if not rgb_path.exists() or not depth_path.exists():
            continue

        color_raw = cv2.imread(str(rgb_path))
        color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        # Undistort if needed
        if data["dist_coeffs"] is not None:
            color_raw = cv2.undistort(color_raw, data["camera_matrix"], data["dist_coeffs"])
            depth_raw = cv2.undistort(depth_raw, data["camera_matrix"], data["dist_coeffs"])

        # Clamp depth to max range
        depth_raw[depth_raw > int(args.max_depth * 1000)] = 0

        # Convert to Open3D images
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_raw))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_raw.astype(np.uint16)))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=args.max_depth,
            convert_rgb_to_intensity=False,
        )

        # Camera pose: Open3D expects extrinsic as world-to-camera
        T_world_camera = get_camera_pose(ts, data)
        T_camera_world = np.linalg.inv(T_world_camera)

        volume.integrate(rgbd, data["intrinsic"], T_camera_world)

        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  Frame {i + 1}/{n_frames} (frame_{num:03d})")

    # Extract mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"Mesh: {n_verts:,} vertices, {n_tris:,} triangles")

    # Save
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    # Also export as point cloud for viewers that prefer it
    pcd_path = output_path.with_name("tsdf_cloud.ply")
    pcd = mesh.sample_points_uniformly(number_of_points=min(n_verts * 2, 10_000_000))
    pcd.colors = pcd.colors  # preserve colors from mesh sampling
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    pcd_size = pcd_path.stat().st_size / 1024 / 1024
    print(f"Saved point cloud: {pcd_path} ({pcd_size:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
