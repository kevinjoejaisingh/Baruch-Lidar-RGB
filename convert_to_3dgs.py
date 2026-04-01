#!/usr/bin/env python3
"""
Convert a LiDAR+D455 scan to nerfstudio Splatfacto format.

Takes our unorthodox pipeline data (LiDAR SLAM trajectory + D455 RGB frames +
extrinsic bracket calibration) and produces the transforms.json + images/ +
points3d.ply that nerfstudio expects.

The key conversion: for each D455 frame, we interpolate the LiDAR SLAM pose
at that timestamp, compose with the bracket extrinsic to get T_world_camera,
and write it in nerfstudio's transforms.json format.

Usage:
  python convert_to_3dgs.py hospital
  python convert_to_3dgs.py hospital --output hospital_3dgs --voxel-size 0.02
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


def interpolate_pose(timestamp_ns, lidar_ts, lidar_poses):
    """Interpolate LiDAR pose at a given timestamp using SLERP + linear."""
    idx = np.searchsorted(lidar_ts, timestamp_ns) - 1
    idx = np.clip(idx, 0, len(lidar_ts) - 2)

    t0, t1 = lidar_ts[idx], lidar_ts[idx + 1]
    if t1 == t0:
        return lidar_poses[idx].copy()

    alpha = float(timestamp_ns - t0) / float(t1 - t0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Linear interpolation for translation
    trans = (1 - alpha) * lidar_poses[idx][:3, 3] + alpha * lidar_poses[idx + 1][:3, 3]

    # SLERP for rotation
    r0 = Rotation.from_matrix(lidar_poses[idx][:3, :3])
    r1 = Rotation.from_matrix(lidar_poses[idx + 1][:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    rot = slerp(alpha)

    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = trans
    return T


def main():
    parser = argparse.ArgumentParser(description="Convert scan to nerfstudio 3DGS format")
    parser.add_argument("scan_dir", type=str, help="Path to scan directory (e.g., hospital)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <scan_dir>_3dgs)")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="Point cloud downsample voxel size in meters (default: 0.02)")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="JPEG quality for image conversion (default: 95)")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip image copying (if already done)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir).expanduser()
    output_dir = Path(args.output) if args.output else Path(f"{scan_dir.name}_3dgs")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Converting {scan_dir.name} → {output_dir}")
    print(f"=" * 60)

    # ── Step 1: Load all data ──────────────────────────────────────────────
    print("\n[1/6] Loading data...")

    # LiDAR trajectory
    lidar_traj = np.load(scan_dir / "lidar_trajectory.npz")
    lidar_poses = lidar_traj["poses"]
    lidar_ts = lidar_traj["timestamps_ns"]
    print(f"  LiDAR trajectory: {len(lidar_poses)} poses")

    # Extrinsic — prefer per-scan calibration, fall back to permanent
    per_scan_extrinsic = scan_dir / "extrinsic.json"
    if per_scan_extrinsic.exists():
        extrinsic_path = per_scan_extrinsic
        print(f"  Using per-scan extrinsic: {extrinsic_path}")
    else:
        extrinsic_path = Path(CFG["paths"]["permanent_extrinsic"]).expanduser()
        print(f"  Using permanent extrinsic: {extrinsic_path}")
    with open(extrinsic_path) as f:
        ext = json.load(f)
    T_lidar_from_d455 = np.array(ext["transform"])
    print(f"  Extrinsic: translation={np.linalg.norm(T_lidar_from_d455[:3, 3])*100:.2f}cm")

    # D455 intrinsics
    d455_dir = scan_dir / "d455"
    with open(d455_dir / "intrinsics.json") as f:
        intr = json.load(f)
    print(f"  Intrinsics: fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
          f"cx={intr['cx']:.1f} cy={intr['cy']:.1f}")

    # D455 frame list with timestamps
    frame_numbers = sorted(
        int(p.stem.split("_")[1]) for p in d455_dir.glob("frame_*_rgb.png")
    )
    d455_timestamps = []
    valid_frames = []
    for num in frame_numbers:
        imu_path = d455_dir / f"frame_{num:03d}_imu.json"
        if not imu_path.exists():
            continue
        with open(imu_path) as f:
            imu = json.load(f)
        ts = imu.get("capture_timestamp_ns", int(imu["timestamp_ms"] * 1e6))
        d455_timestamps.append(ts)
        valid_frames.append(num)

    d455_timestamps = np.array(d455_timestamps, dtype=np.int64)
    print(f"  D455 frames: {len(valid_frames)}")

    # ── Step 2: Compute camera poses ───────────────────────────────────────
    print("\n[2/6] Computing camera poses...")

    camera_poses = []
    for i, ts in enumerate(d455_timestamps):
        # Interpolate LiDAR pose at D455 timestamp
        T_world_lidar = interpolate_pose(ts, lidar_ts, lidar_poses)

        # Compose: T_world_camera = T_world_lidar @ T_lidar_from_d455
        T_world_camera = T_world_lidar @ T_lidar_from_d455
        camera_poses.append(T_world_camera)

    camera_poses = np.array(camera_poses)
    print(f"  Computed {len(camera_poses)} camera poses")

    # ── Step 3: Normalize to scene center ──────────────────────────────────
    print("\n[3/6] Normalizing poses to scene center...")

    positions = camera_poses[:, :3, 3]
    scene_center = np.mean(positions, axis=0)
    print(f"  Scene center: [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")

    # Translate all poses so scene is centered at origin
    for i in range(len(camera_poses)):
        camera_poses[i][:3, 3] -= scene_center

    # Check pose spread
    positions_centered = camera_poses[:, :3, 3]
    spread = np.max(np.abs(positions_centered))
    print(f"  Pose spread: ±{spread:.2f}m")

    # ── Step 4: Prepare images ─────────────────────────────────────────────
    images_dir = output_dir / "images"
    if not args.skip_images:
        print(f"\n[4/6] Preparing images...")
        images_dir.mkdir(exist_ok=True)

        for out_idx, frame_num in enumerate(valid_frames):
            rgb_path = d455_dir / f"frame_{frame_num:03d}_rgb.png"
            dst = images_dir / f"{out_idx:06d}.jpg"
            img = cv2.imread(str(rgb_path))
            if img is not None:
                cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

            if (out_idx + 1) % 50 == 0:
                print(f"  {out_idx + 1}/{len(valid_frames)} images")

        print(f"  Wrote {len(valid_frames)} images to {images_dir}")
    else:
        print(f"\n[4/6] Skipping image copy (--skip-images)")

    # ── Step 5: Build transforms.json ──────────────────────────────────────
    print(f"\n[5/6] Building transforms.json...")

    # Parse distortion coefficients
    coeffs = intr.get("coeffs", [0, 0, 0, 0, 0])
    if len(coeffs) < 5:
        coeffs = coeffs + [0.0] * (5 - len(coeffs))

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(intr["fx"]),
        "fl_y": float(intr["fy"]),
        "cx": float(intr["cx"]),
        "cy": float(intr["cy"]),
        "w": int(intr.get("width", 1280)),
        "h": int(intr.get("height", 720)),
        "k1": float(coeffs[0]),
        "k2": float(coeffs[1]),
        "p1": float(coeffs[2]),
        "p2": float(coeffs[3]),
        "k3": float(coeffs[4]) if len(coeffs) > 4 else 0.0,
        "frames": [],
    }

    for out_idx in range(len(valid_frames)):
        transforms["frames"].append({
            "file_path": f"images/{out_idx:06d}.jpg",
            "transform_matrix": camera_poses[out_idx].tolist(),
        })

    transforms_path = output_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)
    print(f"  Wrote {transforms_path} ({len(transforms['frames'])} frames)")

    # ── Step 6: Prepare point cloud ────────────────────────────────────────
    print(f"\n[6/6] Preparing point cloud...")

    cloud_path = scan_dir / "lidar_cloud.ply"
    if cloud_path.exists():
        pcd = o3d.io.read_point_cloud(str(cloud_path))
        points = np.asarray(pcd.points)
        print(f"  Loaded {len(points):,} points from {cloud_path.name}")

        # Apply same scene center normalization
        points -= scene_center
        pcd.points = o3d.utility.Vector3dVector(points)

        # Downsample
        if args.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
            print(f"  After {args.voxel_size*1000:.0f}mm downsample: {len(pcd.points):,} points")

        pts_path = output_dir / "points3d.ply"
        o3d.io.write_point_cloud(str(pts_path), pcd)
        size_mb = pts_path.stat().st_size / 1024 / 1024
        print(f"  Wrote {pts_path} ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: No lidar_cloud.ply found, skipping point cloud")

    # ── Done ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Conversion complete!")
    print(f"  Output: {output_dir}/")
    print(f"  Frames: {len(transforms['frames'])}")
    print(f"\nNext steps:")
    print(f"  1. pip install nerfstudio")
    print(f"  2. ns-train splatfacto --data {output_dir}")
    print(f"     (or: ns-train splatfacto nerfstudio-data --data {output_dir})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
