#!/usr/bin/env python3
"""Project a single frame onto the full LiDAR cloud and save/open in MeshLab."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from project_color import load_scan_data, project_single_frame

scan_dir = Path(sys.argv[1]).expanduser()
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print("Loading scan data...")
data = load_scan_data(scan_dir)

print(f"Projecting frame index {frame_idx}...")
indices, colors = project_single_frame(scan_dir, data, frame_idx)

n = len(data["points"])
all_colors = np.full((n, 3), 0.3)  # dark gray for uncolored
if len(indices) > 0:
    all_colors[indices] = colors.astype(np.float64) / 255.0
print(f"Colored {len(indices):,} / {n:,} points ({len(indices)/n*100:.1f}%)")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data["points"])
pcd.colors = o3d.utility.Vector3dVector(all_colors)

out_path = scan_dir / f"debug_frame{frame_idx}_on_cloud.ply"
o3d.io.write_point_cloud(str(out_path), pcd)
print(f"Saved: {out_path}")

print("Opening in MeshLab...")
subprocess.Popen(["meshlab", str(out_path)])
