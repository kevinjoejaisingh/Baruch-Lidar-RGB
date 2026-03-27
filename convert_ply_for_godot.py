#!/usr/bin/env python3
"""
Convert a colored PLY to binary format for the Godot point cloud viewer.

Writes a .bin file with:
  - uint32: number of points
  - float32[n*3]: positions (x, y, z) — Y-up for Godot
  - float32[n*3]: colors (r, g, b) in 0-1 range

Usage:
  python3 convert_ply_for_godot.py <input.ply> [output.bin]
  python3 convert_ply_for_godot.py folsom_test/debug_frames0-617_clean.ply
"""

import sys
import numpy as np
import open3d as o3d

input_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else "godot_viewer/cloud.bin"

print(f"Loading {input_path}...")
pcd = o3d.io.read_point_cloud(input_path)
points = np.asarray(pcd.points, dtype=np.float32)
colors = np.asarray(pcd.colors, dtype=np.float32)
n = len(points)
print(f"  {n:,} points")

# Coordinate conversion: Open3D (Z-up or arbitrary) → Godot (Y-up, right-handed)
# Swap Y and Z, negate new Z for right-handed
converted = points.copy()
converted[:, 0] = points[:, 0]      # X stays
converted[:, 1] = points[:, 2]      # Y = old Z (up)
converted[:, 2] = -points[:, 1]     # Z = -old Y (into screen)

print(f"Writing {output_path}...")
with open(output_path, "wb") as f:
    np.array([n], dtype=np.uint32).tofile(f)
    converted.tofile(f)
    colors.tofile(f)

size_mb = (4 + n * 24) / 1024 / 1024
print(f"  Saved: {output_path} ({size_mb:.1f} MB)")
print(f"  Bounds: X[{converted[:,0].min():.1f}, {converted[:,0].max():.1f}] "
      f"Y[{converted[:,1].min():.1f}, {converted[:,1].max():.1f}] "
      f"Z[{converted[:,2].min():.1f}, {converted[:,2].max():.1f}]")
