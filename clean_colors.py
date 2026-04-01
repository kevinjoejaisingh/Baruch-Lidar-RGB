#!/usr/bin/env python3
"""
Clean up colored point cloud by removing color outliers.

For each point, compares its color to its spatial neighbors.
Points with colors that disagree strongly with their neighborhood
get replaced with the neighborhood median color. This removes
scattered wrong-color artifacts from misaligned projections while
preserving real color edges.

Usage:
  python clean_colors.py <input.ply>                    # Writes <input>_clean.ply
  python clean_colors.py <input.ply> -o output.ply      # Custom output path
  python clean_colors.py <input.ply> --radius 0.02      # Larger neighborhood (more smoothing)
  python clean_colors.py <input.ply> --threshold 40     # More aggressive outlier removal
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def clean_color_outliers(pcd, radius=0.015, min_neighbors=5, threshold=30.0):
    """
    Replace color outliers with neighborhood median.

    For each point, find neighbors within `radius`. If the point's color
    differs from the neighbor median by more than `threshold` (in RGB L1 distance),
    replace it with the median.

    Args:
        pcd: Open3D PointCloud with colors
        radius: Search radius in meters
        min_neighbors: Minimum neighbors to perform filtering
        threshold: Color distance threshold (0-255 scale) to consider outlier

    Returns:
        cleaned PointCloud
    """
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.float32)
    n = len(points)

    print(f"Building KD-tree for {n:,} points...")
    tree = o3d.geometry.KDTreeFlann(pcd)

    cleaned_colors = colors.copy()
    n_fixed = 0

    print(f"Filtering (radius={radius*1000:.0f}mm, threshold={threshold:.0f})...")
    for i in range(n):
        [k, idx, _] = tree.search_radius_vector_3d(points[i], radius)

        if k < min_neighbors:
            continue

        neighbor_colors = colors[idx[1:]]  # exclude self
        median_color = np.median(neighbor_colors, axis=0)

        # L1 distance in RGB space
        dist = np.sum(np.abs(colors[i] - median_color))

        if dist > threshold:
            cleaned_colors[i] = median_color
            n_fixed += 1

        if (i + 1) % 500000 == 0:
            print(f"  {i + 1:,}/{n:,} ({n_fixed:,} fixed)")

    print(f"Fixed {n_fixed:,} / {n:,} outlier points ({n_fixed/n*100:.1f}%)")

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(cleaned_colors.astype(np.float64) / 255.0)
    if pcd.has_normals():
        result.normals = pcd.normals

    return result


def smooth_colors(pcd, radius=0.01, iterations=1):
    """
    Gentle bilateral-style color smoothing.

    For each point, average colors of spatial neighbors, weighted by
    color similarity (points with similar colors contribute more).
    This smooths noise without blurring edges.
    """
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.float32)
    n = len(points)

    tree = o3d.geometry.KDTreeFlann(pcd)

    for it in range(iterations):
        new_colors = colors.copy()
        print(f"Smoothing pass {it + 1}/{iterations}...")

        for i in range(n):
            [k, idx, dists] = tree.search_radius_vector_3d(points[i], radius)

            if k < 3:
                continue

            neighbor_colors = colors[idx]

            # Color similarity weight (Gaussian in color space)
            color_diffs = np.sum(np.abs(neighbor_colors - colors[i]), axis=1)
            color_weights = np.exp(-color_diffs ** 2 / (2 * 30.0 ** 2))

            # Spatial weight (already within radius, use distance)
            spatial_dists = np.sqrt(np.array(dists))
            spatial_weights = np.exp(-spatial_dists ** 2 / (2 * (radius / 2) ** 2))

            weights = color_weights * spatial_weights
            weights_sum = weights.sum()

            if weights_sum > 0:
                new_colors[i] = (neighbor_colors * weights[:, np.newaxis]).sum(axis=0) / weights_sum

            if (i + 1) % 500000 == 0:
                print(f"  {i + 1:,}/{n:,}")

        colors = new_colors

    result = o3d.geometry.PointCloud()
    result.points = pcd.points
    result.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    if pcd.has_normals():
        result.normals = pcd.normals

    return result


def main():
    parser = argparse.ArgumentParser(description="Clean color outliers in point cloud")
    parser.add_argument("input", type=str, help="Input PLY file")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output PLY path (default: <input>_clean.ply)")
    parser.add_argument("--radius", type=float, default=0.015,
                        help="Neighbor search radius in meters (default: 0.015 = 15mm)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Color outlier threshold 0-255 (default: 30)")
    parser.add_argument("--smooth", action="store_true",
                        help="Also apply bilateral color smoothing after cleanup")
    parser.add_argument("--smooth-radius", type=float, default=0.01,
                        help="Smoothing radius (default: 10mm)")
    parser.add_argument("--smooth-iterations", type=int, default=1,
                        help="Number of smoothing passes (default: 1)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_clean.ply")

    print(f"Loading {input_path}...")
    pcd = o3d.io.read_point_cloud(str(input_path))
    print(f"  {len(pcd.points):,} points")

    # Step 1: Remove color outliers
    pcd = clean_color_outliers(pcd, radius=args.radius, threshold=args.threshold)

    # Step 2: Optional bilateral smoothing
    if args.smooth:
        pcd = smooth_colors(pcd, radius=args.smooth_radius,
                           iterations=args.smooth_iterations)

    # Save
    o3d.io.write_point_cloud(str(output_path), pcd)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
