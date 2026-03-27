#!/usr/bin/env python3
"""
Post-process a colored PLY to remove silhouette color bleeding artifacts.

At object boundaries (tree trunk vs sky), some points pick up background
color (bright white sky) instead of foreground color (dark bark). This
script detects and removes those points by finding color outliers relative
to their spatial neighbors.

Usage:
  python3 postprocess_ply.py <input.ply> [output.ply]
  python3 postprocess_ply.py folsom_test/debug_frames0-617.ply
  python3 postprocess_ply.py input.ply --brightness 0.7 --k 30 --diff 0.20
"""

import argparse
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def remove_color_outliers(points, colors, k=20, brightness_threshold=0.65,
                          color_diff_threshold=0.25):
    """
    Remove points whose color is a bright outlier relative to spatial neighbors.

    For each point brighter than `brightness_threshold`, compare its color to
    the median color of its K nearest spatial neighbors. If the Euclidean color
    distance exceeds `color_diff_threshold`, the point is a silhouette bleed
    artifact and gets removed.

    Returns:
        keep_mask: (N,) bool — True for points to keep
    """
    n = len(points)
    brightness = colors.mean(axis=1)

    # Only check bright points — dark points can't be sky bleed
    candidates = np.where(brightness > brightness_threshold)[0]
    print(f"  {len(candidates):,} bright candidates (brightness > {brightness_threshold})")

    if len(candidates) == 0:
        return np.ones(n, dtype=bool)

    # Build KD-tree and batch query neighbors for all candidates
    print(f"  Building KD-tree on {n:,} points...")
    t0 = time.time()
    tree = cKDTree(points)
    print(f"  KD-tree built in {time.time() - t0:.1f}s")

    print(f"  Querying {k} neighbors for {len(candidates):,} candidates...")
    t0 = time.time()
    _, indices = tree.query(points[candidates], k=k + 1, workers=-1)
    nn_indices = indices[:, 1:]  # exclude self
    print(f"  Query completed in {time.time() - t0:.1f}s")

    # Compare each candidate's color to its neighbors' median
    nn_colors = colors[nn_indices]  # (n_candidates, k, 3)
    nn_median = np.median(nn_colors, axis=1)  # (n_candidates, 3)
    candidate_colors = colors[candidates]
    diffs = np.linalg.norm(candidate_colors - nn_median, axis=1)

    outlier_mask = diffs > color_diff_threshold
    n_remove = outlier_mask.sum()

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[candidates[outlier_mask]] = False

    print(f"  Removing {n_remove:,} color-outlier points "
          f"({n_remove / n * 100:.1f}%)")

    return keep_mask


def main():
    parser = argparse.ArgumentParser(
        description="Remove silhouette color bleeding from a colored PLY")
    parser.add_argument("input", type=str, help="Input PLY file")
    parser.add_argument("output", type=str, nargs="?", default=None,
                        help="Output PLY file (default: <input>_clean.ply)")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of spatial neighbors to check (default: 20)")
    parser.add_argument("--brightness", type=float, default=0.65,
                        help="Brightness threshold for candidate points (0-1, default: 0.65)")
    parser.add_argument("--diff", type=float, default=0.25,
                        help="Color difference threshold for outlier detection (default: 0.25)")
    args = parser.parse_args()

    output = args.output or args.input.replace(".ply", "_clean.ply")

    print(f"Loading {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  # float64, 0-1 range
    print(f"  {len(points):,} points")

    if len(colors) == 0:
        print("ERROR: PLY has no colors")
        return 1

    keep = remove_color_outliers(
        points, colors,
        k=args.k,
        brightness_threshold=args.brightness,
        color_diff_threshold=args.diff,
    )

    pcd_clean = pcd.select_by_index(np.where(keep)[0])
    o3d.io.write_point_cloud(output, pcd_clean)
    size_mb = float(open(output, 'rb').seek(0, 2)) / 1024 / 1024
    # get actual size
    import os
    size_mb = os.path.getsize(output) / 1024 / 1024
    print(f"\nSaved: {output} ({len(pcd_clean.points):,} points, {size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
