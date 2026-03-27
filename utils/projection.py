"""
Camera projection and Z-buffer utilities for coloring LiDAR points from RGB images.
"""

import cv2
import numpy as np


def project_points_to_image(points_world, T_cam_from_world, fx, fy, cx, cy, image_h, image_w):
    """
    Project 3D world points into camera image coordinates.

    Args:
        points_world: (N, 3) points in world frame
        T_cam_from_world: (4, 4) camera-from-world transform
        fx, fy, cx, cy: camera intrinsics
        image_h, image_w: image dimensions

    Returns:
        pixel_coords: (M, 2) int pixel coordinates (u, v)
        point_indices: (M,) indices into original points array
        depths: (M,) depth values in camera frame
    """
    n = len(points_world)
    if n == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0)

    # Transform to camera frame
    ones = np.ones((n, 1), dtype=points_world.dtype)
    pts_hom = np.hstack([points_world, ones])
    pts_cam = (T_cam_from_world @ pts_hom.T).T[:, :3]  # (N, 3)

    # Filter points behind camera (z <= 0)
    z = pts_cam[:, 2]
    front_mask = z > 0
    if not np.any(front_mask):
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0)

    indices_front = np.where(front_mask)[0]
    pts_front = pts_cam[front_mask]

    # Project to pixel coordinates
    u = (fx * pts_front[:, 0] / pts_front[:, 2] + cx)
    v = (fy * pts_front[:, 1] / pts_front[:, 2] + cy)

    # Filter to image bounds
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)
    bounds_mask = (u_int >= 0) & (u_int < image_w) & (v_int >= 0) & (v_int < image_h)

    if not np.any(bounds_mask):
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int64), np.empty(0)

    pixel_coords = np.column_stack([u_int[bounds_mask], v_int[bounds_mask]])
    point_indices = indices_front[bounds_mask]
    depths = pts_front[bounds_mask, 2]

    return pixel_coords, point_indices, depths


def zbuffer_filter(pixel_coords, depths, point_indices, image_h, image_w, tolerance=0.05):
    """
    Z-buffer filtering: for each pixel, keep only the closest point.

    Args:
        pixel_coords: (M, 2) pixel coordinates (u, v)
        depths: (M,) depth values
        point_indices: (M,) original point indices
        image_h, image_w: image dimensions
        tolerance: depth tolerance — points within this of the closest are kept

    Returns:
        surviving_indices: indices into the original points array that survive Z-buffer
    """
    if len(pixel_coords) == 0:
        return np.empty(0, dtype=np.int64)

    # Build Z-buffer using np.minimum.at
    zbuffer = np.full((image_h, image_w), np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, (pixel_coords[:, 1], pixel_coords[:, 0]), depths)

    # Keep points whose depth is within tolerance of the Z-buffer minimum
    min_depths = zbuffer[pixel_coords[:, 1], pixel_coords[:, 0]]
    surviving_mask = depths <= (min_depths + tolerance)

    return point_indices[surviving_mask]


def depth_consistency_filter(pixel_coords, depths, depth_image, threshold=0.10):
    """
    Check if projected LiDAR depths agree with D455 depth image.

    For each projected point, compare its depth to the D455 measured depth
    at that pixel. Points where they disagree are occluded or misaligned.

    Returns:
        consistent_mask: (M,) bool — True for points that pass
    """
    if depth_image is None:
        return np.ones(len(pixel_coords), dtype=bool)

    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]

    # Read D455 depth at projected pixels (mm → m)
    d455_depth = depth_image[v, u].astype(np.float64) / 1000.0

    # Where D455 has valid depth, check agreement
    valid_d455 = d455_depth > 0.1
    consistent = ~valid_d455 | (np.abs(depths - d455_depth) < threshold)

    return consistent


def depth_edge_erosion_filter(pixel_coords, depth_image, erosion_px=2,
                               edge_threshold_mm=300):
    """
    Reject points projecting near depth discontinuities in the camera image.

    At object silhouettes (e.g., tree trunk against sky), LiDAR foreground
    points can project onto background pixels due to the sensor baseline
    parallax or small pose errors, picking up incorrect colors (e.g., white
    sky bleeding onto tree edges).

    Detects sharp depth transitions via Sobel gradient, dilates the edge
    zone, and rejects any LiDAR point projecting into that exclusion zone.

    Args:
        pixel_coords: (M, 2) int pixel coordinates (u, v)
        depth_image: (H, W) uint16 depth in mm
        erosion_px: radius of exclusion zone around depth edges (pixels)
        edge_threshold_mm: Sobel gradient magnitude threshold for edge detection

    Returns:
        mask: (M,) bool — True for points that pass (not near edges)
    """
    if depth_image is None or erosion_px <= 0:
        return np.ones(len(pixel_coords), dtype=bool)

    depth_f = depth_image.astype(np.float32)

    # Sobel gradient to find depth discontinuities
    grad_x = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Mark pixels with large depth jumps
    edge_mask = (grad_mag > edge_threshold_mm).astype(np.uint8)

    # Dilate to create exclusion zone around edges
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erosion_px + 1, 2 * erosion_px + 1)
    )
    edge_mask = cv2.dilate(edge_mask, kernel)

    # Reject points in the exclusion zone
    u = pixel_coords[:, 0]
    v = pixel_coords[:, 1]
    return edge_mask[v, u] == 0


def color_points_from_frame(points_world, T_cam_from_world, fx, fy, cx, cy,
                            rgb_image, zbuffer_tolerance=0.05,
                            depth_image=None, depth_threshold=0.10,
                            erosion_px=2, edge_threshold_mm=300):
    """
    Color world points using a single camera frame.

    Args:
        points_world: (N, 3) world points
        T_cam_from_world: (4, 4) camera-from-world transform
        fx, fy, cx, cy: intrinsics
        rgb_image: (H, W, 3) uint8 RGB image
        zbuffer_tolerance: Z-buffer tolerance in meters
        depth_image: optional (H, W) uint16 depth in mm for consistency check
        depth_threshold: max depth disagreement (meters) before rejecting
        erosion_px: pixel radius to erode around depth edges (0 to disable)
        edge_threshold_mm: Sobel gradient threshold for depth edge detection (mm)

    Returns:
        colored_indices: (K,) indices into points_world that got colored
        colors: (K, 3) uint8 RGB values
        depths: (K,) distances from camera
    """
    h, w = rgb_image.shape[:2]

    pixel_coords, point_indices, depths = project_points_to_image(
        points_world, T_cam_from_world, fx, fy, cx, cy, h, w
    )

    if len(pixel_coords) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.uint8), np.empty(0)

    # Z-buffer filter
    surviving = zbuffer_filter(pixel_coords, depths, point_indices, h, w, zbuffer_tolerance)

    if len(surviving) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.uint8), np.empty(0)

    # Re-project surviving points to get pixel coords
    surviving_set = set(surviving)
    mask = np.array([idx in surviving_set for idx in point_indices])
    surv_pixels = pixel_coords[mask]
    surv_depths = depths[mask]
    surv_indices = point_indices[mask]

    # Depth consistency filter — reject points where D455 depth disagrees
    if depth_image is not None:
        dc_mask = depth_consistency_filter(
            surv_pixels, surv_depths, depth_image, depth_threshold
        )
        surv_pixels = surv_pixels[dc_mask]
        surv_depths = surv_depths[dc_mask]
        surv_indices = surv_indices[dc_mask]

    # Depth edge erosion — reject points near silhouette boundaries
    if depth_image is not None and erosion_px > 0:
        edge_mask = depth_edge_erosion_filter(
            surv_pixels, depth_image, erosion_px, edge_threshold_mm
        )
        surv_pixels = surv_pixels[edge_mask]
        surv_depths = surv_depths[edge_mask]
        surv_indices = surv_indices[edge_mask]

    if len(surv_indices) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.uint8), np.empty(0)

    # Sample colors from image
    colors = rgb_image[surv_pixels[:, 1], surv_pixels[:, 0]]

    return surv_indices, colors, surv_depths, surv_pixels


def color_points_multi_frame(points_world, frames, zbuffer_tolerance=0.05,
                             max_distance=10.0, default_color=(128, 128, 128),
                             depth_threshold=0.10, erosion_px=2,
                             edge_threshold_mm=300):
    """
    Color points from multiple camera frames using mean blending with depth verification.

    For each point, colors from all depth-verified frames are averaged.
    This is more robust than "closest camera wins" because it rejects
    misaligned observations via depth consistency and averages out noise.

    Args:
        points_world: (N, 3) world points
        frames: list of dicts with keys:
            - T_cam_from_world: (4, 4)
            - rgb_image: (H, W, 3) uint8
            - depth_image: optional (H, W) uint16 depth in mm
            - fx, fy, cx, cy: intrinsics
        zbuffer_tolerance: Z-buffer tolerance
        max_distance: Maximum point-to-camera distance
        default_color: RGB color for uncolored points
        depth_threshold: Depth consistency threshold (meters)

    Returns:
        colors: (N, 3) uint8 RGB array
        stats: dict with coverage statistics
    """
    n = len(points_world)
    color_sum = np.zeros((n, 3), dtype=np.float64)
    color_count = np.zeros(n, dtype=np.int32)

    for i, frame in enumerate(frames):
        indices, colors, depths, pixels = color_points_from_frame(
            points_world,
            frame["T_cam_from_world"],
            frame["fx"], frame["fy"], frame["cx"], frame["cy"],
            frame["rgb_image"],
            zbuffer_tolerance,
            depth_image=frame.get("depth_image"),
            depth_threshold=depth_threshold,
            erosion_px=erosion_px,
            edge_threshold_mm=edge_threshold_mm,
        )

        if len(indices) == 0:
            continue

        # Filter by max distance
        dist_mask = depths < max_distance
        indices = indices[dist_mask]
        colors = colors[dist_mask]

        # Accumulate colors for mean blending
        np.add.at(color_sum, (indices, slice(None)), colors.astype(np.float64))
        np.add.at(color_count, indices, 1)

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            pct = np.sum(color_count > 0) / n * 100
            print(f"  Frame {i + 1}/{len(frames)}: {pct:.1f}% colored")

    # Compute mean colors
    colored_mask = color_count > 0
    best_colors = np.full((n, 3), default_color, dtype=np.uint8)
    if np.any(colored_mask):
        mean_colors = color_sum[colored_mask] / color_count[colored_mask, np.newaxis]
        best_colors[colored_mask] = np.clip(mean_colors, 0, 255).astype(np.uint8)

    stats = {
        "total_points": n,
        "colored_points": int(np.sum(colored_mask)),
        "coverage_pct": float(np.sum(colored_mask) / n * 100),
        "avg_frames_per_colored_point": float(
            np.mean(color_count[colored_mask]) if np.any(colored_mask) else 0
        ),
    }

    return best_colors, stats


def color_points_closest_camera(points_world, frames, zbuffer_tolerance=0.05,
                                max_distance=10.0, default_color=(128, 128, 128),
                                depth_threshold=0.10, erosion_px=2,
                                edge_threshold_mm=300):
    """
    Color points using 'closest camera wins' — each point keeps the color
    from whichever frame had the smallest depth (camera was closest).

    No averaging. This avoids the blur caused by mean-blending many frames
    with small pose errors.
    """
    n = len(points_world)
    best_colors = np.full((n, 3), default_color, dtype=np.uint8)
    best_depth = np.full(n, np.inf, dtype=np.float64)
    color_count = np.zeros(n, dtype=np.int32)

    for i, frame in enumerate(frames):
        indices, colors, depths, pixels = color_points_from_frame(
            points_world,
            frame["T_cam_from_world"],
            frame["fx"], frame["fy"], frame["cx"], frame["cy"],
            frame["rgb_image"],
            zbuffer_tolerance,
            depth_image=frame.get("depth_image"),
            depth_threshold=depth_threshold,
            erosion_px=erosion_px,
            edge_threshold_mm=edge_threshold_mm,
        )

        if len(indices) == 0:
            continue

        # Filter by max distance
        dist_mask = depths < max_distance
        indices = indices[dist_mask]
        colors = colors[dist_mask]
        depths = depths[dist_mask]

        # Only update points where this frame's camera is closer
        closer_mask = depths < best_depth[indices]
        update_idx = indices[closer_mask]
        best_colors[update_idx] = colors[closer_mask]
        best_depth[update_idx] = depths[closer_mask]
        color_count[update_idx] = 1  # mark as colored

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            pct = np.sum(color_count > 0) / n * 100
            print(f"  Frame {i + 1}/{len(frames)}: {pct:.1f}% colored")

    colored_mask = color_count > 0
    stats = {
        "total_points": n,
        "colored_points": int(np.sum(colored_mask)),
        "coverage_pct": float(np.sum(colored_mask) / n * 100),
        "avg_frames_per_colored_point": 1.0,
    }

    return best_colors, stats


def color_points_first_wins(points_world, frames, zbuffer_tolerance=0.05,
                            max_distance=10.0, default_color=(128, 128, 128),
                            depth_threshold=0.10, erosion_px=2,
                            edge_threshold_mm=300):
    """
    Color points using 'first valid wins' — once a point is colored by any
    frame, no subsequent frame can override it.
    """
    n = len(points_world)
    best_colors = np.full((n, 3), default_color, dtype=np.uint8)
    colored = np.zeros(n, dtype=bool)

    for i, frame in enumerate(frames):
        indices, colors, depths, pixels = color_points_from_frame(
            points_world,
            frame["T_cam_from_world"],
            frame["fx"], frame["fy"], frame["cx"], frame["cy"],
            frame["rgb_image"],
            zbuffer_tolerance,
            depth_image=frame.get("depth_image"),
            depth_threshold=depth_threshold,
            erosion_px=erosion_px,
            edge_threshold_mm=edge_threshold_mm,
        )

        if len(indices) == 0:
            continue

        # Filter by max distance
        dist_mask = depths < max_distance
        indices = indices[dist_mask]
        colors = colors[dist_mask]

        # Only color points that haven't been colored yet
        new_mask = ~colored[indices]
        new_idx = indices[new_mask]
        best_colors[new_idx] = colors[new_mask]
        colored[new_idx] = True

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            pct = np.sum(colored) / n * 100
            print(f"  Frame {i + 1}/{len(frames)}: {pct:.1f}% colored")

    stats = {
        "total_points": n,
        "colored_points": int(np.sum(colored)),
        "coverage_pct": float(np.sum(colored) / n * 100),
        "avg_frames_per_colored_point": 1.0,
    }

    return best_colors, stats


def _center_weight(pixels, cx, cy, hw, hh):
    """
    Compute per-pixel weight based on distance from image center.
    Points at the center get weight ~1.0, points at the edge get ~0.0.
    Uses a cosine falloff for smooth transitions.
    """
    dx = (pixels[:, 0] - cx) / hw
    dy = (pixels[:, 1] - cy) / hh
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.clip(dist, 0, 1)
    weight = 0.5 * (1.0 + np.cos(np.pi * dist))
    return weight


def color_points_center_weighted(points_world, frames, zbuffer_tolerance=0.05,
                                  max_distance=10.0, default_color=(128, 128, 128),
                                  depth_threshold=0.10, erosion_px=2,
                                  edge_threshold_mm=300):
    """
    Color points using center-weighted blending.

    Each frame contributes color weighted by how close the point projects
    to the image center. This creates smooth transitions in overlap zones
    (no jagged first-wins boundaries) without the blur of uniform averaging.
    """
    n = len(points_world)
    color_sum = np.zeros((n, 3), dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)

    for i, frame in enumerate(frames):
        indices, colors, depths, pixels = color_points_from_frame(
            points_world,
            frame["T_cam_from_world"],
            frame["fx"], frame["fy"], frame["cx"], frame["cy"],
            frame["rgb_image"],
            zbuffer_tolerance,
            depth_image=frame.get("depth_image"),
            depth_threshold=depth_threshold,
            erosion_px=erosion_px,
            edge_threshold_mm=edge_threshold_mm,
        )

        if len(indices) == 0:
            continue

        # Filter by max distance
        dist_mask = depths < max_distance
        indices = indices[dist_mask]
        colors = colors[dist_mask]
        pixels = pixels[dist_mask]

        # Compute center-distance weight for each point
        h, w = frame["rgb_image"].shape[:2]
        weights = _center_weight(pixels, frame["cx"], frame["cy"], w / 2, h / 2)

        # Accumulate weighted colors
        np.add.at(color_sum, (indices, slice(None)),
                  colors.astype(np.float64) * weights[:, np.newaxis])
        np.add.at(weight_sum, indices, weights)

        if (i + 1) % 20 == 0 or i == len(frames) - 1:
            pct = np.sum(weight_sum > 0) / n * 100
            print(f"  Frame {i + 1}/{len(frames)}: {pct:.1f}% colored")

    # Compute weighted average
    colored_mask = weight_sum > 0
    best_colors = np.full((n, 3), default_color, dtype=np.uint8)
    if np.any(colored_mask):
        avg = color_sum[colored_mask] / weight_sum[colored_mask, np.newaxis]
        best_colors[colored_mask] = np.clip(avg, 0, 255).astype(np.uint8)

    stats = {
        "total_points": n,
        "colored_points": int(np.sum(colored_mask)),
        "coverage_pct": float(np.sum(colored_mask) / n * 100),
        "avg_frames_per_colored_point": float(
            np.mean(weight_sum[colored_mask]) if np.any(colored_mask) else 0
        ),
    }

    return best_colors, stats
