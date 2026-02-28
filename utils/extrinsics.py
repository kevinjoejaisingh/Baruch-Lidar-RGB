"""
Extrinsic calibration utilities: Umeyama alignment + RANSAC.
"""

import numpy as np


def umeyama_alignment(source_pts, target_pts, with_scale=False):
    """
    Umeyama alignment: find rigid transform (R, t) minimizing ||target - (R @ source + t)||.

    SVD-based closed-form solution. No scale by default.

    Args:
        source_pts: (N, 3) points in source frame
        target_pts: (N, 3) corresponding points in target frame
        with_scale: If True, also estimate scale factor

    Returns:
        T: (4, 4) transformation matrix (target = T @ source in homogeneous coords)
        scale: float scale factor (1.0 if with_scale=False)
    """
    assert source_pts.shape == target_pts.shape
    n = source_pts.shape[0]

    # Centroids
    mu_s = source_pts.mean(axis=0)
    mu_t = target_pts.mean(axis=0)

    # Centered
    s_centered = source_pts - mu_s
    t_centered = target_pts - mu_t

    # Cross-covariance
    H = s_centered.T @ t_centered / n

    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Ensure proper rotation (det = +1)
    d = np.linalg.det(V @ U.T)
    D = np.diag([1, 1, np.sign(d)])

    R = V @ D @ U.T

    if with_scale:
        var_s = np.sum(s_centered**2) / n
        scale = np.trace(np.diag(S) @ D) / var_s
    else:
        scale = 1.0

    t = mu_t - scale * R @ mu_s

    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, scale


def umeyama_with_ransac(source_pts, target_pts, threshold=0.05, max_iterations=1000,
                        min_inliers=10, with_scale=False, rng=None):
    """
    RANSAC wrapper for Umeyama alignment to handle outliers.

    Args:
        source_pts: (N, 3) source points
        target_pts: (N, 3) target points
        threshold: Inlier distance threshold in meters
        max_iterations: Maximum RANSAC iterations
        min_inliers: Minimum inliers for a valid model
        with_scale: Estimate scale factor
        rng: numpy random generator

    Returns:
        T_best: (4, 4) best transformation
        inlier_mask: (N,) boolean mask of inliers
        info: dict with diagnostics
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(source_pts)
    min_samples = 3  # Minimum for rigid transform

    best_T = np.eye(4)
    best_inliers = np.zeros(n, dtype=bool)
    best_n_inliers = 0

    for iteration in range(max_iterations):
        # Sample min_samples random correspondences
        indices = rng.choice(n, min_samples, replace=False)
        s_sample = source_pts[indices]
        t_sample = target_pts[indices]

        # Fit model
        try:
            T, _ = umeyama_alignment(s_sample, t_sample, with_scale=with_scale)
        except np.linalg.LinAlgError:
            continue

        # Evaluate on all points
        source_hom = np.column_stack([source_pts, np.ones(n)])
        transformed = (T @ source_hom.T).T[:, :3]
        residuals = np.linalg.norm(transformed - target_pts, axis=1)
        inlier_mask = residuals < threshold

        n_inliers = np.sum(inlier_mask)
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inliers = inlier_mask.copy()

            # Refit on all inliers
            if n_inliers >= min_samples:
                T_refit, _ = umeyama_alignment(
                    source_pts[inlier_mask], target_pts[inlier_mask], with_scale=with_scale
                )
                best_T = T_refit

        # Early exit if we have enough inliers
        if n_inliers > 0.8 * n:
            break

    info = {
        "n_inliers": int(best_n_inliers),
        "n_total": n,
        "inlier_ratio": float(best_n_inliers / n) if n > 0 else 0.0,
        "iterations_used": iteration + 1,
    }

    if best_n_inliers < min_inliers:
        info["warning"] = f"Only {best_n_inliers} inliers (min={min_inliers})"

    return best_T, best_inliers, info
