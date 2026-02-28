"""
Timestamp matching and pose interpolation utilities.
"""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def find_nearest_timestamps(query_ns, reference_ns, max_diff_ns=None):
    """
    For each query timestamp, find the nearest reference timestamp.

    Args:
        query_ns: (M,) int64 array of query timestamps in nanoseconds
        reference_ns: (N,) int64 array of reference timestamps in nanoseconds
        max_diff_ns: Maximum allowed difference in nanoseconds. If None, no limit.

    Returns:
        query_indices: indices into query_ns that have valid matches
        ref_indices: corresponding indices into reference_ns
        diffs_ns: time differences for each match
    """
    query_ns = np.asarray(query_ns, dtype=np.int64)
    reference_ns = np.asarray(reference_ns, dtype=np.int64)

    # Use searchsorted for efficient nearest-neighbor lookup
    # Find insertion point for each query in sorted reference
    sort_idx = np.argsort(reference_ns)
    ref_sorted = reference_ns[sort_idx]

    insert_pos = np.searchsorted(ref_sorted, query_ns)

    # For each query, check the element before and after insertion point
    query_indices = []
    ref_indices = []
    diffs = []

    for i, pos in enumerate(insert_pos):
        best_idx = None
        best_diff = np.iinfo(np.int64).max

        # Check position before
        if pos > 0:
            diff = abs(query_ns[i] - ref_sorted[pos - 1])
            if diff < best_diff:
                best_diff = diff
                best_idx = sort_idx[pos - 1]

        # Check position at
        if pos < len(ref_sorted):
            diff = abs(query_ns[i] - ref_sorted[pos])
            if diff < best_diff:
                best_diff = diff
                best_idx = sort_idx[pos]

        if best_idx is not None:
            if max_diff_ns is None or best_diff <= max_diff_ns:
                query_indices.append(i)
                ref_indices.append(best_idx)
                diffs.append(best_diff)

    return (
        np.array(query_indices, dtype=np.int64),
        np.array(ref_indices, dtype=np.int64),
        np.array(diffs, dtype=np.int64),
    )


def interpolate_pose_at_timestamp(target_ns, timestamps_ns, poses):
    """
    Interpolate a 4x4 pose at target_ns using SLERP (rotation) + linear (translation).

    Args:
        target_ns: int64 target timestamp
        timestamps_ns: (N,) int64 sorted timestamps
        poses: (N, 4, 4) float64 pose matrices

    Returns:
        (4, 4) interpolated pose matrix
    """
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    n = len(timestamps_ns)

    # Clamp to bounds
    if target_ns <= timestamps_ns[0]:
        return poses[0].copy()
    if target_ns >= timestamps_ns[-1]:
        return poses[-1].copy()

    # Find bracketing indices
    idx = np.searchsorted(timestamps_ns, target_ns) - 1
    idx = np.clip(idx, 0, n - 2)

    t0 = timestamps_ns[idx]
    t1 = timestamps_ns[idx + 1]

    if t1 == t0:
        return poses[idx].copy()

    alpha = float(target_ns - t0) / float(t1 - t0)

    # Linear interpolation for translation
    trans0 = poses[idx][:3, 3]
    trans1 = poses[idx + 1][:3, 3]
    trans_interp = (1 - alpha) * trans0 + alpha * trans1

    # SLERP for rotation
    rot0 = Rotation.from_matrix(poses[idx][:3, :3])
    rot1 = Rotation.from_matrix(poses[idx + 1][:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([rot0, rot1]))
    rot_interp = slerp(alpha)

    pose = np.eye(4)
    pose[:3, :3] = rot_interp.as_matrix()
    pose[:3, 3] = trans_interp
    return pose
