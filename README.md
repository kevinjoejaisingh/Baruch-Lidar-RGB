# Fusion Pipeline: LiDAR + RGB-D → Colored 3D Point Cloud

A complete end-to-end system that fuses a **Livox Mid-360 LiDAR** with an **Intel RealSense D455 RGB-D camera** to produce photorealistic colored 3D point clouds of indoor environments.

The LiDAR provides dense, accurate 3D geometry (360° coverage), while the D455 provides high-resolution RGB color (forward-facing). The pipeline calibrates the sensors, aligns their trajectories, and projects camera color onto the LiDAR cloud with occlusion-aware, depth-verified multi-frame blending.

---

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌──────────────┐    ┌───────────────┐
│  1. Capture  │───▶│ 2. Process   │───▶│ 3. Process D455   │───▶│ 4. Calibrate │───▶│ 5. Project    │
│  capture.py  │    │  LiDAR       │    │  process_d455.py  │    │  Extrinsic   │    │  Color        │
│              │    │  process_    │    │                   │    │  align_      │    │  project_     │
│  LiDAR bag + │    │  lidar.py    │    │  D455 depth ICP   │    │  trajectori  │    │  color.py     │
│  D455 RGB-D  │    │              │    │  → camera poses   │    │  es.py       │    │               │
│  frames      │    │  SLAM cloud  │    │                   │    │              │    │  ICP refine + │
│              │    │  + trajectory │    │                   │    │  Grid search │    │  depth verify +│
│              │    │  + pose graph │    │                   │    │  ICP → T_    │    │  mean blend   │
│              │    │  optimization │    │                   │    │  lidar_from_ │    │               │
│              │    │              │    │                   │    │  d455        │    │  → colored    │
│              │    │              │    │                   │    │              │    │    cloud.ply  │
└─────────────┘    └──────────────┘    └───────────────────┘    └──────────────┘    └───────────────┘
```

---

## Quick Start

```bash
# 1. Capture — record LiDAR + D455 simultaneously
python capture.py my_scan

# 2. Process LiDAR — extract point cloud + trajectory from ROS2 bag
python process_lidar.py scans/my_scan

# 3. Process D455 — build camera trajectory from depth frames
python process_d455.py scans/my_scan

# 4. Calibrate — find the rigid bracket transform between sensors
python align_trajectories.py scans/my_scan

# 5. Project color — fuse RGB onto LiDAR cloud
python project_color.py scans/my_scan

# View result
meshlab scans/my_scan/colored_cloud.ply
```

---

## Theory & Algorithms

### The Core Problem

The LiDAR captures a dense 360° point cloud but has no color. The D455 captures color images but only sees forward (~90° FOV). The two sensors are rigidly mounted on a bracket but in different coordinate frames with different origins. The goal is to determine, for each LiDAR point, what color it should be by finding which pixel in which camera frame corresponds to that 3D location.

This requires solving three problems:
1. **Where was each sensor at each moment?** (trajectory estimation)
2. **What is the fixed spatial relationship between the sensors?** (extrinsic calibration)
3. **Which camera pixel corresponds to which 3D point?** (projection with occlusion handling)

### Step 1: Capture (`capture.py`)

Records both sensors simultaneously:
- **LiDAR**: Runs RKO-LIO SLAM via ROS2, recording a bag with raw scans, IMU, odometry, and keyframe clouds
- **D455**: Captures RGB + depth frames at 2 Hz with IMU data (accelerometer + gyroscope) for orientation

The D455 capture rate is deliberately low (2 Hz vs 30 Hz) because we need diverse viewpoints, not video — each frame should show the scene from a meaningfully different angle.

### Step 2: LiDAR Processing (`process_lidar.py`)

Extracts and refines the LiDAR point cloud:

1. **Parse ROS2 bag**: Extract SLAM odometry poses and keyframe point clouds
2. **Chunk-based ICP**: Group frames into chunks (~10 frames each), register consecutive chunks with Point-to-Plane ICP
3. **Loop closure**: Register the last chunk against the first to detect and correct drift accumulation over the full scan
4. **Pose graph optimization**: Build a graph where nodes are chunk poses and edges are ICP constraints. Use Levenberg-Marquardt optimization to distribute drift correction across all poses
5. **Clean up**: Voxel downsample (1cm) + statistical outlier removal

**Why pose graph optimization?** LiDAR SLAM accumulates drift over time — the end of the trajectory doesn't perfectly match the beginning. By detecting this loop closure error and distributing it across all poses, we get a globally consistent point cloud.

**Output**: `lidar_cloud.ply` (dense 3D point cloud) + `lidar_trajectory.npz` (corrected poses at each timestamp)

### Step 3: D455 Processing (`process_d455.py`)

Builds a camera trajectory by chaining frame-to-frame alignments:

1. **Depth unprojection**: Convert each depth image to a 3D point cloud using camera intrinsics: `X = (u - cx) * Z / fx`, `Y = (v - cy) * Z / fy`
2. **Consecutive ICP**: Align each frame's point cloud to the previous frame using Point-to-Plane ICP. The D455 IMU provides an initial rotation guess (from accelerometer pitch/roll)
3. **Loop closure + pose graph**: Same as LiDAR — detect drift and distribute correction

**Output**: `d455_trajectory.npz` (camera poses + intrinsics + frame numbers)

### Step 4: Extrinsic Calibration (`align_trajectories.py`)

Finds the fixed rigid transform `T_lidar_from_d455` — the rotation and translation from the D455 camera frame to the LiDAR frame, determined by the physical bracket mounting.

**The challenge**: The two sensors have completely different coordinate conventions (~90° rotation between frames) and the trajectories have different drift characteristics, so simple trajectory matching (Umeyama alignment) fails.

**Solution — Direct depth ICP with rotation grid search**:

For a given D455 frame:
1. Unproject the D455 depth image to a 3D point cloud (in camera frame)
2. Find where the LiDAR was at the same timestamp (interpolated pose)
3. Crop the LiDAR world cloud near that position and transform to LiDAR local frame
4. Search for the rotation that best aligns D455 cloud → LiDAR cloud:
   - **Coarse search**: Try 216 Euler angle combinations (60° steps), run fast Point-to-Point ICP for each
   - **Fine search**: ±25° around best coarse result (10° steps), 216 more candidates
   - **Final refinement**: Point-to-Plane ICP with tight convergence criteria
5. **Validate**: Repeat on 8+ additional frames, filter outliers by translation consistency
6. **Aggregate**: Weighted average of all good results (weighted by ICP fitness)

**Why grid search?** The ~90° rotation between coordinate frames is too large for ICP alone (it would fall into a local minimum). The grid search ensures we find the correct global rotation, then ICP refines it precisely.

**Output**: `extrinsic.json` containing the 4×4 transform matrix

### Step 5: Color Projection (`project_color.py`)

Projects RGB color from D455 frames onto the LiDAR point cloud. This is the most nuanced step.

#### Transform Chain

For each D455 frame at time `t`:
```
T_world_lidar    = interpolated LiDAR pose at time t    (LiDAR local → world)
T_lidar_from_d455 = bracket extrinsic                    (D455 camera → LiDAR local)

T_world_camera   = T_world_lidar @ T_lidar_from_d455    (D455 camera → world)
T_cam_from_world = inv(T_world_camera)                   (world → D455 camera)
```

To project a world point `P` into the camera:
```
P_cam = T_cam_from_world @ P        → point in camera frame
u = fx * P_cam.x / P_cam.z + cx     → pixel column
v = fy * P_cam.y / P_cam.z + cy     → pixel row
color = rgb_image[v, u]              → sampled color
```

#### Per-Frame ICP Pose Refinement

The interpolated pose has residual error from SLAM drift and timestamp interpolation. Before projecting color, each frame's camera pose is refined by ICP-aligning the D455 depth cloud against the LiDAR world cloud:

1. Unproject D455 depth → point cloud in camera frame
2. Transform to world using initial pose estimate
3. Crop nearby LiDAR points (within depth range + margin)
4. Run Point-to-Plane ICP to find the small correction
5. Apply correction to get refined camera pose

This eliminates ~2-5cm of residual alignment error per frame.

#### Z-Buffer Occlusion Handling

Multiple LiDAR points can project to the same pixel (e.g., a wall point behind a chair). Without filtering, the wall would incorrectly get the chair's color.

**Solution**: Build a Z-buffer — for each pixel, track the minimum depth. Only color points whose depth is within a tolerance (5cm) of the minimum. This naturally keeps foreground objects and rejects occluded background points.

#### Depth Consistency Verification

Even with ICP refinement, small pose errors can cause a LiDAR point to project to the wrong pixel. For example, a chair point might project to a pixel where the camera actually sees wall.

**Solution**: For each projected LiDAR point, compare its projected depth against the D455 measured depth at that pixel. If they disagree by more than 10cm, this point isn't what the camera sees at that pixel — reject the color assignment.

This is the key innovation that fixes:
- **Ghost/duplicate features** (e.g., posters appearing twice): SLAM drift creates duplicate point layers. Without depth verification, both layers get colored. With it, points from the "wrong" layer disagree with D455 depth and get rejected.
- **Background color bleeding** (e.g., chair head getting wall color): When a point projects to the wrong pixel due to pose error, the D455 depth at that pixel reveals the mismatch.

#### Multi-Frame Mean Blending

Each LiDAR point is visible from multiple camera frames. Instead of picking a single "best" frame (fragile — one bad pose ruins the color), we average colors across all depth-verified frames:

```
final_color[point] = mean(color from each frame that passes depth verification)
```

This is robust because:
- Depth verification removes grossly wrong observations
- Averaging across ~12 frames per point smooths out noise and minor misalignments
- No single frame can dominate (unlike "closest camera wins")

#### Height Colormap for Uncolored Points

The LiDAR is 360° but the camera is ~90° forward-facing, so ~45% of points have no camera coverage. These are colored with a muted height-based rainbow (blue at floor → green → red at ceiling) at reduced brightness so they're clearly distinguishable from real RGB color.

---

## Data Format

### Scan Directory Structure
```
scans/<scan_name>/
├── bag/                          # ROS2 bag (LiDAR raw data)
│   ├── metadata.yaml
│   └── *.db3
├── d455/                         # D455 RGB-D frames
│   ├── frame_001_rgb.png         # 1280×720 RGB
│   ├── frame_001_depth.png       # 1280×720 uint16 depth (mm)
│   ├── frame_001_imu.json        # Accelerometer, gyro, orientation
│   └── intrinsics.json           # Camera matrix (fx, fy, cx, cy)
├── lidar_cloud.ply               # Processed LiDAR point cloud
├── lidar_trajectory.npz          # LiDAR poses + timestamps
├── d455_trajectory.npz           # Camera poses + timestamps + intrinsics
├── extrinsic.json                # Bracket calibration (T_lidar_from_d455)
└── colored_cloud.ply             # Final output — colored point cloud
```

### Key File Formats

**Trajectories** (`.npz`):
- `poses`: `(N, 4, 4)` float64 — homogeneous transforms
- `timestamps_ns`: `(N,)` int64 — nanosecond epoch timestamps

**Extrinsic** (`.json`):
- `transform`: 4×4 matrix (T_lidar_from_d455)
- `translation_m`: 3-vector (bracket offset in meters)
- `rotation_matrix`: 3×3 rotation
- `method`: calibration algorithm used
- `alignment_info`: fitness scores, frames used

---

## Configuration

All parameters are in `config.yaml`:

| Section | Key Parameter | Default | Purpose |
|---------|--------------|---------|---------|
| `d455` | `max_depth_m` | 3.0 | D455 depth range limit |
| `capture` | `auto_capture_hz` | 2 | D455 frame capture rate |
| `processing` | `lidar_voxel_size` | 0.01 | LiDAR downsample resolution (m) |
| `processing` | `icp_max_distance` | 0.05 | ICP correspondence distance (m) |
| `projection` | `zbuffer_tolerance` | 0.05 | Occlusion depth tolerance (m) |
| `projection` | `max_distance_m` | 10.0 | Max point-to-camera distance |
| `projection` | `depth_consistency_threshold` | 0.10 | LiDAR-D455 depth agreement (m) |

---

## Performance

For a typical room scan (~4M LiDAR points, 112 D455 frames):

| Stage | Time |
|-------|------|
| LiDAR processing | ~1-2 min |
| D455 processing | ~30 sec |
| Extrinsic calibration | ~2-3 min |
| Color projection (ICP refinement) | ~72 sec |
| Color projection (RGB mapping) | ~47 sec |
| **Total post-processing** | **~5-7 min** |

---

## Dependencies

- **ROS2 Jazzy** + Livox driver + RKO-LIO SLAM
- **Open3D** — point cloud I/O, ICP, visualization
- **OpenCV** — image I/O, color space conversion
- **NumPy** — array operations
- **SciPy** — optimization, spatial transforms (SLERP)
- **PyYAML** — configuration
- **pyrealsense2** — Intel RealSense D455 SDK
- **rosbags** — ROS2 bag reading (without full ROS install)

---

## Hardware

- **Livox Mid-360** — 360° non-repetitive scanning LiDAR (200k pts/sec)
- **Intel RealSense D455** — RGB-D camera (1280×720, stereo depth to 6m, 9-DOF IMU)
- **Rigid bracket** — mounts both sensors with fixed ~13cm baseline, ~90° rotation between coordinate frames
