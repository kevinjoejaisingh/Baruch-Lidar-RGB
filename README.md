# Fusion Pipeline: LiDAR + RGB-D → Colored 3D Point Cloud

A complete end-to-end system that fuses a **Livox Mid-360 LiDAR** with an **Intel RealSense D455 RGB-D camera** to produce photorealistic colored 3D point clouds of indoor environments.

The LiDAR provides dense, accurate 3D geometry (360° coverage), while the D455 provides high-resolution RGB color (forward-facing). The pipeline calibrates the sensors, aligns their trajectories, and projects camera color onto the LiDAR cloud with occlusion-aware, depth-verified multi-frame blending.

---

## Architecture Overview

```
                              ┌──────────────────────────────────────────────┐
                              │           CAPTURE STAGE                      │
                              │                                              │
                              │  capture.py                                  │
                              │  ┌──────────┐      ┌──────────────┐          │
                              │  │ Livox    │      │ Intel D455   │          │
                              │  │ Mid-360  │      │ RGB-D + IMU  │          │
                              │  │ LiDAR    │      │              │          │
                              │  └────┬─────┘      └──────┬───────┘          │
                              │       │                    │                  │
                              │  ROS2 bag             D455 frames             │
                              │  (SLAM odom +         (RGB + depth +         │
                              │   keyframes)           IMU JSON)              │
                              └───────┬────────────────────┬─────────────────┘
                                      │                    │
                    ┌─────────────────▼──┐          ┌─────▼──────────────┐
                    │  LIDAR PROCESSING   │          │  EXTRINSIC         │
                    │  process_lidar.py   │          │  CALIBRATION       │
                    │                     │          │  calibrate_        │
                    │  1. Parse ROS2 bag  │          │  extrinsic.py      │
                    │  2. Chunk-based ICP │          │                    │
                    │  3. Loop closure    │          │  Grid search ICP   │
                    │  4. Pose graph opt  │          │  D455 depth ↔      │
                    │  5. Voxel downsamp  │          │  LiDAR local crop  │
                    │                     │          │                    │
                    │  → lidar_cloud.ply  │          │  → extrinsic.json  │
                    │  → lidar_traj.npz   │          │  (T_lidar_from_    │
                    └────────┬────────────┘          │   d455)            │
                             │                       └──────┬─────────────┘
                             │                              │
                    ┌────────▼──────────────────────────────▼──────────────┐
                    │              COLOR PROJECTION                         │
                    │              project_color.py                         │
                    │                                                       │
                    │  For each D455 frame:                                 │
                    │  1. Interpolate LiDAR pose at D455 timestamp          │
                    │  2. Compose T_world_cam = T_world_lidar × T_extr     │
                    │  3. Optional ICP pose refinement                      │
                    │  4. Project LiDAR points → camera pixels              │
                    │  5. Z-buffer occlusion filtering                      │
                    │  6. Depth consistency verification                    │
                    │  7. Depth edge erosion                                │
                    │  8. Multi-frame blending (mean/center-weighted/       │
                    │     closest/first-wins)                               │
                    │                                                       │
                    │  → colored_cloud.ply                                  │
                    └────────┬──────────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
     ┌────────▼───┐  ┌───────▼──────┐  ┌────────▼──────────┐
     │ POST-PROC  │  │ VIEWERS      │  │ EXPORT             │
     │            │  │              │  │                    │
     │ postproc-  │  │ walk_viewer  │  │ convert_to_3dgs.py │
     │ ess_ply.py │  │ .py          │  │ (nerfstudio fmt)   │
     │            │  │              │  │                    │
     │ clean_     │  │ Godot viewer │  │ convert_ply_for_   │
     │ colors.py  │  │ (main.gd)   │  │ godot.py           │
     │            │  │              │  │                    │
     │ tsdf_      │  │ view_        │  │                    │
     │ fusion.py  │  │ gaussian.py  │  │                    │
     └────────────┘  └──────────────┘  └────────────────────┘
```

---

## Quick Start

```bash
# 1. Capture — record LiDAR + D455 simultaneously
python capture.py my_scan

# 2. Process LiDAR — extract point cloud + trajectory from ROS2 bag
python process_lidar.py scans/my_scan

# 3. Calibrate — find the rigid bracket transform between sensors
python calibrate_extrinsic.py scans/my_scan

# 4. Project color — fuse RGB onto LiDAR cloud
python project_color.py scans/my_scan

# View result
python walk_viewer.py scans/my_scan/colored_cloud.ply
# or open in Godot viewer, MeshLab, CloudCompare, etc.
```

---

## Detailed Algorithm Descriptions

### Stage 1: Simultaneous Capture (`capture.py`)

Records both sensors simultaneously with synchronized timestamps.

**LiDAR pipeline:**
1. Configures network interface for Livox Mid-360 Ethernet communication (192.168.1.x subnet)
2. Launches the Livox ROS2 driver (`livox_ros_driver2`), which streams raw scan data at ~200k points/sec
3. Starts **RKO-LIO SLAM** (ROS2 SLAM package), which consumes the raw scans and built-in IMU and publishes:
   - `/rko_lio/odometry` — real-time 6-DOF pose estimates
   - `/rko_lio/frame` — undistorted keyframe point clouds in the LiDAR local frame
4. Records these ROS2 topics into a bag file (`.db3` format) for offline replay

**D455 pipeline:**
1. Initializes Intel RealSense D455 via `pyrealsense2` SDK with streams:
   - Color: 1280×720 RGB at 30 fps
   - Depth: 1280×720 stereo depth at 30 fps (aligned to color)
   - IMU: 200 Hz accelerometer + 200 Hz gyroscope
2. Runs auto-exposure warmup (30 frames discarded)
3. Auto-captures at 2 Hz (not 30 Hz — we need diverse viewpoints, not video)
4. For each captured frame saves:
   - `frame_NNN_rgb.png` — full-resolution RGB
   - `frame_NNN_depth.png` — 16-bit depth in millimeters
   - `frame_NNN_imu.json` — accelerometer, gyroscope, computed pitch/roll, and `capture_timestamp_ns` (the D455 hardware frame timestamp converted to nanoseconds, used later for LiDAR-D455 time synchronization)
5. Saves camera intrinsics (`intrinsics.json`): focal lengths `fx`, `fy`, principal point `cx`, `cy`, and distortion coefficients

**Live preview:**
- Displays a 2×2 grid: first captured frame (depth + RGB) on top, live feed (depth + RGB) on bottom
- The live depth panel includes a **bubble level** overlay computed from the D455 accelerometer, showing current pitch/roll vs. the reference (first frame) orientation — helps the operator return to the starting pose for loop closure

**Loop closure frame:**
- When the operator presses ENTER to stop, a final "loop closure" frame is captured and flagged in its IMU JSON (`loop_closure: true`)

---

### Stage 2: LiDAR Processing (`process_lidar.py`)

Extracts a drift-corrected 3D point cloud and per-frame trajectory from the recorded ROS2 bag.

#### 2.1 ROS2 Bag Parsing

Reads the bag using the `rosbags` library (no full ROS install needed):
- **Odometry messages** (`nav_msgs/Odometry`): Extracts position (x, y, z) and orientation (quaternion → 3×3 rotation matrix via `scipy.spatial.transform.Rotation`), assembles into 4×4 homogeneous pose matrices
- **Frame messages** (`sensor_msgs/PointCloud2`): Parsed using a **vectorized structured-array approach** — the binary PointCloud2 format has variable field layouts (offsets, types, padding), so the code dynamically builds a NumPy `dtype` from the message's field descriptors, then `np.frombuffer` decodes the entire buffer in one call (no Python-level per-point loop)

#### 2.2 Frame-to-Odometry Matching

Each LiDAR keyframe is matched to its nearest odometry message by timestamp using `np.argmin(|odom_times - frame_time|)`. The matched odometry provides the 4×4 world-frame pose for that frame.

**Pre-filtering per frame:**
- **Minimum range filter**: Removes points closer than 0.5m (sensor noise near the device)
- **FOV filter** (optional): Transforms each point into the D455 camera frame using the extrinsic calibration, then keeps only points within the D455's field of view (87° horizontal × 58° vertical). This is done by checking `|x/z| ≤ tan(FOV_h/2)` and `|y/z| ≤ tan(FOV_v/2)` in camera coordinates. Purpose: discards LiDAR points that no camera frame will ever see, reducing cloud size and preventing uncolorable points

#### 2.3 Chunk-Based ICP Registration

Frames are grouped into chunks of ~10 frames. For each chunk, all points are transformed to world coordinates using their matched odometry poses and combined into a single chunk point cloud.

**Point-to-Plane ICP** aligns consecutive chunks:
1. Both source and target are voxel-downsampled (default 5cm) for speed
2. Surface normals are estimated using hybrid KD-tree search (radius = 2× voxel size, max 30 neighbors)
3. Open3D's `registration_icp` with `TransformationEstimationPointToPlane` finds the rigid transform minimizing point-to-tangent-plane distances
4. The information matrix (inverse covariance of the alignment) is also computed for use in pose graph optimization

**Why Point-to-Plane over Point-to-Point?** Point-to-Plane ICP converges faster and handles planar surfaces (walls, floors) correctly — it allows points to slide along planes where alignment is ambiguous rather than locking them to arbitrary correspondences.

#### 2.4 Loop Closure

If there are >2 chunks, the **last chunk is registered against the first** with a relaxed distance threshold (3× multiplier instead of 1.5×). This detects how much drift accumulated over the entire scan and creates an additional constraint connecting the end of the trajectory back to the beginning.

#### 2.5 Pose Graph Optimization

An Open3D `PoseGraph` is constructed:
- **Nodes**: One per chunk, initialized to identity (corrections relative to raw poses)
- **Edges**: ICP transforms between consecutive chunks + the loop closure edge (marked `uncertain=True`)
- Edges with ICP fitness below threshold (0.1 for sequential, 0.05 for loop) are dropped

**Levenberg-Marquardt optimization** (`GlobalOptimizationLevenbergMarquardt`) distributes the loop closure error across all nodes. This finds the set of per-chunk correction transforms that minimizes the total weighted error across all ICP edges.

**Key parameters:**
- `max_correspondence_distance`: 1.5× ICP voxel size
- `edge_prune_threshold`: 0.25
- `preference_loop_closure`: 2.0 (loop edges weighted 2× higher than sequential)
- `reference_node`: 0 (first chunk stays fixed)

#### 2.6 Applying Corrections

For each frame in each chunk:
1. Points are transformed to world using raw odometry: `P_world = R × P_local + t`
2. The chunk's correction transform (from pose graph) is applied: `P_corrected = T_correction × P_world`
3. Per-frame poses are also corrected: `T_corrected = T_correction × T_raw`

#### 2.7 Cleanup

- **Voxel downsampling** at 3mm resolution (configurable): Each voxel keeps one representative point, reducing density uniformly
- **Statistical outlier removal**: For each point, computes mean distance to its K nearest neighbors (K=50). Points whose mean distance exceeds 1.5 standard deviations from the global average are removed (eliminates noise spikes and isolated phantom points)

**Output:** `lidar_cloud.ply` (dense point cloud, typically 4-10M points) + `lidar_trajectory.npz` (corrected 4×4 poses at each timestamp)

---

### Stage 3: Extrinsic Calibration (`calibrate_extrinsic.py`)

Finds the fixed rigid transform `T_lidar_from_d455` — the 4×4 matrix that transforms a point from the D455 camera coordinate frame to the LiDAR local coordinate frame. This encodes the physical bracket mounting: ~13cm baseline, ~90° rotation between coordinate conventions.

#### 3.1 The Challenge

The two sensors have completely different coordinate frames (~60° rotation due to the tilted bracket) and neither ICP alone nor trajectory-matching algorithms (like Umeyama) can handle this large initial misalignment — ICP falls into local minima and trajectory matching fails because drift characteristics differ.

#### 3.2 Per-Frame Data Preparation

For a chosen D455 frame:
1. **D455 depth unprojection**: The 16-bit depth image (mm) is converted to a 3D point cloud in camera frame using pinhole projection:
   ```
   X = (u - cx) × Z / fx
   Y = (v - cy) × Z / fy
   Z = depth_mm / 1000
   ```
   Points with depth < 0.1m or > 3.0m (max depth) are discarded.

2. **LiDAR local crop**: The D455 frame's timestamp is used to interpolate a LiDAR pose (via SLERP rotation + linear translation). The LiDAR world cloud is cropped to points within `max_depth + 0.5m` of the interpolated rig position, then transformed to the LiDAR's local frame at that pose: `P_local = T_world_lidar⁻¹ × P_world`

#### 3.3 Rotation Grid Search + ICP

**Coarse search (60° steps):**
- Tests 6³ = 216 Euler angle combinations (XYZ convention, 0° to 300° in 60° increments)
- For each candidate rotation, runs fast **Point-to-Point ICP** (15 iterations) with 4× voxel size correspondence distance
- Keeps the transform with highest ICP fitness score, rejecting any with translation > 20cm (bracket can't be that large)

**Fine search (10° steps around best):**
- Takes the best coarse rotation and searches ±25° around each axis (10° steps = 6³ = 216 more candidates)
- Uses 3× voxel size correspondence distance and 30 ICP iterations
- Same fitness-maximizing selection with translation sanity check

**Final refinement:**
- **Point-to-Plane ICP** with tight convergence (`relative_fitness=1e-7`, `relative_rmse=1e-7`, 200 max iterations)
- Uses the fine-search result as initialization
- This stage typically improves fitness by 5-15%

#### 3.4 Multi-Frame Validation

The calibration is validated on 8 additional frames spread across the scan:
- Each frame gets its own LiDAR local crop and runs Point-to-Plane ICP initialized with the primary frame's result
- Results with translation more than 10cm from the median are rejected as outliers
- The final transform is a **fitness-weighted average**: translations are linearly averaged, rotations are averaged as quaternions (with hemisphere alignment to avoid the double-cover issue)

**Output:** `extrinsic.json` containing the 4×4 transform, translation vector, rotation matrix, and alignment quality metrics

---

### Stage 4: Color Projection (`project_color.py`)

Projects RGB color from D455 camera frames onto the LiDAR point cloud. This is the most algorithmically nuanced stage, involving six distinct filtering/blending steps.

#### 4.1 Transform Chain

For each D455 frame captured at time `t`:

```
T_world_lidar     = interpolated LiDAR SLAM pose at time t     (LiDAR local → world)
T_lidar_from_d455 = bracket extrinsic calibration               (D455 camera → LiDAR local)

T_world_camera    = T_world_lidar × T_lidar_from_d455           (D455 camera → world)
T_cam_from_world  = (T_world_camera)⁻¹                          (world → D455 camera)
```

To project a world-frame LiDAR point `P` into the camera image:
```
P_cam = T_cam_from_world × [P; 1]         → point in camera frame (x, y, z)
u = fx × P_cam.x / P_cam.z + cx           → pixel column
v = fy × P_cam.y / P_cam.z + cy           → pixel row
color = rgb_image[v, u]                    → sampled RGB color
```

Points with `P_cam.z ≤ 0` (behind camera) or outside image bounds are discarded.

#### 4.2 Per-Frame ICP Pose Refinement (Optional)

The interpolated pose has residual error from SLAM drift, timestamp quantization, and the extrinsic calibration itself. When `--refine-icp` is enabled:

1. **D455 depth unprojection**: Every 4th pixel of the depth image is unprojected to a 3D point cloud in camera frame (subsampled for speed)
2. **LiDAR crop**: World-frame LiDAR points within 1.5× max depth of the camera position are selected, then transformed to camera frame using the current (unrefinement) pose estimate
3. **Point-to-Plane ICP**: D455 cloud (source) is aligned to LiDAR cloud (target) in camera frame. Both are voxel-downsampled to 2cm. ICP runs for up to 30 iterations with 5cm correspondence distance
4. **Correction application**: If ICP fitness > 0.3, the correction is applied: `T_refined = T_icp × T_cam_from_world`. This accounts for the fact that ICP found how to move the D455 cloud to match the LiDAR cloud — meaning the camera was actually at a slightly different pose than initially estimated

This typically reduces per-frame alignment error from ~5px to ~1-2px.

#### 4.3 Z-Buffer Occlusion Filtering

Multiple LiDAR points can project to the same pixel (e.g., a wall point behind a chair). Without filtering, the wall would incorrectly receive the chair's color.

**Algorithm:**
1. Build a 2D Z-buffer (H×W float array, initialized to infinity)
2. For each projected point, write its depth to the Z-buffer using `np.minimum.at` — this atomically keeps only the minimum depth at each pixel
3. For each projected point, compare its depth to the Z-buffer minimum at its pixel. Only points within `zbuffer_tolerance` (default 5cm) of the minimum survive

This naturally keeps foreground objects and rejects occluded background points. The tolerance allows for the finite LiDAR point spacing — multiple points on the same surface at slightly different depths should all be colored.

#### 4.4 Depth Consistency Verification

Even with ICP refinement, small pose errors can cause a LiDAR point to project to the wrong pixel. For example, a chair point might project to a pixel where the camera actually sees a wall.

**Algorithm:**
For each surviving projected point:
1. Read the D455 measured depth at that pixel (16-bit mm → meters)
2. Compare to the LiDAR point's projected depth
3. If they disagree by more than `depth_consistency_threshold` (default 10cm), reject the color assignment

**What this fixes:**
- **Ghost/duplicate features** (e.g., a poster appearing twice on the wall): SLAM drift creates overlapping point layers at slightly different positions. Without depth verification, both layers get colored from the same frame. With it, the "wrong" layer's projected depth doesn't match the D455 depth, so it gets rejected
- **Background color bleeding**: When a foreground LiDAR point projects to a pixel where the camera sees background, the D455 depth at that pixel reveals the mismatch

#### 4.5 Depth Edge Erosion

At object silhouettes (e.g., a door frame against a hallway), the D455 stereo depth has large discontinuities. Due to the ~9.5cm stereo baseline, foreground LiDAR points near edges can project onto background pixels (parallax), picking up incorrect colors.

**Algorithm:**
1. Compute Sobel gradient magnitude of the D455 depth image (in mm)
2. Threshold at `edge_depth_threshold_mm` (default 300mm) to find depth discontinuity pixels
3. Dilate the edge mask by `edge_erosion_px` (default 2 pixels) using an elliptical structuring element
4. Reject any LiDAR point projecting into this exclusion zone

This prevents color bleeding at object boundaries without affecting the interior of surfaces.

#### 4.6 Motion Filtering

D455 frames captured during fast camera rotation are rejected:
1. Read the gyroscope angular velocity from the frame's IMU JSON
2. Compute the magnitude in degrees/second: `ω = √(gx² + gy² + gz²)`
3. Skip frames where `ω > max_gyro_dps` (default 60°/s)

**Why:** Fast rotation causes motion blur in the RGB image and makes the hardware timestamp less reliable (the frame integrates light over a range of poses). Both degrade color accuracy.

#### 4.7 Multi-Frame Blending

Each LiDAR point is typically visible from multiple D455 frames (average ~12 frames per point). Four blending strategies are available:

**Mean blending** (`--blend-mode mean`):
- For each point, accumulates RGB values from all depth-verified frames
- Final color = arithmetic mean of all contributions
- Robust: depth verification removes bad observations, averaging smooths noise
- Downside: can blur fine detail if poses have residual error

**Center-weighted blending** (`--blend-mode center_weighted`, default):
- Same as mean, but each frame's contribution is weighted by how close the point projects to the image center
- Weight function: `w = 0.5 × (1 + cos(π × d))` where `d` is the normalized distance from center (0=center, 1=edge)
- Points near the image center get weight ~1.0, edges get ~0.0
- Creates smooth transitions in overlap zones between frames without the blur of uniform averaging
- Naturally favors frames where the point is well-centered (better lens quality, less distortion)

**Closest camera** (`--blend-mode closest`):
- Each point keeps the color from whichever frame had the smallest camera-to-point depth
- No averaging — sharpest possible colors
- Fragile: one bad pose can produce a visible seam

**First wins** (`--blend-mode first`):
- First depth-verified frame to see a point determines its color
- No subsequent frames can override
- Fastest, but produces visible boundaries between frame coverage areas

#### 4.8 Height Colormap for Uncolored Points

The LiDAR has 360° coverage but the D455 has ~90° forward FOV, so typically ~45-55% of LiDAR points have no camera coverage. These receive a **height-based HSV rainbow**:
- Hue: blue (240°) at floor → cyan → green → yellow → red (0°) at ceiling
- Saturation: 0.6 (muted, clearly distinguishable from real RGB)
- Value: 0.5 (dimmed to make it obvious these aren't real colors)

The height range is computed from the 2nd and 98th percentiles of Z coordinates to avoid outlier sensitivity.

#### 4.9 Batched Processing

To limit memory usage, frames are processed in batches of 8:
- Each batch loads RGB + depth images via a `ThreadPoolExecutor`
- Projection + filtering + accumulation happens per-frame within the batch
- Batch memory is freed before the next batch loads
- This keeps peak RAM usage manageable even with 100+ frames

**Output:** `colored_cloud.ply` with per-vertex RGB colors

---

### Post-Processing Tools

#### Color Outlier Removal (`postprocess_ply.py`)

Detects and removes silhouette color bleeding artifacts where background color (e.g., bright sky) gets incorrectly assigned to foreground points (e.g., tree bark):

1. Identify candidate points: brightness > 0.65 (only bright points can be sky bleed)
2. Build a KD-tree on all point positions
3. For each candidate, query K=20 nearest spatial neighbors
4. Compute the median color of the neighborhood
5. If the candidate's Euclidean color distance from the median exceeds 0.25, remove the point

#### Color Smoothing (`clean_colors.py`)

Replaces color outliers with their neighborhood median color (preserves point count):
1. For each point, find neighbors within a radius (default 15mm)
2. Compute median neighbor color
3. If L1 RGB distance from median exceeds threshold (default 30/255), replace with median

Optional **bilateral color smoothing**: weighted average of neighbor colors where weights are the product of:
- **Spatial weight**: Gaussian falloff with distance (σ = radius/2)
- **Color similarity weight**: Gaussian falloff with color difference (σ = 30/255)

This smooths noise while preserving color edges — similar colors within radius blend together, but sharp color transitions (e.g., wall-to-door) are preserved.

#### TSDF Fusion (`tsdf_fusion.py`)

Alternative to LiDAR+projection: builds a solid mesh directly from D455 RGB-D frames using **Truncated Signed Distance Function** volumetric integration:

1. Creates an Open3D `ScalableTSDFVolume` (hash-mapped voxel grid, default 5mm voxels)
2. For each D455 frame:
   - Computes camera pose via `T_world_lidar × T_lidar_from_d455` (same as projection pipeline)
   - Integrates RGB-D pair into the TSDF volume (each voxel stores running weighted average of signed distance to nearest surface + RGB color)
3. Extracts a watertight triangle mesh via **marching cubes**
4. Also samples the mesh surface to produce a point cloud

Colors are pixel-perfect (same sensor captures geometry + color) and the mesh has no gaps, but coverage is limited to the D455's forward-facing ~90° FOV and 3m depth range.

---

### Viewing and Export

#### Godot Point Cloud Viewer (`godot_viewer/`)

A real-time 3D viewer built in Godot 4 for navigating large point clouds:

- **PLY loader** (`main.gd`): Parses binary and ASCII PLY files directly in GDScript, handling variable property layouts (float32/float64 positions, uint8/float32 colors). Builds an `ArrayMesh` with `PRIMITIVE_POINTS`
- **Custom shader** (`point_cloud.gdshader`): Renders round points (discards fragments outside a circle using `POINT_COORD` distance check) with sRGB→linear color conversion (`pow(color, 2.2)`)
- **FPS-style controls**: WASD movement, mouse look, Space/C for vertical, Shift for sprint, scroll wheel for point size adjustment
- **Coordinate conversion**: PLY (Z-up or arbitrary) → Godot (Y-up right-handed): `Y_godot = Z_ply, Z_godot = -Y_ply`

#### Binary Converter (`convert_ply_for_godot.py`)

Converts PLY to a compact binary format (`.bin`) for faster Godot loading:
- Header: uint32 point count
- Body: float32[N×3] positions (Y-up converted) + float32[N×3] colors (0-1 range)

#### Walk Viewer (`walk_viewer.py`)

Open3D-based viewer with simultaneous WASD movement (like a game engine):
- Uses **X11 keyboard polling** (`XQueryKeymap`) for held-key detection — Open3D's built-in key callbacks only fire once per press, making smooth WASD movement impossible
- Applies camera movement by modifying the Open3D view control's extrinsic matrix directly

#### 3D Gaussian Splatting Export (`convert_to_3dgs.py`)

Converts the scan data to **nerfstudio's Splatfacto format** for neural rendering:
1. Computes per-frame camera poses: `T_world_camera = T_world_lidar × T_lidar_from_d455`
2. Centers all poses at the scene centroid
3. Converts D455 RGB frames to JPEG
4. Writes `transforms.json` (OPENCV camera model with intrinsics + distortion + per-frame 4×4 poses)
5. Downsamples LiDAR cloud as initialization `points3d.ply`

This provides nerfstudio with everything needed to train 3D Gaussian Splatting — the LiDAR points serve as initial Gaussian positions (much better than random or SfM initialization).

---

### Utility Modules

#### `utils/projection.py`

Core projection and filtering functions used by `project_color.py` and `debug_frame.py`:

- `project_points_to_image()`: Vectorized 3D→2D projection (homogeneous transform + perspective divide + bounds check)
- `zbuffer_filter()`: Builds min-depth buffer using `np.minimum.at`, keeps points within tolerance
- `depth_consistency_filter()`: Compares projected depths vs D455 measured depth
- `depth_edge_erosion_filter()`: Sobel gradient → threshold → dilate → reject
- `color_points_from_frame()`: Orchestrates the full per-frame pipeline (project → z-buffer → depth check → edge erosion → sample colors)
- `color_points_multi_frame()`: Parallel multi-frame projection with various blending modes (mean, closest, first-wins, center-weighted)

#### `utils/timestamps.py`

- `find_nearest_timestamps()`: Efficient nearest-neighbor timestamp matching using `np.searchsorted` on sorted arrays
- `interpolate_pose_at_timestamp()`: Interpolates a 4×4 pose matrix between two bracketing timestamps using:
  - **Linear interpolation** for translation: `t = (1-α)t₀ + αt₁`
  - **SLERP** (Spherical Linear Interpolation) for rotation: ensures the interpolated rotation stays on SO(3) (no scaling/shearing artifacts). Uses `scipy.spatial.transform.Slerp`

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
│   ├── frame_001_imu.json        # Accelerometer, gyro, orientation, timestamp
│   └── intrinsics.json           # Camera matrix (fx, fy, cx, cy, coeffs)
├── lidar_cloud.ply               # Processed LiDAR point cloud
├── lidar_trajectory.npz          # LiDAR poses + timestamps
├── extrinsic.json                # Bracket calibration (T_lidar_from_d455)
└── colored_cloud.ply             # Final output — colored point cloud
```

### Key File Formats

**Trajectories** (`.npz`):
- `poses`: `(N, 4, 4)` float64 — homogeneous transform matrices
- `timestamps_ns`: `(N,)` int64 — nanosecond epoch timestamps

**Extrinsic** (`.json`):
- `transform`: 4×4 matrix (T_lidar_from_d455)
- `translation_m`: 3-vector (bracket offset in meters)
- `rotation_matrix`: 3×3 rotation
- `method`: calibration algorithm used
- `alignment_info`: fitness scores, number of frames used, voxel size

---

## Configuration

All parameters are in `config.yaml`:

| Section | Key Parameter | Default | Purpose |
|---------|--------------|---------|---------|
| `lidar` | `ip` | 192.168.1.171 | Livox Mid-360 IP address |
| `d455` | `max_depth_m` | 3.0 | D455 depth range limit |
| `capture` | `auto_capture_hz` | 2 | D455 frame capture rate |
| `processing` | `lidar_voxel_size` | 0.003 | LiDAR downsample resolution (3mm) |
| `processing` | `lidar_fov_filter` | true | Filter LiDAR to D455 FOV |
| `projection` | `zbuffer_tolerance` | 0.05 | Z-buffer occlusion tolerance (m) |
| `projection` | `max_distance_m` | 10.0 | Max point-to-camera distance |
| `projection` | `depth_consistency_threshold` | 0.10 | LiDAR-D455 depth agreement (m) |
| `projection` | `edge_erosion_px` | 2 | Exclusion zone around depth edges (px) |
| `projection` | `edge_depth_threshold_mm` | 300 | Sobel gradient threshold for edge detection |

---

## Dependencies

- **ROS2 Jazzy** + Livox ROS2 driver + RKO-LIO SLAM
- **Open3D** — point cloud I/O, ICP registration, pose graph optimization, TSDF, visualization
- **OpenCV** — image I/O, undistortion, Sobel gradients, color space conversion
- **NumPy** — all array operations and linear algebra
- **SciPy** — `Rotation`, `Slerp`, `cKDTree` for spatial queries
- **PyYAML** — configuration parsing
- **pyrealsense2** — Intel RealSense D455 SDK (capture only)
- **rosbags** — ROS2 bag reading without full ROS install
- **Godot 4** — point cloud viewer (optional)
- **PyVista** — alternative Gaussian splat rendering (optional)

---

## Hardware

- **Livox Mid-360** — 360° non-repetitive scanning LiDAR (~200k pts/sec, built-in IMU)
- **Intel RealSense D455** — RGB-D camera (1280×720, stereo depth to 6m, 9-DOF IMU)
- **Custom 3D-printed bracket** — rigid mount with ~13cm baseline, ~60° tilt from vertical
