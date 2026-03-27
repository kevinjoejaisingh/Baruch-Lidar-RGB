# Point Cloud Rendering Improvement Options

Current state: 10.6M colored points from LiDAR+RGB fusion. Looks good but blocky in MeshLab (big square splats).

Already tried and rejected:
- PyVista gaussian splatting — didn't work
- Point upsampling — slow, marginal quality gain
- Bilateral color smoothing — too blurry, lost detail
- Poisson surface reconstruction — broke on tree canopy (forces watertight surface)

## Option 1: CloudCompare + EDL Shader (2 minutes)

Open PLY in CloudCompare instead of MeshLab. Enable **Eye-Dome Lighting (EDL)** shader — a screen-space effect that adds depth-aware edges and shading to flat points, making them look solid without meshing. Set point size to 1-2px.

Zero processing, just a better viewer.

```
cloudcompare.CloudCompare debug_frames0-617_clean.ply
# Then: Display > Shaders > EDL
# Then: reduce point size to 1-2
```

## Option 2: Ball Pivoting Algorithm (10 min scripting)

Unlike Poisson (watertight = hallucinated tree canopy), BPA connects nearby points with triangles and leaves gaps as gaps. Handles tree trunks, ground, bushes. Open3D has it built in.

```python
import open3d as o3d
pcd = o3d.io.read_point_cloud("cloud.ply")
pcd.estimate_normals()
radii = [0.005, 0.01, 0.02, 0.04]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
```

## Option 3: 3D Gaussian Splatting (neural rendering)

State-of-the-art photorealistic rendering (Kerbl et al. 2023). Each point becomes an anisotropic 3D Gaussian with learnable color/opacity. Initialize from LiDAR positions + colors, optimize using 618 D455 camera images.

We already have camera poses and images — that's the hard part done.

Tools:
- nerfstudio (splatfacto method) — easiest setup
- github.com/graphdeco-inria/gaussian-splatting — original implementation
- gsplat (github.com/nerfstudio-project/gsplat) — lightweight library

Bigger project but highest quality payoff.

## Option 4: Blender Import + Geometry Nodes

Import PLY, use geometry nodes to instance tiny oriented disks at each point (aligned by estimated normals). Cycles rendering = photorealistic with proper lighting.

Good for final publication-quality renders, not real-time viewing.
