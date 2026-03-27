#!/usr/bin/env python3
"""View a colored PLY with PyVista Gaussian splat rendering."""
import sys
import pyvista as pv

path = sys.argv[1] if len(sys.argv) > 1 else "folsom_test/debug_frames0-617.ply"
point_size = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print(f"Loading {path}...")
mesh = pv.read(path)
print(f"Points: {mesh.n_points:,}")

pl = pv.Plotter()
pl.add_mesh(
    mesh,
    style='points_gaussian',
    render_points_as_spheres=False,
    point_size=point_size,
    emissive=True,
)
pl.background_color = 'black'
pl.show()
