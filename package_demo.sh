#!/bin/bash
# Bundle the BARUCH YC-demo app into a single self-contained folder.
# Output: dist/baruch-demo/   (and dist/baruch-demo.tar.gz)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/dist/baruch-demo"

echo "→ cleaning $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"

# ── Python app ──────────────────────────────────────────────────────────────
echo "→ copying app/"
cp -r "$ROOT/app" "$DEST/"
# strip __pycache__ to keep the bundle clean
find "$DEST/app" -type d -name __pycache__ -prune -exec rm -rf {} +

# ── Godot embedded viewer ───────────────────────────────────────────────────
if [ -d "$ROOT/godot_viewer" ]; then
    echo "→ copying godot_viewer/"
    cp -r "$ROOT/godot_viewer" "$DEST/"
fi

# ── Pipeline scripts (subprocess-invoked by app/pipeline.py) ────────────────
echo "→ copying pipeline scripts"
for f in capture.py process_lidar.py calibrate_extrinsic.py \
         project_color_cuda_sharp.py extrinsic.json clean_colors.py; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$DEST/" || true
done

# ── Shared utils package (imported by pipeline scripts) ─────────────────────
if [ -d "$ROOT/utils" ]; then
    echo "→ copying utils/"
    cp -r "$ROOT/utils" "$DEST/"
    find "$DEST/utils" -type d -name __pycache__ -prune -exec rm -rf {} +
fi

# ── Config + requirements ───────────────────────────────────────────────────
cp "$ROOT/config.yaml" "$DEST/"
cp "$ROOT/app/requirements.txt" "$DEST/requirements.txt"

# ── Data: hero PLY + the single recording ───────────────────────────────────
mkdir -p "$DEST/data/recordings"

REC_SRC="$ROOT/newhospitalscan_cuda.ply"
HERO_SRC="/home/kevin/kevin_website/bedroom_definitive_web.ply"

if [ -f "$REC_SRC" ]; then
    echo "→ copying recording: $(basename "$REC_SRC")"
    cp "$REC_SRC" "$DEST/data/recordings/"
else
    echo "!! missing $REC_SRC — skipping (recording will be empty)"
fi

if [ -f "$HERO_SRC" ]; then
    echo "→ copying hero PLY"
    cp "$HERO_SRC" "$DEST/data/hero.ply"
else
    echo "!! missing $HERO_SRC — title screen will fall back to logo"
fi

# ── run.sh ──────────────────────────────────────────────────────────────────
cat > "$DEST/run.sh" <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
exec python3 -m app.main
EOF
chmod +x "$DEST/run.sh"

# ── README ──────────────────────────────────────────────────────────────────
cat > "$DEST/README.md" <<'EOF'
# BARUCH — YC demo bundle

Self-contained: app code, embedded Godot viewer, hero point cloud, and the
sample recording the dashboard surfaces.

## Run

```
./run.sh
```

(Or: `python3 -m app.main` from this directory.)

## First-time setup

```
pip install -r requirements.txt
```

`pyrealsense2` is only needed for the Connect/Record screens (they probe
for a D455 camera). The demo flow — Title → Dashboard → Viewer — works
without any sensors connected.

## Layout

- `app/`              — UI (imgui-bundle, moderngl, glfw)
- `godot_viewer/`     — Godot project + prebuilt binary, embedded as a
                        child X11 window
- `data/hero.ply`     — landing-page point cloud
- `data/recordings/`  — recordings shown in the dashboard catalog
- `config.yaml`       — paths + LiDAR config
- pipeline scripts at root — invoked as subprocesses by app/pipeline.py
EOF

# ── Tarball for distribution ────────────────────────────────────────────────
echo "→ creating tarball"
( cd "$ROOT/dist" && tar czf baruch-demo.tar.gz baruch-demo/ )

echo
echo "✓ bundle: $DEST"
du -sh "$DEST"
echo "✓ tarball: $ROOT/dist/baruch-demo.tar.gz"
du -sh "$ROOT/dist/baruch-demo.tar.gz"
