"""Scan disk for completed PLYs to show on the dashboard.

For the demo bundle, recordings live exclusively in
<project_root>/data/recordings/. Local-dev fallback also scans the project
root and a few legacy folders so working in-tree still surfaces files.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .state import State


@dataclass
class Recording:
    path: Path
    label: str
    size_mb: float
    mtime: float

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")


def _entry(path: Path) -> Recording:
    try:
        st = path.stat()
        size_mb = st.st_size / (1024 * 1024)
        mtime = st.st_mtime
    except OSError:
        size_mb = 0.0
        mtime = 0.0
    parent = path.parent.name or path.parent.as_posix()
    label = f"{parent}/{path.name}"
    return Recording(path=path, label=label, size_mb=size_mb, mtime=mtime)


def list_recordings(state: State) -> list[Recording]:
    """All PLYs to show on the dashboard, newest first.

    Demo build: only data/recordings/. Dev build: also scans project root
    and a few legacy folders if present.
    """
    rec_dir = state.project_root / "data" / "recordings"
    seen: set[Path] = set()
    found: list[Recording] = []

    if rec_dir.exists():
        for ply in sorted(rec_dir.rglob("*.ply")):
            if ply.name == "lidar_cloud.ply":
                continue
            if ply in seen:
                continue
            seen.add(ply)
            found.append(_entry(ply))

    # If the demo recordings dir already has files, return just those —
    # this is the YC-demo branch, kept clean.
    if found:
        found.sort(key=lambda r: r.mtime, reverse=True)
        return found

    # Local-dev fallback: also surface PLYs in legacy locations
    for name in ("folsom_test", "hosiptal-rooms", "hospital", "hospital_3dgs", "OR2", "outputs"):
        d = state.project_root / name
        if d.exists() and d.is_dir():
            for ply in sorted(d.rglob("*.ply")):
                if ply.name == "lidar_cloud.ply" or ply in seen:
                    continue
                seen.add(ply)
                found.append(_entry(ply))
    for ply in sorted(state.project_root.glob("*.ply")):
        if ply in seen:
            continue
        seen.add(ply)
        found.append(_entry(ply))

    found.sort(key=lambda r: r.mtime, reverse=True)
    return found


def list_scan_dirs(state: State) -> list[Path]:
    """Directories we can run the pipeline on."""
    rec_dir = state.project_root / "data" / "recordings"
    out: list[Path] = []
    if rec_dir.exists():
        for p in sorted(rec_dir.iterdir(), reverse=True):
            if p.is_dir():
                out.append(p)
    return out
