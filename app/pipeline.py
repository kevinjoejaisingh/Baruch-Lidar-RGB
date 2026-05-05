"""Wrap existing pipeline scripts as background subprocesses with stage tracking."""
import os
import re
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from .state import State


STAGES = [
    ("capture",   "capturing"),
    ("lidar",     "processing lidar"),
    ("calibrate", "calibrating extrinsic"),
    ("project",   "coloring point cloud"),
]


def _push_log(state: State, line: str):
    line = line.rstrip()
    if not line:
        return
    state.pipeline_log.append(line)
    if len(state.pipeline_log) > 400:
        del state.pipeline_log[:200]


def _stream(proc: subprocess.Popen, state: State):
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", errors="replace")
        except Exception:
            continue
        _push_log(state, line)
    proc.stdout.close()


def _run(cmd: list[str], cwd: Path, state: State, stage_label: str) -> int:
    state.pipeline_stage = stage_label
    _push_log(state, f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    t = threading.Thread(target=_stream, args=(proc, state), daemon=True)
    t.start()
    rc = proc.wait()
    t.join(timeout=1.0)
    return rc


def run_replay(state: State, scan_dir: Path):
    """Replay = use an existing scan_dir; skip capture, run process+calibrate+project."""
    threading.Thread(target=_replay_thread, args=(state, scan_dir), daemon=True).start()


def _replay_thread(state: State, scan_dir: Path):
    state.scan_dir = scan_dir
    state.pipeline_progress = 0.0
    state.pipeline_done = False
    state.pipeline_error = ""
    root = state.project_root

    try:
        if not (scan_dir / "lidar_cloud.ply").exists():
            state.pipeline_stage = "processing lidar"
            state.pipeline_progress = 0.10
            rc = _run(["python3", "process_lidar.py", str(scan_dir)], root, state, "processing lidar")
            if rc != 0:
                raise RuntimeError(f"process_lidar.py failed (rc={rc})")
        else:
            _push_log(state, "lidar_cloud.ply already exists — skipping process_lidar")

        state.pipeline_progress = 0.40
        if not (scan_dir / "extrinsic.json").exists():
            rc = _run(["python3", "calibrate_extrinsic.py", str(scan_dir)], root, state, "calibrating extrinsic")
            if rc != 0:
                _push_log(state, "calibrate failed — falling back to permanent extrinsic")
        else:
            _push_log(state, "extrinsic.json present — skipping calibration")

        state.pipeline_progress = 0.55
        out_ply = scan_dir / f"colored_{datetime.now().strftime('%H%M%S')}.ply"
        rc = _run(
            ["python3", "project_color_cuda_sharp.py", str(scan_dir), "-o", str(out_ply)],
            root, state, "coloring point cloud",
        )
        if rc != 0:
            raise RuntimeError(f"project_color_cuda_sharp.py failed (rc={rc})")

        state.pipeline_progress = 1.0
        state.output_ply = out_ply
        state.pipeline_stage = "done"
        state.pipeline_done = True
    except Exception as e:
        state.pipeline_error = str(e)
        state.pipeline_stage = "error"
        state.pipeline_done = True


def start_recording(state: State) -> subprocess.Popen | None:
    """Launch capture.py interactively. Returns the Popen so the UI can stop it."""
    name = "scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_root = Path(state.cfg["paths"]["scan_dir"]).expanduser()
    scan_root.mkdir(parents=True, exist_ok=True)
    state.scan_dir = scan_root / name
    state.pipeline_stage = "capturing"
    state.pipeline_progress = 0.05
    state.pipeline_log = []
    state.pipeline_done = False
    state.pipeline_error = ""

    proc = subprocess.Popen(
        ["python3", "capture.py", name],
        cwd=str(state.project_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    threading.Thread(target=_stream, args=(proc, state), daemon=True).start()
    return proc


def stop_recording_and_process(state: State, proc: subprocess.Popen):
    """Send the two ENTERs capture.py expects, then run the rest of the pipeline."""
    def _go():
        try:
            if proc.poll() is None:
                try:
                    proc.stdin.write(b"\n")
                    proc.stdin.flush()
                except Exception:
                    pass
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=10)
            _replay_thread(state, state.scan_dir)
        except Exception as e:
            state.pipeline_error = str(e)
            state.pipeline_stage = "error"
            state.pipeline_done = True

    threading.Thread(target=_go, daemon=True).start()
