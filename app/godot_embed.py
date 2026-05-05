"""Launch Godot and reparent its X11 window into our GLFW window.

Linux/X11 only. Uses python-xlib for the reparent + geometry control,
and `_NET_WM_PID` to find the Godot window after launch.
"""
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from Xlib import display, X, error as xerror
from Xlib.protocol import event as xevent


def _walk_windows(win):
    """Yield this window plus all descendants in the X tree."""
    yield win
    try:
        children = win.query_tree().children
    except xerror.BadWindow:
        return
    for c in children:
        yield from _walk_windows(c)


def _find_window_for_pid(disp, pid: int) -> int | None:
    """Search the X server for a window owned by `pid` that has a name."""
    NET_WM_PID = disp.intern_atom("_NET_WM_PID")
    NET_WM_NAME = disp.intern_atom("_NET_WM_NAME")
    UTF8 = disp.intern_atom("UTF8_STRING")
    root = disp.screen().root

    for w in _walk_windows(root):
        try:
            prop = w.get_full_property(NET_WM_PID, X.AnyPropertyType)
        except xerror.BadWindow:
            continue
        if prop is None or not prop.value:
            continue
        try:
            wpid = int(prop.value[0])
        except Exception:
            continue
        if wpid != pid:
            continue
        # Prefer a window with a name (skip hidden helper windows)
        try:
            name = w.get_full_property(NET_WM_NAME, UTF8)
            if name is None:
                name = w.get_wm_name()
        except xerror.BadWindow:
            continue
        if not name:
            continue
        return w.id
    return None


class GodotEmbed:
    """Manages an embedded Godot subprocess."""

    def __init__(self):
        self.disp = display.Display()
        self.proc: subprocess.Popen | None = None
        self.win = None
        self.exited: bool = False
        self.error: str = ""
        self._geom = (0, 0, 0, 0)

    # ---------- lifecycle ---------------------------------------------------
    def start(self, binary: Path, ply_path: Path, parent_xid: int,
              x: int, y: int, w: int, h: int,
              project_dir: Path | None = None) -> bool:
        """Launch godot with `ply_path`, then reparent into `parent_xid`.

        If `project_dir` is given and a `godot` CLI is on PATH, runs from
        source so live edits to main.gd take effect without a rebuild.
        Otherwise falls back to the prebuilt binary.
        """
        self.exited = False
        self.error = ""

        godot_cli = self._find_godot_cli()
        try:
            if godot_cli is not None and project_dir is not None and project_dir.exists():
                cmd = [
                    godot_cli,
                    "--path", str(project_dir),
                    "--position", "9999,9999",
                    "--resolution", f"{max(w, 320)}x{max(h, 240)}",
                    str(ply_path),
                ]
            else:
                cmd = [
                    str(binary),
                    str(ply_path),
                    "--position", "9999,9999",
                    "--resolution", f"{max(w, 320)}x{max(h, 240)}",
                ]
            print(f"[godot_embed] launching: {' '.join(cmd)}", flush=True)
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            threading.Thread(target=self._drain_stdout, daemon=True).start()
        except Exception as e:
            self.error = f"launch failed: {e}"
            return False

        # Find Godot's window (give it more time for source-mode startup)
        t0 = time.time()
        wid = self._wait_for_window(self.proc.pid, timeout=15.0)
        print(f"[godot_embed] _wait_for_window → {wid} in {time.time()-t0:.2f}s", flush=True)
        if wid is None:
            self.error = "could not locate godot window"
            self.stop()
            return False

        try:
            self.win = self.disp.create_resource_object("window", wid)
            parent = self.disp.create_resource_object("window", parent_xid)
            self.win.reparent(parent, x, y)
            self.win.configure(x=x, y=y, width=max(w, 1), height=max(h, 1))
            self.win.map()
            self.disp.sync()
        except Exception as e:
            self.error = f"reparent failed: {e}"
            self.stop()
            return False

        self._geom = (x, y, w, h)

        # Watcher thread — sets self.exited when Godot quits on its own
        threading.Thread(target=self._watch, daemon=True).start()
        return True

    @staticmethod
    def _find_godot_cli() -> str | None:
        for cand in ("/home/kevin/bin/godot", "godot", "godot4"):
            try:
                r = subprocess.run([cand, "--version"], capture_output=True, timeout=2)
                if r.returncode == 0:
                    return cand
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue
        return None

    def _wait_for_window(self, pid: int, timeout: float) -> int | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            wid = _find_window_for_pid(self.disp, pid)
            if wid is not None:
                return wid
            # also exit early if proc died
            if self.proc is not None and self.proc.poll() is not None:
                return None
            time.sleep(0.1)
        return None

    def _watch(self):
        if self.proc is None:
            return
        try:
            self.proc.wait()
        except Exception:
            pass
        self.exited = True
        print(f"[godot_embed] process exited (rc={self.proc.returncode if self.proc else '?'})", flush=True)

    def _drain_stdout(self):
        if self.proc is None or self.proc.stdout is None:
            return
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace").rstrip()
            except Exception:
                continue
            if line:
                print(f"[godot] {line}", flush=True)

    def update_geometry(self, x: int, y: int, w: int, h: int):
        if self.win is None:
            return
        if (x, y, w, h) == self._geom:
            return
        try:
            self.win.configure(x=x, y=y, width=max(w, 1), height=max(h, 1))
            self.disp.sync()
            self._geom = (x, y, w, h)
        except Exception:
            pass

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=1.0)
                except Exception:
                    pass
            self.proc = None
        self.win = None
        self._geom = (0, 0, 0, 0)
