"""Real connect checks: D455 via pyrealsense2, Livox via ICMP ping."""
import socket
import subprocess
import threading

from .state import State


def _probe_d455(state: State):
    state.d455_status = "connecting"
    state.d455_detail = "querying realsense devices"
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            state.d455_status = "fail"
            state.d455_detail = "no realsense devices found"
            return
        d455 = None
        for d in devices:
            name = d.get_info(rs.camera_info.name)
            if "D455" in name:
                d455 = d
                break
        if d455 is None:
            d = devices[0]
            name = d.get_info(rs.camera_info.name)
            state.d455_status = "fail"
            state.d455_detail = f"found {name}, expected d455"
            return
        serial = d455.get_info(rs.camera_info.serial_number)
        state.d455_status = "ok"
        state.d455_detail = f"d455 — sn {serial[-6:]}"
    except Exception as e:
        state.d455_status = "fail"
        state.d455_detail = f"{type(e).__name__}: {e}"


def _probe_lidar(state: State):
    state.lidar_status = "connecting"
    cfg = state.cfg.get("lidar", {})
    ip = cfg.get("ip", "192.168.1.171")
    state.lidar_detail = f"pinging {ip}"
    try:
        # ICMP ping (1 packet, 1s timeout). Works on linux without root via iputils.
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "1", ip],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            state.lidar_status = "ok"
            state.lidar_detail = f"livox mid-360 — {ip}"
            return

        # fallback: try opening Livox control port (TCP-ish)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.sendto(b"\x00", (ip, 65000))
        state.lidar_status = "fail"
        state.lidar_detail = f"no reply from {ip}"
    except Exception as e:
        state.lidar_status = "fail"
        state.lidar_detail = f"{type(e).__name__}: {e}"


def start_probes(state: State):
    threading.Thread(target=_probe_d455, args=(state,), daemon=True).start()
    threading.Thread(target=_probe_lidar, args=(state,), daemon=True).start()
