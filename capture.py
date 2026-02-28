#!/usr/bin/env python3
"""
Unified capture: Livox Mid-360 LiDAR (via ROS2 SLAM) + Intel RealSense D455 (RGB-D + IMU).

Flow:
  1. ENTER → setup network, start LiDAR driver + SLAM + bag recording
  2. Initialize D455, auto-capture at 2Hz with live 2x2 preview
  3. ENTER → capture loop-closure frame, gracefully stop everything

Usage:
  python capture.py [scan_name]
  python capture.py                  # auto-generates scan_YYYYMMDD_HHMMSS
"""

import os
import sys
import json
import signal
import subprocess
import threading
import time
from datetime import datetime
from math import sqrt, atan2, degrees
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

LIDAR_IP = CFG["lidar"]["ip"]
HOST_IP = CFG["lidar"]["host_ip"]
NET_IFACE = CFG["lidar"]["interface"]
SLAM_CONFIG = Path(CFG["lidar"]["slam_config"]).expanduser()
BAG_TOPICS = " ".join(CFG["lidar"]["bag_topics"])
ROS_SETUP = CFG["lidar"]["ros_setup"]

D455_W = CFG["d455"]["width"]
D455_H = CFG["d455"]["height"]
D455_FPS = CFG["d455"]["fps"]
D455_MAX_DEPTH = CFG["d455"]["max_depth_m"]

CAPTURE_HZ = CFG["capture"]["auto_capture_hz"]
WARMUP_FRAMES = CFG["capture"]["warmup_frames"]
DRIVER_WAIT = CFG["capture"]["driver_startup_wait_s"]
SLAM_WAIT = CFG["capture"]["slam_startup_wait_s"]

SCAN_DIR = Path(CFG["paths"]["scan_dir"]).expanduser()

# ---------------------------------------------------------------------------
# Subprocess helpers (from scan_gui.py)
# ---------------------------------------------------------------------------

def run_shell(cmd, **kwargs):
    """Run a command through bash with ROS2 sourced."""
    full_cmd = f"{ROS_SETUP} && {cmd}"
    return subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        **kwargs,
    )


def kill_process_group(proc):
    """Kill entire process group."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def sigint_process_group(proc):
    """Send SIGINT to process group for clean shutdown."""
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except (ProcessLookupError, OSError):
            pass


def drain_stdout(proc, label):
    """Read stdout in a daemon thread to prevent buffer deadlock."""
    def _reader():
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    print(f"  [{label}] {text}")
        except (ValueError, OSError):
            pass
    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# D455 helpers (from capture_tool.py)
# ---------------------------------------------------------------------------

def calculate_orientation(accel_x, accel_y, accel_z):
    """Calculate pitch and roll from accelerometer data (D455: X-right, Y-down, Z-forward)."""
    magnitude = sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    if magnitude < 0.001:
        return 0.0, 0.0
    ax = accel_x / magnitude
    ay = accel_y / magnitude
    az = accel_z / magnitude
    pitch = atan2(az, ay)
    roll = atan2(ax, ay)
    return degrees(pitch), degrees(roll)


def draw_bubble_level(image, current_pitch, current_roll, ref_pitch, ref_roll):
    """Draw a bubble level overlay on the image."""
    h, w = image.shape[:2]
    center_x, center_y = 80, h - 80
    radius = 50
    scale = 2.0

    # Background circle
    overlay = image.copy()
    cv2.circle(overlay, (center_x, center_y), radius + 5, (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Boundary circles
    cv2.circle(image, (center_x, center_y), radius, (200, 200, 200), 2)
    cv2.circle(image, (center_x, center_y), radius // 2, (100, 100, 100), 1)

    def clamp_to_circle(cx, cy, px, py, r):
        dx, dy = px - cx, py - cy
        dist = sqrt(dx**2 + dy**2)
        if dist > r - 8:
            px = int(cx + dx * (r - 8) / dist)
            py = int(cy + dy * (r - 8) / dist)
        return px, py

    # Reference (green)
    if ref_pitch is not None and ref_roll is not None:
        rx, ry = int(center_x + ref_roll * scale), int(center_y - ref_pitch * scale)
        rx, ry = clamp_to_circle(center_x, center_y, rx, ry, radius)
        cv2.circle(image, (rx, ry), 8, (0, 255, 0), -1)
        cv2.circle(image, (rx, ry), 8, (0, 200, 0), 2)

    # Current (red)
    cx_, cy_ = int(center_x + current_roll * scale), int(center_y - current_pitch * scale)
    cx_, cy_ = clamp_to_circle(center_x, center_y, cx_, cy_, radius)
    cv2.circle(image, (cx_, cy_), 8, (0, 0, 255), -1)
    cv2.circle(image, (cx_, cy_), 8, (0, 0, 200), 2)

    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_y = center_y + radius + 20
    cv2.putText(image, f"Cur: P={current_pitch:.1f} R={current_roll:.1f}",
                (10, text_y), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    if ref_pitch is not None:
        cv2.putText(image, f"Ref: P={ref_pitch:.1f} R={ref_roll:.1f}",
                    (10, text_y + 15), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)


def colorize_depth(depth_frame, max_depth_mm=3000):
    """Convert depth frame to colorized image for display."""
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_clipped = np.clip(depth_image, 0, max_depth_mm)
    depth_scaled = (depth_clipped / max_depth_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_scaled, cv2.COLORMAP_TURBO)


def create_grid_display(frame1_rgb, frame1_depth_col, live_rgb, live_depth_col,
                        current_pitch, current_roll, ref_pitch, ref_roll, frame_count):
    """Create 2x2 grid: [frame1_depth, frame1_rgb] / [live_depth+bubble, live_rgb+counter]."""
    h, w = live_rgb.shape[:2]

    if frame1_rgb is None:
        top_left = np.zeros((h, w, 3), dtype=np.uint8)
        top_right = np.zeros((h, w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(top_left, "Waiting for first capture...",
                    (w // 4, h // 2), font, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(top_right, "Waiting for first capture...",
                    (w // 4, h // 2), font, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        top_left = frame1_depth_col.copy()
        top_right = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)

    bottom_left = live_depth_col.copy()
    bottom_right = cv2.cvtColor(live_rgb, cv2.COLOR_RGB2BGR)

    # Bubble level on live depth
    draw_bubble_level(bottom_left, current_pitch, current_roll, ref_pitch, ref_roll)

    # Frame counter on live RGB
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bottom_right, f"Frame: {frame_count}", (12, 32), font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bottom_right, f"Frame: {frame_count}", (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Labels
    label_color = (255, 255, 255)
    for img, label in [(top_left, "Frame 1 Depth"), (top_right, "Frame 1 RGB"),
                       (bottom_left, "Live Depth"), (bottom_right, "Live RGB")]:
        cv2.putText(img, label, (10, 25), font, 0.5, label_color, 1, cv2.LINE_AA)

    top_row = np.hstack([top_left, top_right])
    bottom_row = np.hstack([bottom_left, bottom_right])
    return np.vstack([top_row, bottom_row])


def save_d455_frame(d455_dir, frame_num, rgb_image, depth_frame, imu_data, is_loop_closure=False):
    """Save RGB PNG, depth PNG (16-bit mm), and IMU JSON for one D455 frame."""
    prefix = f"frame_{frame_num:03d}"

    # RGB
    rgb_path = os.path.join(d455_dir, f"{prefix}_rgb.png")
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

    # Depth (16-bit mm)
    depth_path = os.path.join(d455_dir, f"{prefix}_depth.png")
    depth_image = np.asanyarray(depth_frame.get_data())
    cv2.imwrite(depth_path, depth_image)

    # IMU JSON with capture_timestamp_ns for LiDAR sync
    imu_json = {
        "frame_number": frame_num,
        "capture_timestamp_ns": time.time_ns(),
        "timestamp_ms": time.time() * 1000,
        "accelerometer": {
            "x": imu_data["accel"][0],
            "y": imu_data["accel"][1],
            "z": imu_data["accel"][2],
        },
        "gyroscope": {
            "x": imu_data["gyro"][0],
            "y": imu_data["gyro"][1],
            "z": imu_data["gyro"][2],
        },
        "orientation": {
            "pitch_deg": imu_data["pitch"],
            "roll_deg": imu_data["roll"],
        },
        "loop_closure": is_loop_closure,
    }
    imu_path = os.path.join(d455_dir, f"{prefix}_imu.json")
    with open(imu_path, "w") as f:
        json.dump(imu_json, f, indent=2)

    return rgb_path, depth_path, imu_path


# ---------------------------------------------------------------------------
# Main capture flow
# ---------------------------------------------------------------------------

def main():
    import pyrealsense2 as rs

    # Scan name
    if len(sys.argv) > 1:
        scan_name = sys.argv[1]
    else:
        scan_name = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    scan_path = SCAN_DIR / scan_name
    bag_path = scan_path / "bag"
    d455_path = scan_path / "d455"
    os.makedirs(scan_path, exist_ok=True)
    os.makedirs(d455_path, exist_ok=True)
    # Don't pre-create bag_path — ros2 bag record -o creates it

    print(f"=== Fusion Capture: {scan_name} ===")
    print(f"Output: {scan_path}")
    print()

    # Track subprocesses for cleanup
    driver_proc = None
    slam_proc = None
    bag_proc = None

    def cleanup():
        print("\nCleaning up...")
        if bag_proc:
            sigint_process_group(bag_proc)
            try:
                bag_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                kill_process_group(bag_proc)
        kill_process_group(slam_proc)
        kill_process_group(driver_proc)

    try:
        # ---- Step 1: Start LiDAR pipeline ----
        input("Press ENTER to start LiDAR + D455 capture...")
        print()

        # Setup network (may fail if already configured or no sudo — that's OK)
        print("[1/6] Setting up network...")
        try:
            subprocess.run(
                ["sudo", "-n", "ip", "addr", "add", f"{HOST_IP}/24", "dev", NET_IFACE],
                capture_output=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, Exception) as e:
            print(f"  Network setup skipped ({e.__class__.__name__}), may already be configured")

        # Kill old processes
        print("[2/6] Cleaning up old processes...")
        for name in ["livox_ros_driver2", "rko_lio", "rviz"]:
            subprocess.run(["pkill", "-9", "-f", name], capture_output=True)
        time.sleep(3)

        # Start LiDAR driver
        print("[3/6] Starting Livox driver...")
        driver_proc = run_shell("ros2 launch livox_ros_driver2 msg_MID360_launch.py")
        drain_stdout(driver_proc, "DRIVER")
        time.sleep(DRIVER_WAIT)

        # Verify driver topic
        print("  Verifying /livox/lidar topic...")
        try:
            verify = subprocess.run(
                ["bash", "-c", f"{ROS_SETUP} && ros2 topic list"],
                capture_output=True, timeout=10,
            )
            topics = verify.stdout.decode()
            if "/livox/lidar" not in topics:
                print("ERROR: Driver not publishing /livox/lidar")
                cleanup()
                return 1
            print("  Driver streaming OK")
        except subprocess.TimeoutExpired:
            print("ERROR: Timeout verifying driver topics")
            cleanup()
            return 1

        # Start SLAM
        print("[4/6] Starting RKO-LIO SLAM...")
        slam_proc = run_shell(
            f"ros2 launch rko_lio odometry.launch.py config_file:={SLAM_CONFIG} rviz:=false"
        )
        drain_stdout(slam_proc, "SLAM")
        time.sleep(SLAM_WAIT)

        # Start bag recording
        print("[5/6] Starting bag recording...")
        bag_proc = run_shell(
            f"cd {bag_path.parent} && ros2 bag record --topics {BAG_TOPICS} -o bag"
        )
        drain_stdout(bag_proc, "BAG")
        time.sleep(1)

        # ---- Step 2: Initialize D455 ----
        print("[6/6] Initializing D455...")
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.color, D455_W, D455_H, rs.format.rgb8, D455_FPS)
        rs_config.enable_stream(rs.stream.depth, D455_W, D455_H, rs.format.z16, D455_FPS)
        rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, CFG["d455"]["accel_fps"])
        rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, CFG["d455"]["gyro_fps"])

        profile = pipeline.start(rs_config)
        align = rs.align(rs.stream.color)

        # Save intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        intrinsics_data = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "model": str(intrinsics.model),
            "coeffs": list(intrinsics.coeffs),
        }
        with open(d455_path / "intrinsics.json", "w") as f:
            json.dump(intrinsics_data, f, indent=2)
        print(f"  Intrinsics saved: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
              f"cx={intrinsics.ppx:.1f} cy={intrinsics.ppy:.1f}")

        # Warmup for auto-exposure
        print(f"  Warming up ({WARMUP_FRAMES} frames)...")
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        print()
        print("=" * 60)
        print("  RECORDING — Walk with the sensors now!")
        print(f"  Auto-capturing D455 at {CAPTURE_HZ}Hz")
        print("  Press ENTER in terminal to stop (captures loop-closure frame)")
        print("=" * 60)
        print()

        # ---- Step 3: Capture loop ----
        frame_count = 0
        frame1_rgb = None
        frame1_depth_col = None
        ref_pitch = None
        ref_roll = None
        last_accel = (0.0, 9.81, 0.0)
        last_gyro = (0.0, 0.0, 0.0)
        capture_interval = 1.0 / CAPTURE_HZ
        last_capture_time = 0.0
        stop_requested = False

        # Non-blocking stdin check
        import select

        window_name = "Fusion Capture"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        while not stop_requested:
            # Check for ENTER on stdin (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                stop_requested = True
                # Capture one final loop-closure frame below
                break

            # Get D455 frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # IMU
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if accel_frame:
                ad = accel_frame.as_motion_frame().get_motion_data()
                last_accel = (ad.x, ad.y, ad.z)
            if gyro_frame:
                gd = gyro_frame.as_motion_frame().get_motion_data()
                last_gyro = (gd.x, gd.y, gd.z)

            current_pitch, current_roll = calculate_orientation(*last_accel)
            color_image = np.asanyarray(color_frame.get_data())
            depth_colorized = colorize_depth(depth_frame)

            # Auto-capture at configured rate
            now = time.monotonic()
            if now - last_capture_time >= capture_interval:
                last_capture_time = now
                frame_count += 1

                imu_data = {
                    "accel": last_accel,
                    "gyro": last_gyro,
                    "pitch": current_pitch,
                    "roll": current_roll,
                }
                save_d455_frame(str(d455_path), frame_count, color_image, depth_frame, imu_data)

                if frame_count == 1:
                    frame1_rgb = color_image.copy()
                    frame1_depth_col = depth_colorized.copy()
                    ref_pitch = current_pitch
                    ref_roll = current_roll

                if frame_count % 10 == 0:
                    print(f"  D455 frame {frame_count} captured")

            # Live preview
            display = create_grid_display(
                frame1_rgb, frame1_depth_col, color_image, depth_colorized,
                current_pitch, current_roll, ref_pitch, ref_roll, frame_count,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC — abort without loop closure
                print("\nESC pressed — aborting without loop closure")
                break
            elif key == 13:  # ENTER in OpenCV window
                stop_requested = True
                break

        # ---- Step 4: Loop-closure frame + shutdown ----
        if stop_requested:
            # Capture final loop-closure frame
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if color_frame and depth_frame:
                accel_frame = frames.first_or_default(rs.stream.accel)
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                if accel_frame:
                    ad = accel_frame.as_motion_frame().get_motion_data()
                    last_accel = (ad.x, ad.y, ad.z)
                if gyro_frame:
                    gd = gyro_frame.as_motion_frame().get_motion_data()
                    last_gyro = (gd.x, gd.y, gd.z)

                current_pitch, current_roll = calculate_orientation(*last_accel)
                frame_count += 1
                imu_data = {
                    "accel": last_accel,
                    "gyro": last_gyro,
                    "pitch": current_pitch,
                    "roll": current_roll,
                }
                save_d455_frame(str(d455_path), frame_count, np.asanyarray(color_frame.get_data()),
                                depth_frame, imu_data, is_loop_closure=True)
                print(f"  Loop-closure frame {frame_count} captured")

        # Stop D455
        pipeline.stop()
        cv2.destroyAllWindows()

        # Stop LiDAR pipeline
        print("\nStopping bag recording (flushing)...")
        sigint_process_group(bag_proc)
        try:
            bag_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            kill_process_group(bag_proc)
        time.sleep(1)

        print("Stopping SLAM and driver...")
        kill_process_group(slam_proc)
        kill_process_group(driver_proc)
        time.sleep(2)

        # Verify outputs
        bag_files = list(bag_path.rglob("*"))
        bag_size = sum(f.stat().st_size for f in bag_files if f.is_file())
        d455_frames = len(list(d455_path.glob("frame_*_rgb.png")))

        print()
        print("=" * 60)
        print(f"  Capture complete: {scan_name}")
        print(f"  Bag size: {bag_size / 1024 / 1024:.1f} MB")
        print(f"  D455 frames: {d455_frames}")
        print(f"  Output: {scan_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nInterrupted!")
        cleanup()
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        cleanup()
        raise

    return 0


if __name__ == "__main__":
    sys.exit(main())
