"""Embedded point cloud viewer: moderngl renders to FBO, ImGui shows it as an image."""
import math
import time
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui


VERT = """
#version 330
uniform mat4 u_mvp;
uniform float u_point_size;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_col;
out vec3 v_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    gl_PointSize = u_point_size;
    v_col = in_col;
    v_pos = in_pos;
}
"""

FRAG = """
#version 330
in vec3 v_col;
in vec3 v_pos;
out vec4 frag;
uniform float u_time;
uniform float u_holo;     // 0 = original colors, 1 = full holographic
uniform float u_extent;   // cloud bounding extent (for phase scaling)

vec3 holo(float t) {
    // four-stop loop: cyan → magenta → gold → silver
    vec3 c1 = vec3(0.55, 0.85, 0.95);
    vec3 c2 = vec3(0.95, 0.72, 0.86);
    vec3 c3 = vec3(0.95, 0.86, 0.58);
    vec3 c4 = vec3(0.86, 0.86, 0.92);
    float p = fract(t);
    if (p < 0.25) return mix(c1, c2, p / 0.25);
    if (p < 0.50) return mix(c2, c3, (p - 0.25) / 0.25);
    if (p < 0.75) return mix(c3, c4, (p - 0.50) / 0.25);
    return mix(c4, c1, (p - 0.75) / 0.25);
}

void main() {
    vec2 d = gl_PointCoord - 0.5;
    if (dot(d, d) > 0.25) discard;

    vec3 col = v_col;
    if (u_holo > 0.0) {
        // Phase = position-based ramp + slow time drift, normalized by
        // the cloud extent so the visual frequency stays consistent
        // regardless of source-PLY scale.
        float ext = max(u_extent, 0.001);
        float phase = (v_pos.x + v_pos.y * 0.7 + v_pos.z * 0.4) / ext * 1.4
                    + u_time * 0.28;
        vec3 hcol = holo(phase);
        col = mix(col, hcol, u_holo);
    }
    frag = vec4(col, 1.0);
}
"""


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1
    return m


def _look_at(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f /= np.linalg.norm(f) + 1e-9
    s = np.cross(f, up); s /= np.linalg.norm(s) + 1e-9
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class CloudViewer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = None
        self.n_points = 0
        self.fbo = None
        self.tex = None
        self.depth = None
        self.fbo_size = (0, 0)
        self.point_size = 2.0

        # camera state — orbit around target
        self.target = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.pitch = -0.2
        self.distance = 5.0
        self.cloud_path: Path | None = None
        self.loaded = False
        self.error = ""
        # FBO clear color — title sets this to dark warm to match the
        # Endless-cover B&W image-panel feel; viewer keeps it black.
        self.bg = (0.0, 0.0, 0.0)
        # Holographic overlay (0 = original colors, 1 = full iridescence).
        # Title screen turns this on; viewer keeps it 0 for true colors.
        self.holo_amount = 0.0
        self.extent = 1.0
        self._t0 = time.time()

    def load(self, ply_path: Path):
        # release any previous GPU buffers first
        if self.vao is not None:
            try:
                self.vao.release()
            except Exception:
                pass
            self.vao = None
        self.loaded = False
        self.n_points = 0
        self.cloud_path = ply_path
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(ply_path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            if len(pts) == 0:
                self.error = "empty point cloud"
                return
            cols = np.asarray(pcd.colors, dtype=np.float32)
            if cols.size == 0 or len(cols) != len(pts):
                cols = np.full_like(pts, 0.5)

            center = pts.mean(axis=0)
            pts -= center
            extent = float(np.linalg.norm(pts, axis=1).max())
            self.target = np.zeros(3, dtype=np.float32)
            self.distance = max(0.5, extent * 1.6)
            self.extent = extent

            data = np.concatenate([pts, cols], axis=1).astype("f4")
            vbo = self.ctx.buffer(data.tobytes())
            self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f", "in_pos", "in_col")])
            self.n_points = len(pts)
            self.loaded = True
            self.error = ""
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.loaded = False

    def _ensure_fbo(self, w: int, h: int):
        if self.fbo is not None and self.fbo_size == (w, h):
            return
        if self.fbo is not None:
            self.fbo.release(); self.tex.release(); self.depth.release()
        self.tex = self.ctx.texture((w, h), 4)
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth = self.ctx.depth_renderbuffer((w, h))
        self.fbo = self.ctx.framebuffer(color_attachments=[self.tex], depth_attachment=self.depth)
        self.fbo_size = (w, h)

    def _camera_pos(self) -> np.ndarray:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        d = self.distance
        return self.target + np.array([d * cp * sy, d * sp, d * cp * cy], dtype=np.float32)

    def handle_input(self, dt: float, hovered: bool):
        if not hovered or not self.loaded:
            return
        io = imgui.get_io()
        # mouse drag look
        if imgui.is_mouse_dragging(imgui.MouseButton_.left, 1.0):
            d = imgui.get_mouse_drag_delta(imgui.MouseButton_.left, 1.0)
            self.yaw   -= d.x * 0.005
            self.pitch -= d.y * 0.005
            self.pitch = max(-1.5, min(1.5, self.pitch))
            imgui.reset_mouse_drag_delta(imgui.MouseButton_.left)
        # scroll zoom
        wheel = io.mouse_wheel
        if wheel != 0:
            self.distance *= (0.9 ** wheel)
            self.distance = max(0.1, min(500.0, self.distance))
        # WASD pan in view plane
        speed = self.distance * 0.6 * dt
        f = (self.target - self._camera_pos())
        f[1] = 0; n = np.linalg.norm(f)
        if n > 1e-6: f /= n
        right = np.cross(f, np.array([0, 1, 0], dtype=np.float32))
        if imgui.is_key_down(imgui.Key.w): self.target += f * speed
        if imgui.is_key_down(imgui.Key.s): self.target -= f * speed
        if imgui.is_key_down(imgui.Key.a): self.target -= right * speed
        if imgui.is_key_down(imgui.Key.d): self.target += right * speed
        if imgui.is_key_down(imgui.Key.space): self.target[1] += speed
        if imgui.is_key_down(imgui.Key.left_shift): self.target[1] -= speed

    def render(self, w: int, h: int):
        if not self.loaded or self.vao is None:
            return None
        self._ensure_fbo(w, h)
        self.fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.clear(self.bg[0], self.bg[1], self.bg[2], 1.0)

        eye = self._camera_pos()
        view = _look_at(eye, self.target, [0, 1, 0])
        proj = _perspective(45.0, max(w, 1) / max(h, 1), 0.05, 2000.0)
        mvp = proj @ view
        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.prog["u_point_size"].value = float(self.point_size)
        self.prog["u_time"].value = float(time.time() - self._t0)
        self.prog["u_holo"].value = float(self.holo_amount)
        self.prog["u_extent"].value = float(self.extent)
        self.vao.render(moderngl.POINTS)

        self.ctx.screen.use()
        return self.tex
