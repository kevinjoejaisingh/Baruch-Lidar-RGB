"""Holographic point-cloud strip — Poisson-disk-scattered iridescent dots
filling a rect, color cycling in time so it reads like a hologram.

Same call-site API as before: drawn between imgui pixels (below) and
grain (above). Bake-once Poisson sample makes the spacing organic
(no visible rows/cols), each point gets its own phase + size jitter so
neighboring dots show distinct holographic stops.
"""
import time

import moderngl
import numpy as np


VERT = """
#version 330
uniform vec2  u_fb;          // framebuffer size (px)
uniform vec2  u_pos;         // strip top-left (px)
uniform vec2  u_size;        // strip size (px)
uniform float u_point_size;  // base px
uniform float u_time;
uniform float u_angle;       // rotation around strip center (radians)

in vec2 in_local;            // 0..1 inside strip
in vec3 in_attr;             // (size_mul, phase_off, twinkle)
out vec2  v_local;
out float v_phase;
out float v_twinkle;

void main() {
    // rotate around strip center (small angle for off-axis composition)
    vec2 c = in_local - 0.5;
    float ca = cos(u_angle);
    float sa = sin(u_angle);
    vec2 rot = vec2(c.x * ca - c.y * sa, c.x * sa + c.y * ca) + 0.5;

    vec2 px = u_pos + rot * u_size;
    vec2 ndc;
    ndc.x = (px.x / u_fb.x) * 2.0 - 1.0;
    ndc.y = 1.0 - (px.y / u_fb.y) * 2.0;
    gl_Position  = vec4(ndc, 0.0, 1.0);
    gl_PointSize = u_point_size * in_attr.x;
    v_local   = rot;
    v_phase   = in_attr.y;
    v_twinkle = in_attr.z;
}
"""

FRAG = """
#version 330
in vec2  v_local;
in float v_phase;
in float v_twinkle;
out vec4 frag;
uniform float u_time;
uniform float u_alpha;

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
    // crisp round point — tight AA so dots stay discrete
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;
    float aa = smoothstep(0.25, 0.21, r2);

    // per-point phase = positional ramp + per-point random offset + time drift
    float phase = v_local.x * 1.4
                + v_phase
                + sin(v_local.y * 4.0 + u_time * 0.45) * 0.18
                + u_time * 0.20;
    vec3 col = holo(phase);

    // subtle per-point twinkle (intensity wobble)
    float tw = 0.85 + 0.15 * sin(u_time * 1.7 + v_twinkle * 6.2831);

    // vertical falloff (gentle on paper — keep colors saturated)
    float yfade = smoothstep(0.0, 0.18, v_local.y)
                * smoothstep(0.0, 0.18, 1.0 - v_local.y);
    col *= (0.78 + yfade * 0.22) * tw;

    // horizontal edge fade
    float xfade = smoothstep(0.0, 0.04, v_local.x)
                * smoothstep(0.0, 0.04, 1.0 - v_local.x);

    frag = vec4(col, u_alpha * aa * xfade);
}
"""


# Domain aspect for the Poisson bake. Most strips are wide (title hero
# is ~14:1, caption bands are ~64:1), so we bake in 16:1 — close enough
# that pixel spacing ends up roughly isotropic on the typical hero strip.
_DOMAIN_W = 16.0
_DOMAIN_H = 1.0
# Min distance between points (in domain units). Sets density.
# Bridson typically fills to ~50% of the theoretical max packing,
# so r ≈ 0.085 lands around ~1300 points across the 16×1 domain.
_R = 0.085


def _poisson_disk(width: float, height: float, r: float,
                  k: int = 30, seed: int = 42) -> np.ndarray:
    """Bridson's Poisson-disk sampling on [0,width)×[0,height)."""
    rng = np.random.default_rng(seed)
    cell = r / np.sqrt(2.0)
    nx = int(np.ceil(width / cell))
    ny = int(np.ceil(height / cell))
    grid = -np.ones((nx, ny), dtype=np.int64)
    pts: list[tuple[float, float]] = []
    active: list[int] = []

    p0 = (float(rng.random()) * width, float(rng.random()) * height)
    pts.append(p0)
    active.append(0)
    grid[int(p0[0] / cell), int(p0[1] / cell)] = 0
    r2 = r * r

    while active:
        ai = int(rng.integers(0, len(active)))
        idx = active[ai]
        cx, cy = pts[idx]
        found = False
        for _ in range(k):
            theta = float(rng.random()) * 2.0 * np.pi
            rad = r * (1.0 + float(rng.random()))   # uniform in [r, 2r)
            x = cx + rad * np.cos(theta)
            y = cy + rad * np.sin(theta)
            if x < 0.0 or x >= width or y < 0.0 or y >= height:
                continue
            gx, gy = int(x / cell), int(y / cell)
            ok = True
            for ix in range(max(0, gx - 2), min(nx, gx + 3)):
                if not ok:
                    break
                for iy in range(max(0, gy - 2), min(ny, gy + 3)):
                    j = grid[ix, iy]
                    if j >= 0:
                        dx = pts[j][0] - x
                        dy = pts[j][1] - y
                        if dx * dx + dy * dy < r2:
                            ok = False
                            break
            if ok:
                pts.append((x, y))
                active.append(len(pts) - 1)
                grid[gx, gy] = len(pts) - 1
                found = True
                break
        if not found:
            # remove from active list
            active[ai] = active[-1]
            active.pop()

    return np.asarray(pts, dtype=np.float32)


class Holo:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        # ── Poisson sample once at startup ─────────────────────────────────
        raw = _poisson_disk(_DOMAIN_W, _DOMAIN_H, _R)
        # Normalize x into [0,1]; y already in [0,1]
        raw[:, 0] /= _DOMAIN_W

        rng = np.random.default_rng(7)
        n = len(raw)
        size_mul = (0.75 + 0.5 * rng.random(n)).astype("f4")   # 0.75..1.25
        phase_off = rng.random(n).astype("f4")                  # 0..1
        twinkle = rng.random(n).astype("f4")                    # 0..1
        attrs = np.stack([size_mul, phase_off, twinkle], axis=1)

        data = np.concatenate([raw, attrs], axis=1).astype("f4")
        self.vbo = ctx.buffer(data.tobytes())
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f 3f", "in_local", "in_attr")],
        )
        self.n_points = n
        self.t0 = time.time()

    def draw(self, x: float, y: float, w: float, h: float,
             fb_w: int, fb_h: int, alpha: float = 1.0, angle: float = 0.0):
        if w <= 0 or h <= 0 or fb_w <= 0 or fb_h <= 0:
            return
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Base point size scales with strip height; per-point size_mul
        # in the VBO adds 0.75–1.25× variation
        point_size = max(3.0, min(8.0, h / 9.0))

        t = time.time() - self.t0
        self.prog["u_fb"].value = (float(fb_w), float(fb_h))
        self.prog["u_pos"].value = (float(x), float(y))
        self.prog["u_size"].value = (float(w), float(h))
        self.prog["u_point_size"].value = float(point_size)
        self.prog["u_time"].value = t
        self.prog["u_alpha"].value = float(alpha)
        self.prog["u_angle"].value = float(angle)
        self.vao.render(moderngl.POINTS)
