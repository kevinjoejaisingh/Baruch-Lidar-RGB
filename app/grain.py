"""Paper-fiber grain overlay — multi-octave value noise drawn over the page.

Replaces the high-frequency film grain. Paper grain is low-frequency,
warm-toned, near-static (slow drift only) so the page feels printed
rather than televised. Drawn after ImGui each frame.
"""
import time

import moderngl
import numpy as np


VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAG = """
#version 330
in vec2 v_uv;
out vec4 frag;
uniform float u_time;
uniform float u_intensity;
uniform vec2  u_res;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        v += amp * vnoise(p);
        p *= 2.07;
        amp *= 0.5;
    }
    return v;
}

void main() {
    vec2 px = v_uv * u_res;

    // Slow drift — gives the page a barely-perceptible breath without
    // looking like animated TV static.
    vec2 drift = vec2(u_time * 0.02, u_time * 0.013);

    // Two scales of fibers: fine specks and broader clumps
    float fine    = fbm(px * 0.45  + drift * 11.0);
    float clumps  = fbm(px * 0.045 + drift);
    float n = (fine - 0.5) * 0.7 + (clumps - 0.5) * 0.5;

    // Soft vignette darkens edges slightly, like the cover scan
    vec2 c = v_uv - 0.5;
    float vign = smoothstep(0.95, 0.30, 1.0 - dot(c, c) * 1.4);

    // Warm sepia tint on the noise so paper feels off-white, not gray
    vec3 tint = vec3(0.55, 0.42, 0.28);
    vec3 col = tint * n * u_intensity * (0.6 + 0.4 * vign);

    // Use plain alpha-blend with signed luminance: positive n darkens,
    // negative n lightens — paper texture wobble around the page color.
    frag = vec4(col, abs(n) * u_intensity * (0.85 + 0.15 * vign));
}
"""


class Grain:
    def __init__(self, ctx: moderngl.Context, intensity: float = 0.18):
        self.ctx = ctx
        self.intensity = intensity
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        verts = np.array([-1, -1, 3, -1, -1, 3], dtype="f4")
        self.vbo = ctx.buffer(verts.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f", "in_pos")])
        self.t0 = time.time()

    def draw(self, w: int, h: int):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.prog["u_time"].value = time.time() - self.t0
        self.prog["u_intensity"].value = self.intensity
        self.prog["u_res"].value = (float(w), float(h))
        self.vao.render(moderngl.TRIANGLES)
