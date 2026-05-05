"""Title screen — point cloud hero panel, BARUCH wordmark, prototype serial.

Hero panel renders a real PLY (LIDAR bedroom scan) into an FBO every frame
with a slow auto-orbit, drawn into the Endless-cover-style image-panel slot.
Falls back to the static logo if the PLY can't be loaded.
"""
import math
import time
from pathlib import Path

import moderngl
import numpy as np
from imgui_bundle import imgui
from PIL import Image

from ..state import State, Screen
from ..style import (Fonts, FG, DIM, BG, HAIRLINE,
                     text_centered, push_font, pop_font)


_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "baruch_logo.png"
_logo = {"tex": None, "size": (0, 0)}

# Hero PLY: bundled inside the package at data/hero.ply, with a dev-machine
# fallback so working in-tree still finds it.
_HERO_FALLBACK = Path("/home/kevin/kevin_website/bedroom_definitive_web.ply")
_cloud_state = {"requested": False, "error": ""}
_t0 = time.time()


def _resolve_hero_ply(project_root: Path) -> Path | None:
    for candidate in (project_root / "data" / "hero.ply", _HERO_FALLBACK):
        if candidate.exists():
            return candidate
    return None


def _load_logo():
    if _logo["tex"] is not None:
        return
    try:
        ctx = moderngl.get_context()
        im = Image.open(_LOGO_PATH).convert("RGBA")
        data = np.array(im, dtype=np.uint8)
        data = np.ascontiguousarray(data[::-1])
        tex = ctx.texture(im.size, 4, data.tobytes())
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        _logo["tex"] = tex
        _logo["size"] = im.size
    except Exception as e:
        print(f"[title] logo load failed: {e}", flush=True)
        _logo["tex"] = False


def _ensure_cloud(cloud, project_root: Path):
    """Load the hero PLY into the shared CloudViewer once."""
    if _cloud_state["requested"]:
        return
    _cloud_state["requested"] = True
    hero = _resolve_hero_ply(project_root)
    if hero is None:
        _cloud_state["error"] = "no hero PLY found (looked in data/hero.ply)"
        return
    cloud.load(hero)
    if cloud.loaded:
        cloud.pitch = -0.20
        cloud.point_size = 1.8
        # Dark warm panel (matches the Endless cover's printed photo block)
        cloud.bg = (FG[0], FG[1], FG[2])
        # Full holographic palette — rainbow shifts across the cloud
        cloud.holo_amount = 1.0
    else:
        _cloud_state["error"] = cloud.error


def _measure(text: str, font) -> tuple[float, float]:
    push_font(font)
    sz = imgui.calc_text_size(text)
    pop_font(None)
    return sz.x, sz.y


def draw(state: State, w: int, h: int, cloud):
    a = state.fade.value()
    _load_logo()
    _ensure_cloud(cloud, state.project_root)

    nudge_x = w * 0.012

    # Endless cover order (top→bottom):
    #   image panel → ENDLESS title → Frank Ocean subtitle → foil panel
    # Mapped here:
    #   logo        → BARUCH        → technologies         → cloud panel

    # ── 1. Logo (top, image-panel slot) ─────────────────────────────────────
    logo_bottom = h * 0.04
    if _logo["tex"] not in (None, False):
        iw, ih = _logo["size"]
        target_h = h * 0.20
        target_w = target_h * (iw / ih)
        max_w = w * 0.22
        if target_w > max_w:
            target_w = max_w
            target_h = target_w * (ih / iw)
        logo_x = (w - target_w) * 0.5 - nudge_x
        logo_y = h * 0.04
        imgui.set_cursor_pos((logo_x, logo_y))
        tex_ref = imgui.ImTextureRef(_logo["tex"].glo)
        try:
            imgui.image_with_bg(
                tex_ref,
                imgui.ImVec2(target_w, target_h),
                imgui.ImVec2(0, 0), imgui.ImVec2(1, 1),
                imgui.ImVec4(0, 0, 0, 0),
                imgui.ImVec4(1.0, 1.0, 1.0, a),
            )
        except AttributeError:
            imgui.image(
                tex_ref,
                imgui.ImVec2(target_w, target_h),
                imgui.ImVec2(0, 0), imgui.ImVec2(1, 1),
            )
        logo_bottom = logo_y + target_h

    # ── 2. "BARUCH" big wordmark, tight under the logo ──────────────────────
    word_w, word_h = _measure("BARUCH", Fonts.huge)
    name_y = logo_bottom + h * 0.010
    name_x = (w - word_w) * 0.5 - nudge_x
    imgui.set_cursor_pos((name_x, name_y))
    push_font(Fonts.huge)
    imgui.text_colored(imgui.ImVec4(*FG[:3], a), "BARUCH")
    pop_font(None)

    # ── 3. "technologies" micro-subtitle ────────────────────────────────────
    sub_w, sub_h = _measure("technologies", Fonts.body)
    sub_y = name_y + word_h + h * 0.002
    sub_x = (w - sub_w) * 0.5 - nudge_x
    imgui.set_cursor_pos((sub_x, sub_y))
    push_font(Fonts.body)
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.85), "technologies")
    pop_font(None)

    # ── 4. Prototype serial + edition tag ───────────────────────────────────
    serial = "PROTOTYPE-V03"
    edition = "ED. 01  ·  POINT CLOUD CAPTURE"
    sw_x, _ = _measure(serial, Fonts.mono)
    side_x = w * 0.17
    serial_y = sub_y + sub_h + h * 0.025

    imgui.set_cursor_pos((w - side_x - sw_x, serial_y))
    push_font(Fonts.mono)
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.9), serial)
    pop_font(None)

    imgui.set_cursor_pos((side_x, serial_y))
    push_font(Fonts.mono)
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), edition)
    pop_font(None)

    # ── 5. Hero panel — live point cloud (bigger square, just above press start) ─
    footer_y = h * 0.92
    panel_h = h * 0.42
    panel_w = panel_h  # square
    panel_x = (w - panel_w) * 0.5
    panel_y = footer_y - panel_h - h * 0.02

    if cloud.loaded:
        cloud.yaw = (time.time() - _t0) * 0.10
        tex = cloud.render(int(panel_w), int(panel_h))
        if tex is not None:
            imgui.set_cursor_pos((panel_x, panel_y))
            tex_ref = imgui.ImTextureRef(tex.glo)
            try:
                imgui.image_with_bg(
                    tex_ref,
                    imgui.ImVec2(panel_w, panel_h),
                    imgui.ImVec2(0, 1), imgui.ImVec2(1, 0),  # FBO Y-flip
                    imgui.ImVec4(0, 0, 0, 0),
                    imgui.ImVec4(1.0, 1.0, 1.0, a),
                )
            except AttributeError:
                imgui.image(
                    tex_ref,
                    imgui.ImVec2(panel_w, panel_h),
                    imgui.ImVec2(0, 1), imgui.ImVec2(1, 0),
                )
            dl = imgui.get_window_draw_list()
            dl.add_rect(
                imgui.ImVec2(panel_x, panel_y),
                imgui.ImVec2(panel_x + panel_w, panel_y + panel_h),
                imgui.get_color_u32(imgui.ImVec4(FG[0], FG[1], FG[2], 0.55 * a)),
                rounding=0.0,
                thickness=1.0,
            )

    # ── 6. Footer ───────────────────────────────────────────────────────────
    imgui.set_cursor_pos((0, footer_y))
    text_centered("press start", font=Fonts.body, color=(*DIM[:3], a))

    # ── invisible click target ──────────────────────────────────────────────
    imgui.set_cursor_pos((0, 0))
    if imgui.invisible_button("##start", imgui.ImVec2(w, h)):
        state.transition(Screen.DASHBOARD)
    if imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.space):
        state.transition(Screen.DASHBOARD)
