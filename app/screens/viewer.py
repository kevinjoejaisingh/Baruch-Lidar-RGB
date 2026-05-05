"""Viewer screen — top hairline bar (back/filename) + embedded Godot below.

The Godot subprocess is reparented into the GLFW X11 window so it lives
inside the app. The reserved top strip is the only area where ImGui pixels
are visible — everything below is the Godot child window.
"""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, ACCENT, HAIRLINE
from .. import godot_embed


BAR_H = 44


_state = {
    "embed": None,            # GodotEmbed | None
    "loaded": None,           # Path | None
    "error": "",
}


def _get_embed():
    if _state["embed"] is None:
        _state["embed"] = godot_embed.GodotEmbed()
    return _state["embed"]


def _exit_to_dashboard(state: State):
    e = _state["embed"]
    if e is not None:
        e.stop()
    _state["loaded"] = None
    _state["error"] = ""
    state.output_ply = None
    state.transition(Screen.DASHBOARD)


def draw(state: State, w: int, h: int, _unused_viewer):
    a = state.fade.value()
    embed = _get_embed()

    GODOT = state.project_root / "godot_viewer" / "build" / "PointCloudViewer.x86_64"
    GODOT_PROJ = state.project_root / "godot_viewer"

    # ── Top bar (one big clickable strip) ──────────────────────────────────
    # Detect hover so the bar visibly highlights when the cursor enters it.
    mp = imgui.get_mouse_pos()
    bar_hovered = (0 <= mp.x <= w) and (0 <= mp.y <= BAR_H)

    dl = imgui.get_window_draw_list()
    if bar_hovered:
        dl.add_rect_filled(
            imgui.ImVec2(0, 0),
            imgui.ImVec2(w, BAR_H),
            imgui.get_color_u32(imgui.ImVec4(0.10, 0.10, 0.10, a)),
        )

    push_font(Fonts.body)
    imgui.set_cursor_pos((24.0, 14.0))
    label_color = ACCENT if bar_hovered else FG
    imgui.text_colored(imgui.ImVec4(*label_color[:3], a), "← back to dashboard")

    # exit hint in the middle
    hint = "click anywhere on this bar  ·  q / f10 / esc inside viewer"
    sz = imgui.calc_text_size(hint)
    imgui.set_cursor_pos((max(260.0, (w - sz.x) * 0.5), 16.0))
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.65), hint)

    # filename caption (right)
    if state.output_ply is not None:
        sub = state.output_ply.name
        sz2 = imgui.calc_text_size(sub)
        imgui.set_cursor_pos((max(280.0, w - sz2.x - 24.0), 14.0))
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), sub)
    pop_font(None)

    # full-width invisible button covers the bar
    imgui.set_cursor_pos((0.0, 0.0))
    if imgui.invisible_button("##back", imgui.ImVec2(float(w), float(BAR_H))):
        _exit_to_dashboard(state)
        return

    # hairline divider under bar
    dl.add_line(
        imgui.ImVec2(20.0, BAR_H - 1.0),
        imgui.ImVec2(w - 20.0, BAR_H - 1.0),
        imgui.get_color_u32(imgui.ImVec4(*HAIRLINE[:3], 0.6 * a)),
        1.0,
    )

    # ── Embedded Godot below the bar ────────────────────────────────────────
    region_x = 0
    region_y = BAR_H
    region_w = max(int(w), 320)
    region_h = max(int(h - BAR_H), 240)

    have_cli = godot_embed.GodotEmbed._find_godot_cli() is not None
    if not GODOT.exists() and not have_cli:
        push_font(Fonts.title)
        imgui.set_cursor_pos((24.0, BAR_H + 24.0))
        imgui.text_colored(imgui.ImVec4(0.85, 0.42, 0.38, a), "godot not available")
        pop_font(None)
        push_font(Fonts.body)
        imgui.set_cursor_pos((24.0, BAR_H + 80.0))
        imgui.text_colored(
            imgui.ImVec4(*DIM[:3], a),
            "no godot binary at godot_viewer/build/ and no `godot` cli on PATH",
        )
        pop_font(None)
        if imgui.is_key_pressed(imgui.Key.escape):
            _exit_to_dashboard(state)
        return

    # Launch / reload if the target file changed
    if state.output_ply is not None and _state["loaded"] != state.output_ply:
        embed.stop()
        ok = embed.start(
            GODOT, state.output_ply, state.parent_xid,
            region_x, region_y, region_w, region_h,
            project_dir=GODOT_PROJ,
        )
        if ok:
            _state["loaded"] = state.output_ply
            _state["error"] = ""
        else:
            _state["error"] = embed.error or "failed to embed godot"

    # Keep geometry in sync as the parent resizes
    if embed.is_alive():
        embed.update_geometry(region_x, region_y, region_w, region_h)

    # If Godot was closed by the user, fall back to dashboard
    if embed.exited and _state["loaded"] is not None:
        _exit_to_dashboard(state)
        return

    # Embed launch error message (only visible because top region is uncovered)
    if _state["error"]:
        push_font(Fonts.body)
        imgui.set_cursor_pos((24.0, BAR_H + 24.0))
        imgui.text_colored(imgui.ImVec4(0.85, 0.42, 0.38, a), _state["error"])
        pop_font(None)

    if imgui.is_key_pressed(imgui.Key.escape):
        _exit_to_dashboard(state)
