"""Connect screen — D455 + LiDAR status side by side, hairline between."""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, ACCENT, text_centered, hairline
from .. import sensors


_started = {"v": False}


def _status_color(status: str):
    if status == "ok":   return (0.62, 0.76, 0.55, 1.0)   # muted green
    if status == "fail": return (0.85, 0.42, 0.38, 1.0)   # muted red
    return DIM


def draw(state: State, w: int, h: int):
    if not _started["v"]:
        sensors.start_probes(state)
        _started["v"] = True

    imgui.set_cursor_pos_y(h * 0.18)
    text_centered("connect.", font=Fonts.title, color=(*FG[:3], state.fade.value()))

    col_w = w * 0.36
    gap = w * 0.04
    left_x = (w - (col_w * 2 + gap)) * 0.5
    right_x = left_x + col_w + gap
    y = h * 0.42

    def _column(x: float, label: str, status: str, detail: str):
        imgui.set_cursor_pos((x, y))
        push_font(Fonts.body)
        imgui.text_colored(imgui.ImVec4(*DIM), label)
        pop_font(None)

        imgui.set_cursor_pos((x, y + 26))
        push_font(Fonts.title)
        line = {"idle": "—", "connecting": "connecting…", "ok": "ok", "fail": "fail"}.get(status, status)
        col = _status_color(status)
        imgui.text_colored(imgui.ImVec4(col[0], col[1], col[2], col[3] * state.fade.value()), line)
        pop_font(None)

        imgui.set_cursor_pos((x, y + 96))
        push_font(Fonts.body)
        imgui.text_colored(imgui.ImVec4(*DIM), detail or " ")
        pop_font(None)

    _column(left_x, "d455 — depth + rgb", state.d455_status, state.d455_detail)
    _column(right_x, "livox mid-360 — geometry", state.lidar_status, state.lidar_detail)

    # vertical hairline
    dl = imgui.get_window_draw_list()
    cx = left_x + col_w + gap * 0.5
    col = imgui.get_color_u32(imgui.ImVec4(0.22, 0.22, 0.21, 0.6 * state.fade.value()))
    dl.add_line(imgui.ImVec2(cx, y - 8), imgui.ImVec2(cx, y + 130), col, 1.0)

    # continue
    a = state.fade.value()
    both_ok = state.d455_status == "ok" and state.lidar_status == "ok"
    any_fail = state.d455_status == "fail" or state.lidar_status == "fail"

    if both_ok:
        imgui.set_cursor_pos((0, h * 0.78))
        text_centered("start recording →", font=Fonts.body, color=(*ACCENT[:3], a))
        imgui.set_cursor_pos((w * 0.5 - 100, h * 0.78))
        if imgui.invisible_button("##cont", imgui.ImVec2(200, 30)):
            state.transition(Screen.RECORD)
        if imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.space):
            state.transition(Screen.RECORD)
    elif any_fail:
        imgui.set_cursor_pos((0, h * 0.78))
        text_centered("retry", font=Fonts.body, color=(*DIM[:3], a))
        imgui.set_cursor_pos((w * 0.5 - 60, h * 0.78))
        if imgui.invisible_button("##retry", imgui.ImVec2(120, 30)):
            state.d455_status = "idle"; state.lidar_status = "idle"
            _started["v"] = False

    # back to dashboard
    imgui.set_cursor_pos((0, h - 48))
    text_centered("← back", font=Fonts.body, color=(*DIM[:3], a * 0.7))
    imgui.set_cursor_pos((w * 0.5 - 60, h - 52))
    if imgui.invisible_button("##back", imgui.ImVec2(120, 30)) or imgui.is_key_pressed(imgui.Key.escape):
        state.d455_status = "idle"; state.lidar_status = "idle"
        _started["v"] = False
        state.transition(Screen.DASHBOARD)
