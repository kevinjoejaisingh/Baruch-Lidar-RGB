"""Process screen — calibration-block progress + stage name + tiny log tail."""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import (push_font, pop_font, Fonts, FG, DIM, ACCENT, HAIRLINE,
                     text_centered, draw_calibration_progress)


def draw(state: State, w: int, h: int):
    a = state.fade.value()

    imgui.set_cursor_pos_y(h * 0.28)
    text_centered(state.pipeline_stage or "preparing", font=Fonts.title, color=(*FG[:3], a))

    # caption: percentage + step
    pct = int(state.pipeline_progress * 100)
    imgui.set_cursor_pos_y(h * 0.38)
    text_centered(f"{pct:03d} %  ·  step {min(4, int(state.pipeline_progress*4)+1)} of 4",
                  font=Fonts.body, color=(*DIM[:3], a * 0.85))

    # ── calibration-block progress bar ─────────────────────────────────────
    bar_w = w * 0.62
    bar_x = (w - bar_w) * 0.5
    bar_y = h * 0.46
    bar_h = 18.0
    draw_calibration_progress(bar_x, bar_y, bar_w, bar_h, state.pipeline_progress, alpha=a)

    # log tail (mono, very dim)
    imgui.set_cursor_pos((bar_x, bar_y + 60))
    push_font(Fonts.mono)
    tail = state.pipeline_log[-6:]
    for line in tail:
        if len(line) > 120: line = line[:120] + "…"
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), line)
    pop_font(None)

    if state.pipeline_done:
        from . import dashboard as _dash
        _dash.refresh()  # so the new recording appears in the list
        if state.pipeline_error:
            imgui.set_cursor_pos_y(h * 0.78)
            text_centered("error", font=Fonts.title, color=(0.85, 0.42, 0.38, a))
            imgui.set_cursor_pos_y(h * 0.84)
            text_centered(state.pipeline_error[:140], font=Fonts.body, color=(*DIM[:3], a))
            imgui.set_cursor_pos_y(h * 0.92)
            text_centered("← back to dashboard", font=Fonts.body, color=(*DIM[:3], a * 0.8))
            imgui.set_cursor_pos((w * 0.5 - 140, h * 0.92))
            if imgui.invisible_button("##back", imgui.ImVec2(280, 30)) or imgui.is_key_pressed(imgui.Key.escape):
                state.transition(Screen.DASHBOARD)
        else:
            imgui.set_cursor_pos_y(h * 0.78)
            text_centered("open viewer →", font=Fonts.title, color=(*ACCENT[:3], a))
            imgui.set_cursor_pos((w * 0.3, h * 0.78))
            if imgui.invisible_button("##open", imgui.ImVec2(w * 0.4, 80)) or imgui.is_key_pressed(imgui.Key.enter):
                state.transition(Screen.VIEWER)
            imgui.set_cursor_pos_y(h * 0.92)
            text_centered("← back to dashboard", font=Fonts.body, color=(*DIM[:3], a * 0.7))
            imgui.set_cursor_pos((w * 0.5 - 140, h * 0.92))
            if imgui.invisible_button("##dash", imgui.ImVec2(280, 30)) or imgui.is_key_pressed(imgui.Key.escape):
                state.transition(Screen.DASHBOARD)
