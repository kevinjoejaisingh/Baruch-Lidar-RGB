"""Recording screen — captures via capture.py subprocess. 'stop' commits + processes."""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, ACCENT, text_centered, hairline
from .. import pipeline


_proc = {"v": None}


def draw(state: State, w: int, h: int):
    a = state.fade.value()

    if _proc["v"] is None:
        _proc["v"] = pipeline.start_recording(state)

    imgui.set_cursor_pos_y(h * 0.18)
    text_centered("recording.", font=Fonts.title, color=(*FG[:3], a))

    imgui.set_cursor_pos_y(h * 0.40)
    text_centered(state.scan_dir.name if state.scan_dir else "", font=Fonts.body, color=(*DIM[:3], a))

    # log tail
    log_y = h * 0.50
    imgui.set_cursor_pos((w * 0.10, log_y))
    push_font(Fonts.mono)
    tail = state.pipeline_log[-8:]
    for line in tail:
        if len(line) > 120: line = line[:120] + "…"
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.8), line)
    pop_font(None)

    # stop — full-width hairline button
    imgui.set_cursor_pos((0, h * 0.78))
    text_centered("stop", font=Fonts.title, color=(*ACCENT[:3], a))
    imgui.set_cursor_pos((w * 0.4, h * 0.78))
    if imgui.invisible_button("##stop", imgui.ImVec2(w * 0.2, 80)) or imgui.is_key_pressed(imgui.Key.enter):
        if _proc["v"] is not None:
            pipeline.stop_recording_and_process(state, _proc["v"])
            _proc["v"] = None
            state.transition(Screen.PROCESS)

    # cancel
    imgui.set_cursor_pos((0, h - 48))
    text_centered("← cancel", font=Fonts.body, color=(*DIM[:3], a * 0.7))
    imgui.set_cursor_pos((w * 0.5 - 80, h - 52))
    if imgui.invisible_button("##cancel", imgui.ImVec2(160, 30)) or imgui.is_key_pressed(imgui.Key.escape):
        if _proc["v"] is not None:
            try:
                _proc["v"].terminate()
            except Exception:
                pass
            _proc["v"] = None
        state.transition(Screen.DASHBOARD)
