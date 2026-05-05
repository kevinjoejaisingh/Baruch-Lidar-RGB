"""Bag-pick screen — choose an existing scan dir, run the pipeline on it."""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, ACCENT, HAIRLINE, text_centered
from .. import pipeline, library


def draw(state: State, w: int, h: int):
    a = state.fade.value()
    imgui.set_cursor_pos_y(h * 0.10)
    text_centered("process a bag.", font=Fonts.title, color=(*FG[:3], a))

    imgui.set_cursor_pos_y(h * 0.18)
    text_centered("pick a scan directory", font=Fonts.body, color=(*DIM[:3], a * 0.7))

    list_x = w * 0.20
    list_y = h * 0.28
    list_w = w * 0.60
    list_h = h * 0.55

    imgui.set_cursor_pos((list_x, list_y))
    imgui.begin_child("##scans", imgui.ImVec2(list_w, list_h), False)

    cands = library.list_scan_dirs(state)
    if not cands:
        push_font(Fonts.body)
        imgui.set_cursor_pos((0.0, 16.0))
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a), "no scan directories found")
        pop_font(None)
    else:
        push_font(Fonts.body)
        cur_y = 8.0
        for p in cands:
            row_h = 56.0
            label = f"{p.parent.name}/{p.name}"
            has_lidar = (p / "lidar_cloud.ply").exists()
            has_d455 = (p / "d455").exists() or any(p.glob("frame_*_rgb.png"))
            sub = []
            if has_lidar: sub.append("lidar.ply ✓")
            else:         sub.append("needs lidar processing")
            if has_d455:  sub.append("d455 frames ✓")
            else:         sub.append("no d455 frames")
            sub_text = " · ".join(sub)

            p_screen = imgui.get_cursor_screen_pos()
            dl = imgui.get_window_draw_list()
            dl.add_line(
                imgui.ImVec2(p_screen.x, p_screen.y + row_h - 1),
                imgui.ImVec2(p_screen.x + list_w - 16, p_screen.y + row_h - 1),
                imgui.get_color_u32(imgui.ImVec4(*HAIRLINE[:3], 0.35 * a)),
                1.0,
            )
            imgui.set_cursor_pos((4.0, cur_y + 8.0))
            imgui.text_colored(imgui.ImVec4(*FG[:3], a), label)
            imgui.set_cursor_pos((4.0, cur_y + 30.0))
            imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.8), sub_text)
            imgui.set_cursor_pos((list_w - 80.0, cur_y + 18.0))
            imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.6), "process →")
            imgui.set_cursor_pos((0.0, cur_y))
            if imgui.invisible_button(f"##s_{p}", imgui.ImVec2(list_w - 16, row_h - 4)):
                pipeline.run_replay(state, p)
                state.transition(Screen.PROCESS)
            cur_y += row_h + 4.0
            imgui.dummy(imgui.ImVec2(0.0, row_h + 4.0))
        pop_font(None)

    imgui.end_child()

    # back to dashboard
    imgui.set_cursor_pos((0, h - 48))
    text_centered("← back", font=Fonts.body, color=(*DIM[:3], a * 0.7))
    imgui.set_cursor_pos((w * 0.5 - 60, h - 52))
    if imgui.invisible_button("##back", imgui.ImVec2(120, 30)) or imgui.is_key_pressed(imgui.Key.escape):
        state.transition(Screen.DASHBOARD)
