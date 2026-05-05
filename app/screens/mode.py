"""Mode screen — two giant lowercase words: 'record.' and 'replay.'"""
from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, text_centered


def _giant_choice(label: str, x: float, y: float, w: float, h: float, alpha: float, dim_other: bool) -> bool:
    color = DIM if dim_other else FG
    imgui.set_cursor_pos((x, y))
    push_font(Fonts.huge)
    sz = imgui.calc_text_size(label)
    imgui.set_cursor_pos((x + (w - sz.x) * 0.5, y + (h - sz.y) * 0.5))
    imgui.text_colored(imgui.ImVec4(color[0], color[1], color[2], color[3] * alpha), label)
    pop_font(None)

    imgui.set_cursor_pos((x, y))
    return imgui.invisible_button(f"##{label}", imgui.ImVec2(w, h))


def draw(state: State, w: int, h: int):
    a = state.fade.value()
    box_h = h * 0.4
    box_y = h * 0.30

    # detect hover on each half to dim the other
    mx, my = imgui.get_mouse_pos().x, imgui.get_mouse_pos().y
    in_left  = (0      <= mx <= w*0.5) and (box_y <= my <= box_y + box_h)
    in_right = (w*0.5  <= mx <= w)     and (box_y <= my <= box_y + box_h)

    if _giant_choice("record.", 0,       box_y, w*0.5, box_h, a, dim_other=in_right):
        state.transition(Screen.RECORD)
    if _giant_choice("replay.", w*0.5,   box_y, w*0.5, box_h, a, dim_other=in_left):
        state.transition(Screen.REPLAY)

    # divider between
    dl = imgui.get_window_draw_list()
    col = imgui.get_color_u32(imgui.ImVec4(0.22, 0.22, 0.21, 0.5 * a))
    dl.add_line(imgui.ImVec2(w*0.5, box_y + 24), imgui.ImVec2(w*0.5, box_y + box_h - 24), col, 1.0)

    # tiny caption
    imgui.set_cursor_pos((0, h * 0.78))
    text_centered("choose a path", font=Fonts.body, color=(*DIM[:3], a * 0.8))
