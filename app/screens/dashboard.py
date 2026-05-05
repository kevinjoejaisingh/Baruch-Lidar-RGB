"""Dashboard — central hub. Recordings list + two actions."""
import time
from pathlib import Path

from imgui_bundle import imgui

from ..state import State, Screen
from ..style import push_font, pop_font, Fonts, FG, DIM, ACCENT, HAIRLINE, text_centered, hairline
from .. import library


_cache = {"t": 0.0, "items": []}
_pending_delete = {"path": None}
_last_error = {"msg": ""}


def _items(state: State):
    now = time.time()
    if now - _cache["t"] > 1.5:
        try:
            _cache["items"] = library.list_recordings(state)
        except Exception:
            _cache["items"] = []
        _cache["t"] = now
    return _cache["items"]


def refresh():
    _cache["t"] = 0.0


def draw(state: State, w: int, h: int, on_open):
    a = state.fade.value()

    # ── Header ──────────────────────────────────────────────────────────────
    EDGE = 76.0  # leave breathing room for the corner registration marks

    # back-to-landing button (hover changes color, click returns to title)
    back_label = "← back"
    push_font(Fonts.body)
    back_w = imgui.calc_text_size(back_label).x
    pop_font(None)
    mp = imgui.get_mouse_pos()
    back_hot = (EDGE - 4 <= mp.x <= EDGE + back_w + 8) and (28 <= mp.y <= 52)
    back_color = FG if back_hot else DIM
    imgui.set_cursor_pos((EDGE, 32.0))
    push_font(Fonts.body)
    imgui.text_colored(imgui.ImVec4(*back_color[:3], a), back_label)
    pop_font(None)
    imgui.set_cursor_pos((EDGE - 4, 28.0))
    esc_pressed = imgui.is_key_pressed(imgui.Key.escape)
    has_modal = (_pending_delete["path"] is not None) or _last_error["msg"]
    if (imgui.invisible_button("##back_to_title", imgui.ImVec2(back_w + 12, 24))
            or (esc_pressed and not has_modal)):
        state.transition(Screen.TITLE)
        return

    # wordmark on the right side of the header
    push_font(Fonts.body)
    word_w = imgui.calc_text_size("BARUCH").x
    pop_font(None)
    imgui.set_cursor_pos((w - EDGE - word_w, 32.0))
    push_font(Fonts.body)
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a), "BARUCH")
    pop_font(None)
    imgui.set_cursor_pos((EDGE, 64.0))
    dl = imgui.get_window_draw_list()
    col = imgui.get_color_u32(imgui.ImVec4(*HAIRLINE[:3], 0.5 * a))
    dl.add_line(imgui.ImVec2(EDGE, 64), imgui.ImVec2(w - EDGE, 64), col, 1.0)

    # caption under the header rule — plain ink on paper, no foil
    push_font(Fonts.body)
    imgui.set_cursor_pos((EDGE, 76.0))
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.85),
                       "POINT CLOUD CATALOG")
    pop_font(None)

    # ── Two-column layout ───────────────────────────────────────────────────
    left_x = 64.0
    left_w = w * 0.32
    right_x = w * 0.42
    right_w = w - right_x - 64.0
    top_y = 130.0

    # ── Left: actions ───────────────────────────────────────────────────────
    push_font(Fonts.body)
    imgui.set_cursor_pos((left_x, top_y))
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), "actions")
    pop_font(None)

    def _action(label: str, sub: str, y: float, target: Screen) -> None:
        push_font(Fonts.title)
        imgui.set_cursor_pos((left_x, y))
        imgui.text_colored(imgui.ImVec4(*FG[:3], a), label)
        pop_font(None)
        push_font(Fonts.body)
        imgui.set_cursor_pos((left_x, y + 56))
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.8), sub)
        pop_font(None)
        imgui.set_cursor_pos((left_x - 8, y - 8))
        if imgui.invisible_button(f"##act_{label}", imgui.ImVec2(left_w, 84)):
            state.transition(target)

    _action("new recording.", "connect d455 + livox, capture a scene", top_y + 44, Screen.CONNECT)
    _action("process a bag.",  "pick an existing scan, run the pipeline", top_y + 180, Screen.BAG_PICK)

    # vertical hairline between columns
    cx = right_x - 28.0
    dl.add_line(imgui.ImVec2(cx, top_y), imgui.ImVec2(cx, h - 64), col, 1.0)

    # ── Right: recordings list ──────────────────────────────────────────────
    push_font(Fonts.body)
    imgui.set_cursor_pos((right_x, top_y))
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), "recordings")
    pop_font(None)

    items = _items(state)
    push_font(Fonts.body)
    imgui.set_cursor_pos((right_x + right_w - 100, top_y))
    imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.5), f"{len(items)} files")
    pop_font(None)

    list_y = top_y + 44.0
    list_h = h - list_y - 80.0
    imgui.set_cursor_pos((right_x, list_y))
    imgui.begin_child("##rec", imgui.ImVec2(right_w, list_h), False)

    if not items:
        push_font(Fonts.body)
        imgui.set_cursor_pos((0.0, 16.0))
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a), "no recordings yet — process a bag to get started")
        pop_font(None)
    else:
        push_font(Fonts.body)
        cur_y = 8.0
        DEL_W = 90.0   # right-side hit area for delete
        for rec in items:
            row_h = 56.0
            pending = _pending_delete["path"] == rec.path

            # underline hairline
            p_screen = imgui.get_cursor_screen_pos()
            dl2 = imgui.get_window_draw_list()
            dl2.add_line(
                imgui.ImVec2(p_screen.x, p_screen.y + row_h - 1),
                imgui.ImVec2(p_screen.x + right_w - 16, p_screen.y + row_h - 1),
                imgui.get_color_u32(imgui.ImVec4(*HAIRLINE[:3], 0.35 * a)),
                1.0,
            )

            # text
            imgui.set_cursor_pos((4.0, cur_y + 8.0))
            imgui.text_colored(imgui.ImVec4(*FG[:3], a), rec.label)
            imgui.set_cursor_pos((4.0, cur_y + 30.0))
            imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.8), f"{rec.when}    {rec.size_mb:,.1f} mb")

            # right side: open + delete labels
            open_label = "cancel" if pending else "open →"
            del_label  = "really delete?" if pending else "delete"
            del_color  = (0.85, 0.42, 0.38, a) if pending else (*DIM[:3], a * 0.5)

            imgui.set_cursor_pos((right_w - DEL_W - 110.0, cur_y + 18.0))
            imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.6), open_label)
            imgui.set_cursor_pos((right_w - DEL_W - 16.0, cur_y + 18.0))
            imgui.text_colored(imgui.ImVec4(*del_color), del_label)

            # open / cancel hit area (left+middle)
            imgui.set_cursor_pos((0.0, cur_y))
            if imgui.invisible_button(f"##rec_{rec.path}", imgui.ImVec2(right_w - DEL_W - 24.0, row_h - 4)):
                if pending:
                    _pending_delete["path"] = None
                else:
                    on_open(rec.path)

            # delete hit area (right slice)
            imgui.set_cursor_pos((right_w - DEL_W - 20.0, cur_y))
            if imgui.invisible_button(f"##del_{rec.path}", imgui.ImVec2(DEL_W + 4.0, row_h - 4)):
                if pending:
                    try:
                        rec.path.unlink()
                        _last_error["msg"] = ""
                    except Exception as e:
                        _last_error["msg"] = f"delete failed: {e}"
                    _pending_delete["path"] = None
                    refresh()
                else:
                    _pending_delete["path"] = rec.path

            cur_y += row_h + 4.0
            imgui.dummy(imgui.ImVec2(0.0, row_h + 4.0))
        pop_font(None)

    imgui.end_child()

    # ── Footer caption ──────────────────────────────────────────────────────
    push_font(Fonts.body)
    imgui.set_cursor_pos((EDGE, h - 36.0))
    if _last_error["msg"]:
        imgui.text_colored(imgui.ImVec4(0.85, 0.42, 0.38, a), _last_error["msg"])
    elif _pending_delete["path"] is not None:
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.7), "click 'really delete?' again to confirm · esc to cancel")
    else:
        imgui.text_colored(imgui.ImVec4(*DIM[:3], a * 0.6), "enter — new recording · click delete twice to remove")
    pop_font(None)

    if imgui.is_key_pressed(imgui.Key.escape):
        _pending_delete["path"] = None
        _last_error["msg"] = ""
    if imgui.is_key_pressed(imgui.Key.enter):
        state.transition(Screen.CONNECT)
