"""Endless-inspired ImGui style: paper, near-black ink, lowercase, flat."""
from pathlib import Path
from imgui_bundle import imgui

FONT_DIR = Path(__file__).parent / "fonts"

# ── Paper palette (cream sheet, dark warm ink) ──────────────────────────────
BG       = (0.930, 0.916, 0.882, 1.0)   # warm cream paper
FG       = (0.090, 0.080, 0.060, 1.0)   # near-black, warm
DIM      = (0.460, 0.430, 0.380, 1.0)   # mid warm-gray
GHOST    = (0.860, 0.842, 0.802, 1.0)   # paper but a step darker (button hover)
HAIRLINE = (0.700, 0.680, 0.620, 1.0)   # rule lines
ACCENT   = (0.090, 0.080, 0.060, 1.0)   # ink, used for emphasis

# convenience for code that needs the bg as an rgb tuple (ctx.clear etc.)
BG_RGB = BG[:3]


class Fonts:
    body = None
    title = None
    huge = None
    mono = None


def _try_font(name: str, size: float):
    """Load font at given size; fall back through user fonts → system fonts → default."""
    io = imgui.get_io()
    p = FONT_DIR / name
    if p.exists():
        return io.fonts.add_font_from_file_ttf(str(p), size)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return io.fonts.add_font_from_file_ttf(c, size)
    return None


def load_fonts():
    Fonts.body = _try_font("InterTight-Regular.ttf", 18)
    Fonts.title = _try_font("InterTight-Light.ttf", 56)
    Fonts.huge = _try_font("InterTight-Light.ttf", 140)
    Fonts.mono = _try_font("JetBrainsMono-Regular.ttf", 13)


def apply_style():
    s = imgui.get_style()
    s.window_rounding = 0
    s.frame_rounding = 0
    s.popup_rounding = 0
    s.scrollbar_rounding = 0
    s.grab_rounding = 0
    s.tab_rounding = 0
    s.window_border_size = 0
    s.frame_border_size = 0
    s.popup_border_size = 0
    s.window_padding = imgui.ImVec2(48, 48)
    s.frame_padding = imgui.ImVec2(12, 8)
    s.item_spacing = imgui.ImVec2(12, 16)
    s.scrollbar_size = 6

    def C(idx, rgba):
        s.set_color_(idx, imgui.ImVec4(*rgba))

    C(imgui.Col_.window_bg.value,       BG)
    C(imgui.Col_.child_bg.value,        BG)
    C(imgui.Col_.popup_bg.value,        BG)
    C(imgui.Col_.text.value,            FG)
    C(imgui.Col_.text_disabled.value,   DIM)
    C(imgui.Col_.button.value,          (0, 0, 0, 0))
    C(imgui.Col_.button_hovered.value,  GHOST)
    C(imgui.Col_.button_active.value,   HAIRLINE)
    C(imgui.Col_.frame_bg.value,        (0, 0, 0, 0))
    C(imgui.Col_.frame_bg_hovered.value, GHOST)
    C(imgui.Col_.frame_bg_active.value, HAIRLINE)
    C(imgui.Col_.border.value,          HAIRLINE)
    C(imgui.Col_.separator.value,       HAIRLINE)
    C(imgui.Col_.header.value,          (0, 0, 0, 0))
    C(imgui.Col_.header_hovered.value,  GHOST)
    C(imgui.Col_.header_active.value,   HAIRLINE)
    C(imgui.Col_.scrollbar_bg.value,    (0, 0, 0, 0))
    C(imgui.Col_.scrollbar_grab.value,  GHOST)
    C(imgui.Col_.scrollbar_grab_hovered.value, DIM)


def draw_chrome(w: int, h: int, alpha: float = 1.0):
    """Print-proof framing: trim marks + corner registration + edge calibration bars.

    All drawn in dark ink on the cream paper.
    """
    dl = imgui.get_window_draw_list()
    ink = imgui.ImVec4(0.18, 0.16, 0.13, 0.65 * alpha)
    col = imgui.get_color_u32(ink)

    # ── Trim marks (solid filled corner squares — guillotine cut markers) ───
    trim = 6
    trim_off = 4
    trim_col = imgui.get_color_u32(imgui.ImVec4(0.10, 0.09, 0.07, 0.85 * alpha))
    for x0, y0 in (
        (trim_off,             trim_off),
        (w - trim_off - trim,  trim_off),
        (trim_off,             h - trim_off - trim),
        (w - trim_off - trim,  h - trim_off - trim),
    ):
        dl.add_rect_filled(
            imgui.ImVec2(x0, y0),
            imgui.ImVec2(x0 + trim, y0 + trim),
            trim_col,
        )

    # ── Corner registration marks (circle + center dot) ─────────────────────
    margin = 26
    r = 6.5
    dot_r = 1.6
    for cx, cy in (
        (margin, margin),
        (w - margin, margin),
        (margin, h - margin),
        (w - margin, h - margin),
    ):
        dl.add_circle(imgui.ImVec2(cx, cy), r, col, 32, 1.0)
        dl.add_circle_filled(imgui.ImVec2(cx, cy), dot_r, col, 12)

    # secondary tiny circle + dot marks, like the album reference
    inner = 60
    r2 = 3.0
    dot_r2 = 1.0
    for cx, cy in (
        (inner, margin), (w - inner, margin),
        (inner, h - margin), (w - inner, h - margin),
        (margin, inner), (w - margin, inner),
        (margin, h - inner), (w - margin, h - inner),
    ):
        dl.add_circle(imgui.ImVec2(cx, cy), r2, col, 16, 1.0)
        dl.add_circle_filled(imgui.ImVec2(cx, cy), dot_r2, col, 8)

    # ── Calibration tone strips on the far left/right edges (printed on paper) ──
    bar_w = 4
    bar_h = 11
    gap = 2
    n = 9
    grays = [0.95, 0.82, 0.66, 0.50, 0.36, 0.22, 0.06, 0.50, 0.82]
    block_total = n * (bar_h + gap) - gap
    cy0 = (h - block_total) * 0.5
    for side_x in (12, w - 12 - bar_w):
        for i in range(n):
            g = grays[i]
            c = imgui.get_color_u32(imgui.ImVec4(g, g, g, 0.85 * alpha))
            y0 = cy0 + i * (bar_h + gap)
            dl.add_rect_filled(
                imgui.ImVec2(side_x, y0),
                imgui.ImVec2(side_x + bar_w, y0 + bar_h),
                c,
            )


def draw_calibration_progress(x: float, y: float, w: float, h: float,
                              progress: float, alpha: float = 1.0,
                              n: int = 14):
    """Stepped grayscale-block progress bar in the album-edge style."""
    dl = imgui.get_window_draw_list()
    progress = max(0.0, min(1.0, progress))
    gap = 3.0
    block_w = (w - gap * (n - 1)) / n
    for i in range(n):
        x0 = x + i * (block_w + gap)
        # active blocks pop bright, inactive ones sit muted
        active = (i + 0.5) / n <= progress
        if active:
            shade = 0.55 + 0.40 * ((i + 1) / n)   # left dim → right bright
            a = 0.95 * alpha
        else:
            shade = 0.18
            a = 0.45 * alpha
        c = imgui.get_color_u32(imgui.ImVec4(shade, shade, shade, a))
        dl.add_rect_filled(imgui.ImVec2(x0, y), imgui.ImVec2(x0 + block_w, y + h), c)


def hairline(width: float = 0, alpha: float = 1.0):
    """Draw a thin horizontal divider."""
    dl = imgui.get_window_draw_list()
    p = imgui.get_cursor_screen_pos()
    avail = imgui.get_content_region_avail().x if width <= 0 else width
    col = imgui.get_color_u32(imgui.ImVec4(HAIRLINE[0], HAIRLINE[1], HAIRLINE[2], HAIRLINE[3] * alpha))
    dl.add_line(imgui.ImVec2(p.x, p.y), imgui.ImVec2(p.x + avail, p.y), col, 1.0)
    imgui.dummy(imgui.ImVec2(avail, 1))


_font_stack: list[bool] = []


def push_font(font):
    if font is not None:
        imgui.push_font(font, 0.0)
        _font_stack.append(True)
    else:
        _font_stack.append(False)


def pop_font(_=None):
    if _font_stack and _font_stack.pop():
        imgui.pop_font()


def text_centered(s: str, font=None, color=None):
    push_font(font)
    sz = imgui.calc_text_size(s)
    avail = imgui.get_content_region_avail().x
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0, (avail - sz.x) * 0.5))
    if color is not None:
        imgui.text_colored(imgui.ImVec4(*color), s)
    else:
        imgui.text_unformatted(s)
    pop_font(font)
