"""BARUCH — single-window prototype UI.

Run from project root:
    python3 -m app.main
"""
import sys
from pathlib import Path

import glfw
import moderngl
import OpenGL.GL as gl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from .state import State, Screen
from . import style as st
from .grain import Grain
from .cloud_viewer import CloudViewer
from .screens import title, dashboard, connect, record, replay, process, viewer as viewer_screen


WINDOW_W = 1600
WINDOW_H = 1000
TITLE = "BARUCH"


def _gl_window():
    if not glfw.init():
        raise RuntimeError("glfw init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)
    win = glfw.create_window(WINDOW_W, WINDOW_H, TITLE, None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("glfw create_window failed")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    return win


def main():
    win = _gl_window()
    ctx = moderngl.create_context()

    imgui.create_context()
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard.value
    st.load_fonts()
    st.apply_style()

    impl = GlfwRenderer(win)

    state = State()
    state.load_config()
    try:
        state.parent_xid = int(glfw.get_x11_window(win))
    except Exception:
        state.parent_xid = 0

    grain = Grain(ctx, intensity=0.05)
    cloud = CloudViewer(ctx)

    while not glfw.window_should_close(win):
        glfw.poll_events()
        impl.process_inputs()

        w, h = glfw.get_framebuffer_size(win)
        ctx.viewport = (0, 0, w, h)
        ctx.clear(*st.BG_RGB, 1.0)

        imgui.new_frame()

        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_resize.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_nav_focus.value
            | imgui.WindowFlags_.no_background.value
        )
        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(imgui.ImVec2(float(w), float(h)))
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(0, 0))
        imgui.begin("##root", None, flags)
        imgui.pop_style_var()
        # reserve full-window content region so set_cursor_pos calls stay in bounds
        imgui.dummy(imgui.ImVec2(float(w) - 2, float(h) - 2))
        imgui.set_cursor_pos(imgui.ImVec2(0.0, 0.0))

        state.tick()

        def _open_recording(p):
            state.output_ply = p
            state.transition(Screen.VIEWER)

        s = state.screen
        if   s == Screen.TITLE:     title.draw(state, w, h, cloud)
        elif s == Screen.DASHBOARD: dashboard.draw(state, w, h, _open_recording)
        elif s == Screen.CONNECT:   connect.draw(state, w, h)
        elif s == Screen.RECORD:    record.draw(state, w, h)
        elif s == Screen.BAG_PICK:  replay.draw(state, w, h)
        elif s == Screen.PROCESS:   process.draw(state, w, h)
        elif s == Screen.VIEWER:    viewer_screen.draw(state, w, h, cloud)

        # print-proof chrome on every screen except the embedded viewer
        # (where it would float over the godot region awkwardly)
        if s != Screen.VIEWER:
            st.draw_chrome(w, h, alpha=state.fade.value())

        imgui.end()
        imgui.render()
        impl.render(imgui.get_draw_data())

        # film grain on top
        grain.draw(w, h)

        glfw.swap_buffers(win)

    # tear down any embedded subprocess (godot)
    try:
        from .screens.viewer import _state as _viewer_state
        if _viewer_state.get("embed") is not None:
            _viewer_state["embed"].stop()
    except Exception:
        pass

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main() or 0)
