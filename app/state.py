"""Global app state + screen state machine + fade controller."""
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import time
import yaml


class Screen(Enum):
    TITLE = auto()
    DASHBOARD = auto()
    CONNECT = auto()
    RECORD = auto()
    BAG_PICK = auto()
    PROCESS = auto()
    VIEWER = auto()


@dataclass
class Fade:
    """Slow cross-fade controller. 0 = invisible, 1 = visible."""
    duration: float = 0.9
    t_started: float = field(default_factory=time.time)
    direction: int = 1  # +1 fade in, -1 fade out

    def value(self) -> float:
        dt = time.time() - self.t_started
        v = max(0.0, min(1.0, dt / self.duration))
        return v if self.direction == 1 else (1.0 - v)

    def done(self) -> bool:
        return (time.time() - self.t_started) >= self.duration

    def reset(self, direction: int = 1):
        self.direction = direction
        self.t_started = time.time()


@dataclass
class State:
    screen: Screen = Screen.TITLE
    fade: Fade = field(default_factory=Fade)
    next_screen: Screen | None = None

    # sensor status
    d455_status: str = "idle"   # idle | connecting | ok | fail
    d455_detail: str = ""
    lidar_status: str = "idle"
    lidar_detail: str = ""

    # pipeline
    scan_dir: Path | None = None
    pipeline_stage: str = ""
    pipeline_progress: float = 0.0
    pipeline_log: list[str] = field(default_factory=list)
    pipeline_done: bool = False
    pipeline_error: str = ""
    output_ply: Path | None = None

    # config
    cfg: dict = field(default_factory=dict)
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # x11 — set by main.py after window creation, used by godot embed
    parent_xid: int = 0

    # screens append (x, y, w, h, alpha) tuples; main.py renders them
    # after imgui (so imgui text overlays the foil) and before grain.
    holo_strips: list = field(default_factory=list)

    def transition(self, target: Screen):
        if self.next_screen is None and target != self.screen:
            self.next_screen = target
            self.fade.reset(direction=-1)

    def tick(self):
        if self.next_screen is not None and self.fade.done() and self.fade.direction == -1:
            self.screen = self.next_screen
            self.next_screen = None
            self.fade.reset(direction=+1)

    def load_config(self):
        cfg_path = self.project_root / "config.yaml"
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)
