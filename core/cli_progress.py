# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional, TextIO

_ACTIVE_PROGRESS_BAR = None


def set_active_progress_bar(bar) -> None:
    global _ACTIVE_PROGRESS_BAR
    _ACTIVE_PROGRESS_BAR = bar


def get_active_progress_bar():
    return _ACTIVE_PROGRESS_BAR


@dataclass
class _RenderState:
    last_value: Optional[int] = None
    last_text: str = ""
    last_render_ts: float = 0.0


class CliProgressBar:
    """
    Lightweight ASCII progress bar for CLI usage.

    - In TTY: redraws in-place with carriage returns.
    - Non-TTY (e.g. nohup): prints sparse progress lines to avoid huge logs.
    """

    def __init__(
        self,
        *,
        width: int = 30,
        stream: Optional[TextIO] = None,
        label: str = "Progress",
        enabled: bool = True,
        non_tty_step: int = 5,
    ) -> None:
        self.width = max(10, int(width))
        self.stream = stream or sys.stdout
        self.label = str(label or "Progress")
        self.enabled = bool(enabled)
        self.non_tty_step = max(1, int(non_tty_step))
        try:
            self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        except Exception:
            self.is_tty = False
        self._state = _RenderState()

    def clear(self) -> None:
        if not self.enabled or not self.is_tty:
            return
        line = self._state.last_text
        if not line:
            return
        try:
            self.stream.write("\r" + (" " * len(line)) + "\r")
            self.stream.flush()
        except Exception:
            pass

    def redraw(self) -> None:
        if not self.enabled:
            return
        if not self.is_tty:
            return
        if self._state.last_text:
            try:
                line = self._state.last_text
                self.stream.write("\r" + line)
                self.stream.flush()
            except Exception:
                pass

    def _format_line(self, value: int, extra: str = "") -> str:
        value = max(0, min(100, int(value)))
        filled = int(round(self.width * (value / 100.0)))
        bar = ("#" * filled) + ("-" * (self.width - filled))
        suffix = f" {extra}" if extra else ""
        return f"{self.label}: [{bar}] {value:3d}%{suffix}"

    def update(self, value: int, *, extra: str = "") -> None:
        if not self.enabled:
            return

        v = max(0, min(100, int(value)))

        # Non-TTY: only print every N% to reduce log volume.
        if not self.is_tty:
            if self._state.last_value is None:
                pass
            else:
                if v not in (0, 100) and abs(v - self._state.last_value) < self.non_tty_step:
                    return
            line = self._format_line(v, extra=extra)
            self._state.last_value = v
            self._state.last_text = line
            self._state.last_render_ts = time.time()
            try:
                self.stream.write(line + "\n")
                self.stream.flush()
            except Exception:
                pass
            return

        # TTY: render in-place.
        line = self._format_line(v, extra=extra)
        if self._state.last_text == line:
            return

        prev_len = len(self._state.last_text or "")
        self._state.last_value = v
        self._state.last_text = line
        self._state.last_render_ts = time.time()
        try:
            pad = ""
            if prev_len > len(line):
                pad = " " * (prev_len - len(line))
            self.stream.write("\r" + line + pad)
            self.stream.flush()
        except Exception:
            pass

    def close(self, *, final_newline: bool = True) -> None:
        if not self.enabled:
            return
        if self.is_tty and final_newline:
            try:
                self.stream.write("\n")
                self.stream.flush()
            except Exception:
                pass


def ensure_progress_bar_safe_logging() -> None:
    """
    Ensure CLI logging does not mangle an active in-place progress bar.

    Adds a lightweight filter to root logger + handlers that clears the active bar
    before any log record is written.
    """

    class _ProgressBarClearingFilter(logging.Filter):
        __progress_bar_filter__ = True

        def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
            try:
                bar = get_active_progress_bar()
                if bar:
                    bar.clear()
            except Exception:
                pass
            return True

    root = logging.getLogger()
    try:
        for f in root.filters:
            if getattr(f, "__progress_bar_filter__", False):
                break
        else:
            root.addFilter(_ProgressBarClearingFilter())
    except Exception:
        pass

    try:
        for h in list(root.handlers or []):
            if any(getattr(f, "__progress_bar_filter__", False) for f in (getattr(h, "filters", []) or [])):
                continue
            h.addFilter(_ProgressBarClearingFilter())
    except Exception:
        pass
