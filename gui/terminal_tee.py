# gui/terminal_tee.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Callable, Optional


class TerminalTee:
    """
    A lightweight stdout/stderr tee:
      - Writes to the original stream (terminal)
      - Also forwards text to a callback (e.g., Qt signal emit)

    Notes:
      - Buffers until newline for cleaner GUI logs.
      - Treats '\\r' as a newline to avoid "stuck" progress lines.
    """

    def __init__(self, original_stream, emit_line: Callable[[str], None]):
        self._orig = original_stream
        self._emit = emit_line
        self._buf = ""

    @property
    def encoding(self) -> str:
        return getattr(self._orig, "encoding", "utf-8") or "utf-8"

    def _emit_buffered_lines(self) -> None:
        while "\n" in self._buf:
            line, rest = self._buf.split("\n", 1)
            self._buf = rest
            try:
                self._emit(line)
            except Exception:
                # Never let GUI logging break the program.
                pass

    def write(self, s) -> int:
        if s is None:
            return 0
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return 0

        # 1) terminal
        try:
            self._orig.write(s)
        except Exception:
            pass

        # 2) gui (line-buffered)
        try:
            text = s.replace("\r\n", "\n").replace("\r", "\n")
            self._buf += text
            # Prevent unbounded growth on streams that never send newlines.
            if len(self._buf) > 10_000:
                try:
                    self._emit(self._buf)
                except Exception:
                    pass
                self._buf = ""
            self._emit_buffered_lines()
        except Exception:
            pass

        return len(s)

    def flush(self) -> None:
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            if self._buf:
                self._emit(self._buf)
                self._buf = ""
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return bool(getattr(self._orig, "isatty", lambda: False)())
        except Exception:
            return False

    def fileno(self) -> Optional[int]:
        try:
            fn = getattr(self._orig, "fileno", None)
            if fn is None:
                return None
            return int(fn())
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass

    def writelines(self, lines) -> None:
        try:
            for line in lines:
                self.write(line)
        except Exception:
            pass

