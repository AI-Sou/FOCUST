# gui/subprocess_runner.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

from PyQt5.QtCore import QThread, pyqtSignal


class SubprocessRunner(QThread):
    """
    Run a subprocess and stream its output line-by-line.

    - Emits `line` for each output line (stdout + stderr merged)
    - Emits `finished_rc` with the process return code
    """

    line = pyqtSignal(str)
    finished_rc = pyqtSignal(int)

    def __init__(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.cmd = list(cmd or [])
        self.cwd = cwd
        self.env = env or {}
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def run(self) -> None:
        if not self.cmd:
            self.finished_rc.emit(127)
            return

        merged_env = os.environ.copy()
        merged_env.update({k: str(v) for k, v in (self.env or {}).items()})
        # Ensure consistent decoding when the child is Python.
        merged_env.setdefault("PYTHONIOENCODING", "utf-8")
        merged_env.setdefault("PYTHONUTF8", "1")

        try:
            self.line.emit("[CMD] " + " ".join(self.cmd))
        except Exception:
            pass

        try:
            self._proc = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            try:
                self.line.emit(f"[ERROR] Failed to start process: {type(e).__name__}: {e}")
            except Exception:
                pass
            self.finished_rc.emit(127)
            return

        assert self._proc is not None
        try:
            stream = self._proc.stdout
            if stream is not None:
                for raw in stream:
                    if self._stop_requested:
                        break
                    line = raw.rstrip("\n")
                    try:
                        self.line.emit(line)
                    except Exception:
                        pass
        except Exception as e:
            try:
                self.line.emit(f"[ERROR] Process output error: {type(e).__name__}: {e}")
            except Exception:
                pass
        finally:
            try:
                if self._stop_requested and self._proc.poll() is None:
                    try:
                        self._proc.terminate()
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            rc = int(self._proc.wait(timeout=10))
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                rc = int(self._proc.wait(timeout=5))
            except Exception:
                rc = 1
        self.finished_rc.emit(rc)
