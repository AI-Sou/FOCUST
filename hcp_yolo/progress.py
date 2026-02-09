from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def iter_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
) -> Iterable[T]:
    """
    Progress iterator wrapper.

    - If `tqdm` is available, uses a real progress bar.
    - Otherwise, prints a simple ASCII progress line (still shows percentage).
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore[assignment]

    if tqdm is not None:
        return tqdm(
            iterable,
            total=total,
            desc=desc or None,
            unit=unit,
            leave=leave,
            ascii=True,  # safer on Windows consoles
            dynamic_ncols=True,
            file=sys.stdout,
            disable=False,  # force show even when piped (e.g. through `tee`)
            mininterval=0.5,
        )

    return _simple_iter_progress(iterable, total=total, desc=desc, unit=unit)


def _simple_iter_progress(
    iterable: Iterable[T],
    *,
    total: Optional[int],
    desc: str,
    unit: str,
) -> Iterator[T]:
    start = time.time()
    bar_len = 28
    is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
    every = 1
    if (not is_tty) and total and total > 0:
        # When stdout is piped, carriage-return style "updating line" often won't show.
        # Print a line every ~2% (max 50 lines) to guarantee visible progress.
        every = max(1, int(total / 50))

    def _render(i: int):
        if total and total > 0:
            frac = min(1.0, max(0.0, i / float(total)))
            filled = int(round(frac * bar_len))
            bar = "=" * filled + "-" * (bar_len - filled)
            pct = int(round(frac * 100))
            elapsed = time.time() - start
            msg = f"{desc} [{bar}] {pct:3d}% {i}/{total} {unit}  ({elapsed:.1f}s)"
        else:
            elapsed = time.time() - start
            msg = f"{desc} {i} {unit}  ({elapsed:.1f}s)"
        if is_tty:
            sys.stdout.write("\r" + msg + " " * 4)
            sys.stdout.flush()
        else:
            print(msg, flush=True)

    i = 0
    if desc:
        if is_tty:
            _render(0)
    for item in iterable:
        i += 1
        if desc:
            if is_tty:
                _render(i)
            else:
                if (i % every) == 0 or (total and i >= total):
                    _render(i)
        yield item
    if desc:
        if is_tty:
            sys.stdout.write("\n")
            sys.stdout.flush()
