import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple


def _contains_cjk(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0x20000 <= code <= 0x2A6DF  # Extension B
            or 0x2A700 <= code <= 0x2B73F  # Extension C
            or 0x2B740 <= code <= 0x2B81F  # Extension D
            or 0x2B820 <= code <= 0x2CEAF  # Extension E/F
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        ):
            return True
    return False


def _iter_candidate_font_paths() -> Tuple[Path, ...]:
    env = (
        os.environ.get("FOCUST_CJK_FONT")
        or os.environ.get("CJK_FONT_PATH")
        or os.environ.get("FONT_PATH")
    )
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())

    this_file = Path(__file__).resolve()
    for parent in (this_file.parent,) + tuple(this_file.parents):
        candidates.extend(
            [
                parent / "assets" / "fonts" / "NotoSansSC-Regular.ttf",
                parent / "assets" / "fonts" / "NotoSansSC-Regular.otf",
                parent / "assets" / "fonts" / "NotoSansSC-Regular.ttc",
            ]
        )
        # Common alternative names (in case users replace the bundled font)
        candidates.extend(
            [
                parent / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf",
                parent / "assets" / "fonts" / "SourceHanSansSC-Regular.otf",
                parent / "assets" / "fonts" / "wqy-zenhei.ttc",
            ]
        )
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        unique.append(c)
    return tuple(unique)


@lru_cache(maxsize=1)
def get_cjk_font_path() -> Optional[str]:
    for candidate in _iter_candidate_font_paths():
        try:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        except Exception:
            continue
    return None


def ensure_qt_cjk_font() -> Optional[str]:
    """
    Load the bundled CJK font into Qt and set it as application default.

    Returns:
        Loaded font family name if successful, otherwise None.
    """
    try:
        from PyQt5.QtGui import QFont, QFontDatabase  # type: ignore
        from PyQt5.QtWidgets import QApplication  # type: ignore
    except Exception:
        return None

    font_path = get_cjk_font_path()
    if not font_path:
        return None

    try:
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            return None
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            return None
        family = families[0]

        app = QApplication.instance()
        if app is not None:
            app.setFont(QFont(family))
        return family
    except Exception:
        return None


def cv2_put_text(
    image,
    text: str,
    org,
    font_face=None,
    font_scale: float = 0.5,
    color=(255, 255, 255),
    thickness: int = 1,
    line_type=None,
):
    """
    A drop-in replacement for `cv2.putText` that renders CJK text via Pillow + bundled font.
    Falls back to OpenCV when Pillow is unavailable or when text contains no CJK chars.
    """
    import cv2  # local import for environments without cv2 during docs build

    if text is None:
        return image
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""

    if not _contains_cjk(text):
        return cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX if font_face is None else font_face,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )

    font_path = get_cjk_font_path()
    if not font_path:
        return cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX if font_face is None else font_face,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX if font_face is None else font_face,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )

    if image is None:
        return image

    # OpenCV uses BGR; PIL uses RGB.
    try:
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            image_bgr = image
    except Exception:
        image_bgr = image

    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return cv2.putText(
            image,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX if font_face is None else font_face,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )

    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    font_size = max(10, int(font_scale * 32))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    try:
        text_w, text_h = draw.textsize(text, font=font)  # Pillow < 10 compatibility
    except Exception:
        try:
            text_w, text_h = font.getsize(text)
        except Exception:
            text_w, text_h = (len(text) * font_size, font_size)

    x, y = int(org[0]), int(org[1])
    y_top = y - int(text_h)  # approximate OpenCV baseline behavior

    try:
        b, g, r = int(color[0]), int(color[1]), int(color[2])
    except Exception:
        b, g, r = 255, 255, 255
    fill = (r, g, b)

    stroke_width = max(0, int(round(thickness / 2)))
    try:
        draw.text(
            (x, y_top),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )
    except TypeError:
        # Older Pillow versions without stroke support
        draw.text((x, y_top), text, font=font, fill=fill)

    out_rgb = np.asarray(pil_img)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    try:
        # Modify in-place when possible to match cv2.putText semantics.
        if (
            hasattr(image, "shape")
            and hasattr(out_bgr, "shape")
            and len(image.shape) >= 2
            and image.shape[:2] == out_bgr.shape[:2]
        ):
            if len(image.shape) == 2:
                out_gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
                image[:, :] = out_gray
                return image
            if image.shape[2] == 4:
                out_bgra = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA)
                image[:, :, :] = out_bgra
                return image
            image[:, :, :] = out_bgr
            return image
    except Exception:
        pass

    return out_bgr


def measure_text(text: str, font_scale: float = 0.5, thickness: int = 1) -> Tuple[int, int]:
    """
    Measure text width/height in pixels. Uses Pillow for CJK if available.
    """
    import cv2

    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""

    if not _contains_cjk(text):
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)

    font_path = get_cjk_font_path()
    if not font_path:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h)

    font_size = max(10, int(font_scale * 32))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(img)
    try:
        w, h = draw.textsize(text, font=font)
    except Exception:
        try:
            w, h = font.getsize(text)
        except Exception:
            w, h = (len(text) * font_size, font_size)
    return int(w), int(h)


def ensure_matplotlib_cjk_font() -> Optional[str]:
    """
    Register the bundled CJK font for Matplotlib and update rcParams to avoid "□" glyphs.

    Returns:
        The resolved font family name if registered, otherwise None.
    """
    font_path = get_cjk_font_path()
    if not font_path:
        return None

    try:
        import matplotlib
        from matplotlib import font_manager as fm  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    family = None
    try:
        fm.fontManager.addfont(font_path)
        family = fm.FontProperties(fname=font_path).get_name()
    except Exception:
        family = None

    try:
        sans = list(plt.rcParams.get("font.sans-serif", []))
        if family and family not in sans:
            plt.rcParams["font.sans-serif"] = [family] + sans
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    return family
