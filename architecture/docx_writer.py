from __future__ import annotations

import datetime as _dt
import html
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Union


def _xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def write_simple_docx(
    output_path: Union[str, Path],
    title: str,
    paragraphs: Iterable[str],
    *,
    created_at: Optional[_dt.datetime] = None,
) -> Path:
    """
    Write a minimal .docx without external dependencies.

    This is intentionally simple: it produces a document with paragraphs only.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = created_at or _dt.datetime.now()
    created_iso = created_at.replace(microsecond=0).isoformat()

    paras: List[str] = [str(title).strip(), f"Generated at: {created_iso}", ""]
    for p in paragraphs:
        if p is None:
            continue
        paras.append(str(p))

    def _p(text: str) -> str:
        # WordprocessingML paragraph with a single run.
        # Preserve explicit line breaks by splitting.
        parts = str(text).splitlines() or [""]
        runs = []
        for i, part in enumerate(parts):
            if i:
                runs.append("<w:br/>")
            runs.append(f"<w:t xml:space=\"preserve\">{_xml_escape(part)}</w:t>")
        return f"<w:p><w:r>{''.join(runs)}</w:r></w:p>"

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
        "<w:body>"
        + "".join(_p(x) for x in paras)
        + "<w:sectPr/>"
        "</w:body>"
        "</w:document>"
    )

    content_types_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
        "<Override PartName=\"/word/document.xml\" "
        "ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
        "</Types>"
    )

    rels_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        "<Relationship Id=\"rId1\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
        "Target=\"word/document.xml\"/>"
        "</Relationships>"
    )

    # Write zip
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("word/document.xml", document_xml)

    return output_path


__all__ = ["write_simple_docx"]
