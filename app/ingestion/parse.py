"""Parsing — text, tables as tables, figures as images.

Standard text extraction destroys the two content types that matter most
in this corpus:

  - Tables hold much of the answerable content (which board supports
    which function, which terminal does what). Flattened to a line of
    text the row/column relationship is lost and the answer becomes
    wrong. Extract as structured objects; index both the table and a
    generated prose description of it.

  - Wiring diagrams are spatial. OCR turns them into token soup. Extract
    the image, store it, index its caption and surrounding prose, and
    return the image itself alongside the answer.

Detects missing text layers (scanned, image-only PDFs) and routes those
to the OCR path with a low-confidence flag. Fails cleanly with a readable
reason on corrupt or password-protected files.

Library choice: pdfplumber (MIT) for text, per-word font/geometry and
tables; pypdf (BSD-3) for the open/encryption/corruption gate;
pypdfium2 (BSD-3, via pdfplumber) to raster figure regions. PyMuPDF was
rejected despite being the fastest option: it is AGPL, and this is a
commercial product.

Callout detection is LAYOUT-based, not keyword-based. In this corpus a
safety warning carries no "WARNING:" string. It is a pictogram drawn in
the left margin beside indented text:

    warning -> triangle, emitted as a vector `curve` ~18-48pt square
    note    -> boxed "i", emitted as a vector `rect` of the same size

Both sit left of the callout text and vertically overlap its first line.
That is the only signal present in the body text, so it is the one used.
Verified against the legend page of all seven manuals in Documents/.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber
import pypdf

BlockType = Literal["heading", "paragraph", "callout", "table", "figure"]
CalloutKind = Literal["warning", "note"]

ICON_MIN_PT = 18.0
ICON_MAX_PT = 48.0
ICON_SQUARENESS_PT = 14.0

# A callout's text is indented this far past the page's body margin. The
# margin itself alternates between recto and verso pages in this corpus
# (43pt / 71pt), so it is measured per page, never per document.
CALLOUT_INDENT_PT = 16.0
ICON_TEXT_GAP_PT = 20.0

HEADER_BAND_PT = 56.0
FOOTER_BAND_PT = 56.0

# Below this many characters of extracted text per page, the page has no
# usable text layer and belongs on the OCR path.
MIN_CHARS_PER_PAGE = 120

FIGURE_CAPTION_RE = re.compile(r"^\s*(?:Fig\.?|Figure|Abb\.?|Bild)\s*(\d+)\s*[:.]\s*(.+)$", re.I)
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
DOC_NUMBER_RE = re.compile(r"\b(\d{5}[A-Z]\d(?:\.[A-Z]{2}\d)?)\b")
REVISION_RE = re.compile(r"\b((?:0[1-9]|1[0-2]|xx)\.(?:19|20)\d{2})\b", re.I)
PART_NUMBER_RE = re.compile(r"(?:Part[\s-]*No\.?|Art\.?-?Nr\.?:?)\s*([0-9A-Z]{6,10})", re.I)
DOT_LEADER_RE = re.compile(r"\.{6,}")
PRODUCT_RE = re.compile(r"(Systevo[\w\s+/\-\.]{0,48})")
UNCAPTIONED_FIGURE_LABEL_RE = re.compile(
    r"^\s*(schematic drawing|block diagram|wiring diagram|connection diagram|overview|"
    r"schematic|circuit diagram|terminal assignment)\b",
    re.I,
)

# A frame this large is a diagram panel, not a text rule.
FIGURE_FRAME_MIN_W_PT = 150.0
FIGURE_FRAME_MIN_H_PT = 90.0


class ParseError(Exception):
    """A file could not be parsed. The message is shown to the operator."""


@dataclass
class Block:
    type: BlockType
    text: str
    page: int
    bbox: tuple[float, float, float, float]
    level: int | None = None
    section_number: str | None = None
    callout_kind: CalloutKind | None = None
    ref: str | None = None


@dataclass
class Table:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    header: list[str]
    rows: list[list[str | None]]
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.header) if self.header else (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self) -> str:
        def cell(v: str | None) -> str:
            return "" if v is None else " ".join(str(v).split())

        out = []
        if self.header:
            out.append("| " + " | ".join(cell(h) for h in self.header) + " |")
            out.append("| " + " | ".join("---" for _ in self.header) + " |")
        for r in self.rows:
            out.append("| " + " | ".join(cell(c) for c in r) + " |")
        return "\n".join(out)

    def to_text(self) -> str:
        """One line per row, cells joined by spaces, empty cells dropped.

        This is close to how the row reads in the PDF's own text layer,
        which is what makes a part number searchable: markdown pipes and
        prose scaffolding both sit between the query term and the match
        for lexical search, and a row like "72591D Systevo Electronic
        Module plus with Audio SEM+" is exactly what someone types.
        """
        def cell(v: str | None) -> str:
            return "" if v is None else " ".join(str(v).split())

        lines = []
        if self.header:
            head = " ".join(c for c in (cell(h) for h in self.header) if c)
            if head:
                lines.append(head)
        for r in self.rows:
            line = " ".join(c for c in (cell(v) for v in r) if c)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _subject_column(self) -> int:
        """The first column with a value on most rows.

        Not simply column 0: these manuals lead product tables with an
        image column, which is empty in every row and would make every
        row subject-less.
        """
        width = max((len(r) for r in self.rows), default=0)
        for col in range(width):
            filled = sum(
                1 for r in self.rows if col < len(r) and r[col] and str(r[col]).strip()
            )
            if filled >= max(1, len(self.rows) // 2):
                return col
        return 0

    def to_prose(self) -> str:
        """A sentence per row, so the row/column association survives
        embedding even though the markdown grid does not."""
        if not self.header:
            return ""
        subject_col = self._subject_column()
        lines = []
        for r in self.rows:
            subject = (
                " ".join(str(r[subject_col]).split())
                if subject_col < len(r) and r[subject_col]
                else None
            )
            if not subject:
                continue
            pairs = [
                f"{' '.join(str(h).split())}: {' '.join(str(v).split())}"
                for i, (h, v) in enumerate(zip(self.header, r))
                if i != subject_col
                and v is not None
                and str(v).strip()
                and h is not None
                and str(h).strip()
            ]
            if pairs:
                lines.append(f"{subject} — " + "; ".join(pairs) + ".")
        return "\n".join(lines)


@dataclass
class Figure:
    id: str
    page: int
    bbox: tuple[float, float, float, float]
    caption: str
    number: str | None = None
    image_png: bytes | None = None


@dataclass
class ParsedDocument:
    path: Path
    page_count: int
    title: str
    product: str | None
    doc_number: str | None
    revision: str | None
    part_number: str | None
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    needs_ocr: bool = False
    low_confidence: bool = False
    notes: list[str] = field(default_factory=list)


def _open_or_fail(path: Path) -> pypdf.PdfReader:
    if not path.exists():
        raise ParseError(f"{path.name}: file not found")
    if path.stat().st_size == 0:
        raise ParseError(f"{path.name}: file is empty")
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as exc:
        raise ParseError(f"{path.name}: not a readable PDF ({exc})") from exc
    if reader.is_encrypted:
        try:
            opened = reader.decrypt("")
        except Exception as exc:
            raise ParseError(f"{path.name}: encrypted, cannot be opened ({exc})") from exc
        if not opened:
            raise ParseError(f"{path.name}: password-protected, no password supplied")
    try:
        if len(reader.pages) == 0:
            raise ParseError(f"{path.name}: contains no pages")
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(f"{path.name}: page tree is corrupt ({exc})") from exc
    return reader


def _lines(page: pdfplumber.page.Page) -> list[dict]:
    """Words grouped into visual lines, each carrying the geometry and
    font of its leftmost word."""
    words = page.extract_words(extra_attrs=["size", "fontname"])
    buckets: dict[int, list[dict]] = {}
    for w in words:
        buckets.setdefault(round(w["top"] / 3.0), []).append(w)
    out = []
    for key in sorted(buckets):
        group = sorted(buckets[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in group)
        if not text.strip():
            continue
        lead = group[0]
        out.append(
            {
                "text": text,
                "x0": lead["x0"],
                "x1": max(w["x1"] for w in group),
                "top": min(w["top"] for w in group),
                "bottom": max(w["bottom"] for w in group),
                "size": lead["size"],
                "font": lead["fontname"],
                "bold": "bold" in lead["fontname"].lower(),
                "words": group,
            }
        )
    return out


def _body_margin(lines: list[dict], body_size: float) -> float:
    cand = [round(ln["x0"]) for ln in lines if abs(ln["size"] - body_size) <= 1.5]
    if not cand:
        cand = [round(ln["x0"]) for ln in lines]
    return float(Counter(cand).most_common(1)[0][0]) if cand else 0.0


def _icons(page: pdfplumber.page.Page, margin: float) -> list[dict]:
    """Left-margin pictograms, deduplicated. A pictogram is drawn as
    several nested primitives (outline, fill, the '!' stem, its dot), so
    overlapping primitives are merged into one icon. A merged icon
    containing a `curve` is the warning triangle; a pure `rect` icon is
    the boxed "i" note."""
    prims = []
    for kind, objs in (("curve", page.curves), ("rect", page.rects)):
        for o in objs:
            w, h = abs(o["width"]), abs(o["height"])
            if not (ICON_MIN_PT < w < ICON_MAX_PT and ICON_MIN_PT < h < ICON_MAX_PT):
                continue
            if abs(w - h) > ICON_SQUARENESS_PT:
                continue
            if not (margin - 14.0 <= o["x0"] <= margin + 58.0):
                continue
            prims.append({"kind": kind, "x0": o["x0"], "x1": o["x1"], "top": o["top"], "bottom": o["bottom"]})

    prims.sort(key=lambda p: p["top"])
    merged: list[dict] = []
    for p in prims:
        for m in merged:
            if p["top"] < m["bottom"] and m["top"] < p["bottom"]:
                m["top"] = min(m["top"], p["top"])
                m["bottom"] = max(m["bottom"], p["bottom"])
                m["x0"] = min(m["x0"], p["x0"])
                m["x1"] = max(m["x1"], p["x1"])
                m["kinds"].add(p["kind"])
                break
        else:
            merged.append({**p, "kinds": {p["kind"]}})
    for m in merged:
        m["callout_kind"] = "warning" if "curve" in m["kinds"] else "note"
    return merged


def _repeated_running_heads(pages_lines: list[list[dict]], heights: list[float]) -> set[str]:
    """Header/footer text that repeats across pages. Left in place it
    lands in every chunk and pollutes every embedding."""
    counts: Counter[str] = Counter()
    for lines, height in zip(pages_lines, heights):
        seen = set()
        for ln in lines:
            if ln["top"] <= HEADER_BAND_PT or ln["bottom"] >= height - FOOTER_BAND_PT:
                seen.add(" ".join(ln["text"].split()))
        counts.update(seen)
    threshold = max(2, int(len(pages_lines) * 0.4))
    return {t for t, n in counts.items() if n >= threshold and len(t) < 160}


def _is_toc_page(lines: list[dict]) -> bool:
    hits = sum(1 for ln in lines if DOT_LEADER_RE.search(ln["text"]))
    return hits >= 4


def _heading_level(line: dict, body_size: float, number: str | None) -> int | None:
    if number:
        return min(number.count(".") + 1, 4)
    if line["size"] >= body_size + 3.5 and line["bold"]:
        return 1
    if line["size"] >= body_size + 1.5 and line["bold"] and len(line["text"]) < 90:
        return 2
    return None


def _table_is_real(rows: list[list[str | None]]) -> bool:
    """find_tables() also returns layout boxes and diagram frames. A real
    table has at least two populated rows, two populated columns, and a
    reasonable share of filled cells."""
    grid = [[(c if c is not None and str(c).strip() else None) for c in r] for r in rows]
    grid = [r for r in grid if any(c is not None for c in r)]
    if len(grid) < 2:
        return False
    width = max(len(r) for r in grid)
    live_cols = [i for i in range(width) if any(i < len(r) and r[i] is not None for r in grid)]
    if len(live_cols) < 2:
        return False
    filled = sum(1 for r in grid for c in r if c is not None)
    if filled / float(len(grid) * width) < 0.25:
        return False
    # A diagram frame captured as a table holds a few cells of many
    # newline-separated labels rather than many short cells.
    longest = max((len(str(c)) for r in grid for c in r if c is not None), default=0)
    return not (filled <= 4 and longest > 180)


def _compact(rows: list[list[str | None]]) -> tuple[list[str], list[list[str | None]]]:
    grid = [[(None if c is None or not str(c).strip() else " ".join(str(c).split())) for c in r] for r in rows]
    grid = [r for r in grid if any(c is not None for c in r)]
    if not grid:
        return [], []
    width = max(len(r) for r in grid)
    grid = [r + [None] * (width - len(r)) for r in grid]
    live = [i for i in range(width) if any(r[i] is not None for r in grid)]
    grid = [[r[i] for i in live] for r in grid]
    header = [c if c is not None else "" for c in grid[0]]
    return header, grid[1:]


def _nearest_caption(lines: list[dict], bbox: tuple[float, float, float, float]) -> str | None:
    best, best_gap = None, 72.0
    for ln in lines:
        m = FIGURE_CAPTION_RE.match(ln["text"])
        if not m:
            continue
        gap = ln["top"] - bbox[3]
        if 0 <= gap < best_gap:
            best, best_gap = ln["text"].strip(), gap
    return best


def _doc_identity(reader: pypdf.PdfReader, path: Path, front_matter: str = "") -> dict:
    """Identity comes from the XMP metadata when the document has it. The
    two draft documents in this corpus have an empty /Subject and a
    working title ("GB_Version_Draft_SEM_DOKU"), so the front matter is
    the fallback — those PDFs carry the real product name, part number,
    document number and revision on the title page instead."""
    meta = reader.metadata or {}
    title = (meta.get("/Title") or "").strip() or path.stem
    subject = (meta.get("/Subject") or "").strip()
    blob = f"{path.stem} {title} {subject}"

    doc_m = DOC_NUMBER_RE.search(path.stem) or DOC_NUMBER_RE.search(blob)
    rev_m = REVISION_RE.search(subject) or REVISION_RE.search(blob)
    part_m = PART_NUMBER_RE.search(title)
    prod_m = PRODUCT_RE.search(title)

    if front_matter:
        doc_m = doc_m or DOC_NUMBER_RE.search(front_matter)
        rev_m = rev_m or REVISION_RE.search(front_matter)
        part_m = part_m or PART_NUMBER_RE.search(front_matter)
        prod_m = prod_m or PRODUCT_RE.search(front_matter)

    product = None
    if prod_m:
        product = " ".join(prod_m.group(1).split()).rstrip(" .-/")
    display = " ".join(title.split())
    if front_matter and re.search(r"draft|_doku|version", display, re.I) and product:
        display = product
    return {
        "title": display,
        "doc_number": doc_m.group(1) if doc_m else None,
        "revision": rev_m.group(1) if rev_m else None,
        "part_number": part_m.group(1) if part_m else None,
        "product": product,
    }


def parse_pdf(path: str | Path, extract_figure_images: bool = True) -> ParsedDocument:
    path = Path(path)
    reader = _open_or_fail(path)

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise ParseError(f"{path.name}: text layer could not be read ({exc})") from exc

    with pdf:
        pages_lines = [_lines(p) for p in pdf.pages]
        heights = [float(p.height) for p in pdf.pages]
        total_chars = sum(len(ln["text"]) for lines in pages_lines for ln in lines)

        front_matter = "\n".join(
            ln["text"] for lines in pages_lines[:3] for ln in lines
        )
        identity = _doc_identity(reader, path, front_matter)

        doc = ParsedDocument(
            path=path,
            page_count=len(pdf.pages),
            title=identity["title"],
            product=identity["product"],
            doc_number=identity["doc_number"],
            revision=identity["revision"],
            part_number=identity["part_number"],
        )

        if total_chars < MIN_CHARS_PER_PAGE * doc.page_count:
            doc.needs_ocr = True
            doc.low_confidence = True
            doc.notes.append(
                f"no usable text layer ({total_chars} chars over {doc.page_count} pages); "
                "routed to OCR with low confidence"
            )
            return doc

        running = _repeated_running_heads(pages_lines, heights)
        sizes = Counter(
            round(ln["size"]) for lines in pages_lines for ln in lines if 6 <= ln["size"] <= 14
        )
        body_size = float(sizes.most_common(1)[0][0]) if sizes else 10.0

        for idx, page in enumerate(pdf.pages):
            page_no = idx + 1
            lines = pages_lines[idx]
            height = heights[idx]
            margin = _body_margin(lines, body_size)

            try:
                found = page.find_tables()
            except Exception:
                found = []
            page_tables: list[Table] = []
            table_zones: list[tuple[float, float, float, float]] = []
            for t_idx, tbl in enumerate(found):
                try:
                    raw = tbl.extract()
                except Exception:
                    continue
                if not _table_is_real(raw):
                    continue
                header, rows = _compact(raw)
                if not rows:
                    continue
                bbox = tuple(float(v) for v in tbl.bbox)
                page_tables.append(
                    Table(
                        id=f"{page_no}-t{t_idx}",
                        page=page_no,
                        bbox=bbox,  # type: ignore[arg-type]
                        header=header,
                        rows=rows,
                    )
                )
                table_zones.append(bbox)  # type: ignore[arg-type]

            def in_table(ln: dict) -> bool:
                cy = (ln["top"] + ln["bottom"]) / 2.0
                cx = (ln["x0"] + ln["x1"]) / 2.0
                return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in table_zones)

            # The cover page's logo and title artwork produce icon-shaped
            # vectors beside the title block. There are no procedures on a
            # cover, so no callouts are looked for there.
            icons = [] if page_no == 1 else _icons(page, margin)
            toc = _is_toc_page(lines)

            # Every line that belongs to a callout, so it is not also
            # emitted as a loose paragraph.
            claimed: set[int] = set()
            callouts: list[Block] = []
            for icon in icons:
                start = None
                for i, ln in enumerate(lines):
                    if i in claimed or in_table(ln):
                        continue
                    if ln["x0"] < icon["x1"] + ICON_TEXT_GAP_PT - 20.0:
                        continue
                    if ln["x0"] < margin + CALLOUT_INDENT_PT:
                        continue
                    if ln["bottom"] < icon["top"] - 14.0 or ln["top"] > icon["bottom"] + 22.0:
                        continue
                    start = i
                    break
                if start is None:
                    continue
                body = []
                for i in range(start, len(lines)):
                    ln = lines[i]
                    if i in claimed or in_table(ln):
                        break
                    if ln["top"] <= HEADER_BAND_PT or ln["bottom"] >= height - FOOTER_BAND_PT:
                        break
                    if i > start and ln["x0"] < margin + CALLOUT_INDENT_PT:
                        break
                    if i > start and ln["top"] - lines[i - 1]["bottom"] > 26.0:
                        break
                    if SECTION_NUMBER_RE.match(ln["text"]) and ln["size"] > body_size + 1.0:
                        break
                    body.append(ln["text"])
                    claimed.add(i)
                text = " ".join(" ".join(b.split()) for b in body).strip()
                if len(text) < 12:
                    continue
                callouts.append(
                    Block(
                        type="callout",
                        text=text,
                        page=page_no,
                        bbox=(icon["x0"], icon["top"], max(ln["x1"] for ln in lines), icon["bottom"]),
                        callout_kind=icon["callout_kind"],
                    )
                )

            figures: list[Figure] = []
            for ln in lines:
                m = FIGURE_CAPTION_RE.match(ln["text"])
                if not m:
                    continue
                top = max(HEADER_BAND_PT, ln["top"] - 300.0)
                for other in lines:
                    if other["bottom"] < ln["top"] - 4.0 and other["bottom"] > top:
                        if other["x0"] <= margin + 4.0 and not FIGURE_CAPTION_RE.match(other["text"]):
                            top = other["bottom"]
                region = (
                    max(0.0, margin - 12.0),
                    top + 2.0,
                    float(page.width) - 20.0,
                    ln["top"] - 2.0,
                )
                if region[3] - region[1] < 24.0:
                    continue
                figures.append(
                    Figure(
                        id=f"{page_no}-f{m.group(1)}",
                        page=page_no,
                        bbox=region,
                        caption=" ".join(ln["text"].split()),
                        number=m.group(1),
                    )
                )

            # Wiring and schematic diagrams in the draft manuals carry no
            # "Fig. n:" caption at all; they are a large framing rect with
            # a label line ("Schematic drawing") above. Without this the
            # whole diagram chapter of 89470K2 extracts as nothing.
            if not figures:
                frames = [
                    r
                    for r in page.rects
                    if abs(r["width"]) >= FIGURE_FRAME_MIN_W_PT
                    and abs(r["height"]) >= FIGURE_FRAME_MIN_H_PT
                    and not any(
                        z[0] - 2 <= r["x0"] and r["x1"] <= z[2] + 2 and z[1] - 2 <= r["top"]
                        for z in table_zones
                    )
                ]
                kept: list[tuple[float, float, float, float]] = []
                for r in sorted(frames, key=lambda r: (r["top"], r["x0"])):
                    box = (float(r["x0"]), float(r["top"]), float(r["x1"]), float(r["bottom"]))
                    if any(
                        abs(box[1] - k[1]) < 12.0 and abs(box[0] - k[0]) < 12.0 for k in kept
                    ):
                        continue
                    kept.append(box)
                for f_idx, box in enumerate(kept):
                    label = ""
                    for ln in lines:
                        if ln["bottom"] <= box[1] + 2.0 and box[1] - ln["bottom"] < 42.0:
                            if UNCAPTIONED_FIGURE_LABEL_RE.match(ln["text"]):
                                label = " ".join(ln["text"].split())
                    figures.append(
                        Figure(
                            id=f"{page_no}-d{f_idx}",
                            page=page_no,
                            bbox=box,
                            caption=label,
                        )
                    )

            if extract_figure_images:
                for fig in figures:
                    try:
                        img = page.crop(fig.bbox).to_image(resolution=120)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        fig.image_png = buf.getvalue()
                    except Exception as exc:
                        doc.notes.append(f"p{page_no} figure {fig.id}: render failed ({exc})")

            for fig in figures:
                if not fig.caption:
                    fig.caption = _nearest_caption(lines, fig.bbox) or ""

            # Emit blocks in reading order.
            emitted: list[Block] = list(callouts)
            for i, ln in enumerate(lines):
                if i in claimed:
                    continue
                flat = " ".join(ln["text"].split())
                if flat in running:
                    continue
                if ln["top"] <= HEADER_BAND_PT or ln["bottom"] >= height - FOOTER_BAND_PT:
                    continue
                if toc or DOT_LEADER_RE.search(flat):
                    continue
                if in_table(ln):
                    continue
                if FIGURE_CAPTION_RE.match(flat):
                    continue
                m = SECTION_NUMBER_RE.match(flat)
                number = m.group(1) if m else None
                level = _heading_level(ln, body_size, number if m and ln["bold"] else None)
                if level is not None and (ln["bold"] or ln["size"] > body_size + 1.0):
                    emitted.append(
                        Block(
                            type="heading",
                            text=(m.group(2).strip() if m else flat),
                            page=page_no,
                            bbox=(ln["x0"], ln["top"], ln["x1"], ln["bottom"]),
                            level=level,
                            section_number=number,
                        )
                    )
                else:
                    emitted.append(
                        Block(
                            type="paragraph",
                            text=flat,
                            page=page_no,
                            bbox=(ln["x0"], ln["top"], ln["x1"], ln["bottom"]),
                        )
                    )

            for tbl in page_tables:
                emitted.append(
                    Block(
                        type="table",
                        text=tbl.to_markdown(),
                        page=page_no,
                        bbox=tbl.bbox,
                        ref=tbl.id,
                    )
                )
            for fig in figures:
                emitted.append(
                    Block(
                        type="figure",
                        text=fig.caption,
                        page=page_no,
                        bbox=fig.bbox,
                        ref=fig.id,
                    )
                )

            emitted.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            doc.blocks.extend(emitted)
            doc.tables.extend(page_tables)
            doc.figures.extend(figures)

    return doc
