#!/usr/bin/env python3
"""
extract_all_from_scanned_pdf.py

End-to-end PDF page rendering → layout detection → OCR → tables/figures export.

Features
--------
- Renders pages with pypdfium2 (fast, high-quality). Optional fallback to PyMuPDF.
- Auto-rotation using Tesseract OSD (safe fallback if OSD fails).
- Gentle denoise + adaptive threshold to help OCR while preserving fine lines.
- Dual OCR (Tesseract + PaddleOCR) with reconciliation (digit-biased, fuzz match).
- Layout detection with PaddleOCR's PP-Structure V3 (tables/figures/text blocks).
- Table HTML → CSV/XLSX extraction when available.
- Figure crops saved as PNGs.
- Searchable PDF + sidecar text (optional) via OCRmyPDF prepass.
- Per-page JSON artifacts and a top-level report.json.
- Defensive handling of optional dependencies and version differences.

Usage
-----
python extract_all_from_scanned_pdf.py <input.pdf> [--out OUTDIR] [--dpi DPI]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Third-party imports (core hard deps) ---
import numpy as np
from PIL import Image
import cv2
import pypdfium2 as pdfium  # primary renderer
import pytesseract          # requires native Tesseract installed
from paddleocr import PaddleOCR, PPStructureV3  # layout + OCR
from rapidfuzz import fuzz
import pandas as pd

# --- Optional dependencies (soft deps) ---
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    import layoutparser as lp  # not used directly but often requested
    HAVE_LAYOUTPARSER = True
except Exception:
    HAVE_LAYOUTPARSER = False

HAVE_TATR = False
try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # noqa: F401
    import torch  # noqa: F401
    HAVE_TATR = True
except Exception:
    pass

try:
    import ocrmypdf
    HAVE_OCRMYPDF = True
except Exception:
    HAVE_OCRMYPDF = False

# --------------------------- Data Models ---------------------------

@dataclass
class Word:
    text: str
    conf: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    engine: str

@dataclass
class TableResult:
    bbox: Tuple[int, int, int, int]
    engine: str
    html: Optional[str] = None
    csv_paths: Optional[List[str]] = None
    xlsx_paths: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class FigureResult:
    bbox: Tuple[int, int, int, int]
    path: str
    engine: str = "layout"

# --------------------------- Utilities ----------------------------

def configure_logging(verbosity: int) -> None:
    """Configure root logger level and format."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    """Return MD5 hex digest of bytes."""
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()

def _render_with_pdfium(pdf_path: Path, dpi: int) -> List[Image.Image]:
    """Render pages using pypdfium2."""
    images: List[Image.Image] = []
    scale = dpi / 72.0  # PDF point is 1/72 inch
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF with pypdfium2: {e}") from e

    for i in range(len(pdf)):
        page = pdf[i]
        try:
            pil = page.render(scale=scale).to_pil().convert("RGB")
            images.append(pil)
        except Exception as e:
            logging.error("Failed to render page %d with pypdfium2: %s", i + 1, e)
            if not HAVE_PYMUPDF:
                raise
    return images

def _render_with_pymupdf(pdf_path: Path, dpi: int) -> List[Image.Image]:
    """Optional fallback render using PyMuPDF if available."""
    if not HAVE_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is not installed.")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images: List[Image.Image] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

def render_pdf_pages(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """
    Render all pages to PIL images. Tries pypdfium2; falls back to PyMuPDF if needed.
    Raises RuntimeError if rendering fails.
    """
    try:
        return _render_with_pdfium(pdf_path, dpi)
    except Exception as e_pdfium:
        logging.warning("pypdfium2 rendering failed (%s).", e_pdfium)
        if HAVE_PYMUPDF:
            logging.info("Trying PyMuPDF fallback...")
            try:
                return _render_with_pymupdf(pdf_path, dpi)
            except Exception as e_fitz:
                raise RuntimeError(f"Both renderers failed: pdfium={e_pdfium} ; fitz={e_fitz}") from e_fitz
        raise

def auto_rotate_with_osd(pil_img: Image.Image) -> Image.Image:
    """
    Attempt to auto-rotate using Tesseract OSD.
    Silently falls back to original image on failure.
    """
    try:
        img = np.array(pil_img)
        osd = pytesseract.image_to_osd(img)
        angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                try:
                    angle = int(line.split(":")[1].strip())
                except Exception:
                    angle = 0
                break
        if angle:
            return pil_img.rotate(-angle, expand=True)
    except Exception as e:
        logging.debug("OSD rotation skipped: %s", e)
    return pil_img

def basic_clean(pil_img: Image.Image) -> Image.Image:
    """
    Light denoise + adaptive threshold. Gentle to preserve edges/lines.
    """
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, 10
    )
    return Image.fromarray(th)

def tess_words(pil_img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> List[Word]:
    """
    Run Tesseract word-level OCR with configurable PSM and optional whitelist.
    Returns list of Word objects.
    """
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(
            pil_img, output_type=pytesseract.Output.DICT, config=cfg
        )
    except pytesseract.pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract not found. Please install native Tesseract and ensure it's on PATH."
        ) from e
    except Exception as e:
        logging.error("Tesseract OCR failed: %s", e)
        return []

    words: List[Word] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt:
            continue
        conf_str = data.get("conf", ["-1"] * n)[i]
        try:
            conf = float(conf_str) if conf_str != "-1" else 0.0
        except Exception:
            conf = 0.0
        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        words.append(Word(txt, conf, (x, y, w, h), engine="tesseract"))
    return words

def paddle_words(paddle_ocr: PaddleOCR, pil_img: Image.Image) -> List[Word]:
    """
    Run PaddleOCR (English) and normalize to Word list.
    Handles API differences across PaddleOCR versions.
    """
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    result = None
    try:
        if hasattr(paddle_ocr, "predict"):
            try:
                result = paddle_ocr.predict(img)
            except TypeError:
                result = None
        if result is None:
            result = paddle_ocr.ocr(img)  # legacy path
    except Exception as e:
        logging.error("PaddleOCR failed: %s", e)
        return []

    words: List[Word] = []
    # Typical structure: result ~ [ [ (points), (text, conf) ], ... ] per line
    try:
        for line in result or []:
            if not line:
                continue
            for item in line:
                if not item or len(item) < 2:
                    continue
                box, info = item[0], item[1]
                if not isinstance(box, (list, tuple)) or not isinstance(info, (list, tuple)) or len(info) < 2:
                    continue
                txt, conf = info[0], info[1]
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
                x = int(min(x1, x2, x3, x4))
                y = int(min(y1, y2, y3, y4))
                w = int(max(x1, x2, x3, x4)) - x
                h = int(max(y1, y2, y3, y4)) - y
                words.append(Word(str(txt).strip(), float(conf) * 100.0, (x, y, w, h), engine="paddleocr"))
    except Exception as e:
        logging.debug("Failed to normalize PaddleOCR result: %s", e)
    return words

def iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Intersection-over-Union for (x,y,w,h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    a1 = w1 * h1
    a2 = w2 * h2
    return inter / float(a1 + a2 - inter)

def reconcile_words(tess: List[Word], padd: List[Word], digit_bias: bool = True) -> List[Word]:
    """
    Merge two word lists; prefer higher confidence.
    For digit-like tokens, prefer candidate with more digits and better similarity.
    """
    merged: List[Word] = []
    used: set[int] = set()

    for tw in tess:
        best = None
        best_j = -1
        best_iou = 0.0
        for j, pw in enumerate(padd):
            if j in used:
                continue
            ov = iou(tw.bbox, pw.bbox)
            if ov > best_iou:
                best_iou = ov
                best = pw
                best_j = j

        candidate = tw
        if best and best_iou > 0.2:
            isnum_t = any(ch.isdigit() for ch in tw.text) if digit_bias else False
            isnum_p = any(ch.isdigit() for ch in best.text) if digit_bias else False
            sim = fuzz.token_set_ratio(tw.text, best.text)
            if (best.conf > tw.conf + 5) or (isnum_p and not isnum_t) or (sim < 70 and best.conf > tw.conf):
                candidate = best
                used.add(best_j)
        merged.append(candidate)

    for j, pw in enumerate(padd):
        if j not in used:
            merged.append(pw)
    return merged

# ---- PP-Structure helpers ----

def to_bgr3(pil_img: Image.Image) -> np.ndarray:
    """Ensure 3-channel BGR uint8 for PP-Structure."""
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def ppstruct_run(ppstruct: PPStructureV3, img_bgr: np.ndarray):
    """
    Call PPStructureV3 regardless of interface differences across versions.
    Input must be BGR 3-channel uint8.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("ppstruct_run expects a 3-channel BGR image.")
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    # 1) Callable
    try:
        return ppstruct(img_bgr)
    except TypeError:
        pass
    except Exception:
        raise
    # 2) .predict
    if hasattr(ppstruct, "predict"):
        return ppstruct.predict(img_bgr)
    # 3) .structure
    if hasattr(ppstruct, "structure"):
        return ppstruct.structure(img_bgr)
    # 4) .inference
    if hasattr(ppstruct, "inference"):
        return ppstruct.inference(img_bgr)
    raise RuntimeError(
        "Unsupported PPStructureV3 interface: not callable and no known predict/structure/inference method."
    )

def detect_layout_PPStructureV3(ppstruct: PPStructureV3, pil_img: Image.Image) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Use PP-Structure to get regions (table/image/text). Always feed BGR 3-channel.
    Returns dict with keys: 'tables', 'figures', 'texts'.
    """
    img_bgr = to_bgr3(pil_img)
    try:
        results = ppstruct_run(ppstruct, img_bgr)
    except Exception as e:
        logging.warning("PP-Structure layout detection failed: %s", e)
        return {"tables": [], "figures": [], "texts": []}

    layout = {"tables": [], "figures": [], "texts": []}
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("res", []) or []

    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get("type") or r.get("label") or "text"
        bbox_raw = r.get("bbox") or r.get("box") or r.get("rect")
        if bbox_raw is None:
            continue
        try:
            bbox = tuple(map(int, bbox_raw))
            if t == "table":
                layout["tables"].append(bbox)
            elif str(t).lower() in ("figure", "image", "picture", "pic", "chart"):
                layout["figures"].append(bbox)
            else:
                layout["texts"].append(bbox)
        except Exception:
            logging.debug("Skipping malformed layout record: %s", r)
            continue
    return layout

def crop(pil_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Crop (x,y,w,h) from PIL image."""
    x, y, w, h = bbox
    return pil_img.crop((x, y, x + w, y + h))

def extract_tables_PPStructureV3(ppstruct: PPStructureV3, pil_img: Image.Image, page_dir: Path) -> List[TableResult]:
    """
    Run PP-Structure and collect only table items. If table HTML is provided,
    caller may export CSV/XLSX via pandas.read_html.
    """
    img_bgr = to_bgr3(pil_img)
    try:
        results = ppstruct_run(ppstruct, img_bgr)
    except Exception as e:
        logging.warning("PP-Structure table extraction failed: %s", e)
        return []

    out: List[TableResult] = []
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("res", []) or []

    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get("type") or r.get("label")
        if t != "table":
            continue
        bbox_raw = r.get("bbox") or r.get("box") or r.get("rect")
        if bbox_raw is None:
            continue
        try:
            bbox = tuple(map(int, bbox_raw))
        except Exception:
            logging.debug("Skipping table with invalid bbox: %s", bbox_raw)
            continue
        meta = {k: r[k] for k in r if k not in ("img", "res", "html")}
        html = None
        res = r.get("res")
        if isinstance(res, dict):
            html = res.get("html") or res.get("table", {}).get("html")
        elif isinstance(r.get("html"), str):
            html = r.get("html")
        out.append(TableResult(bbox=bbox, engine="pp-structure", html=html, meta=meta, csv_paths=[], xlsx_paths=[]))
    return out

def save_figures(pil_img: Image.Image, figure_bboxes: List[Tuple[int, int, int, int]], page_dir: Path) -> List[FigureResult]:
    """Save figure crops to disk and return FigureResult list."""
    figs: List[FigureResult] = []
    for i, bb in enumerate(figure_bboxes, start=1):
        try:
            crop_img = crop(pil_img, bb)
            p = page_dir / f"figure_{i}.png"
            crop_img.save(p)
            figs.append(FigureResult(bb, str(p)))
        except Exception as e:
            logging.debug("Failed to save figure %d: %s", i, e)
    return figs

def page_quality(pil_img: Image.Image) -> float:
    """Simple sharpness score via variance of Laplacian; higher is sharper."""
    try:
        lap = cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var()
        return float(lap)
    except Exception:
        return 0.0

# --------------------------- Pipeline -----------------------------

def run_pipeline(pdf_path: Path, outdir: Path, dpi: int) -> None:
    """
    Execute the full pipeline for the given PDF:
    - Optional OCRmyPDF prepass
    - Page rendering
    - For each page: rotation, cleaning, layout, OCR, reconciliation, tables, figures
    - Per-page JSON + text, plus a summary report.json
    """
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    ensure_dir(outdir)

    # (Optional) Searchable PDF + sidecar
    if HAVE_OCRMYPDF:
        try:
            logging.info("Running OCRmyPDF prepass...")
            ocrmypdf.ocr(
                str(pdf_path),
                str(outdir / "searchable.pdf"),
                force_ocr=True,
                deskew=True,
                rotate_pages=True,
                clean=True,
                sidecar=str(outdir / "sidecar.txt"),
            )
        except Exception as e:
            logging.warning("OCRmyPDF prepass skipped: %s", e)

    # Initialize OCR engines (errors here should be explicit)
    try:
        paddle_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="en",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e

    try:
        ppstruct = PPStructureV3()  # avoid kwargs for cross-version compatibility
    except Exception as e:
        logging.warning("PP-Structure initialization failed: %s (layout will be disabled)", e)
        ppstruct = None  # type: ignore

    # Render pages
    pages = render_pdf_pages(pdf_path, dpi=dpi)
    if not pages:
        raise RuntimeError("No pages rendered from input PDF.")

    report: Dict[str, Any] = {"pages": []}

    for idx, page in enumerate(pages, start=1):
        page_dir = outdir / "pages" / f"page_{idx:03d}"
        ensure_dir(page_dir)

        # Orientation + cleaning
        rot = auto_rotate_with_osd(page)  # keep color
        cleaned = basic_clean(rot)        # binarized

        # Layout detection on COLOR (rot)
        if ppstruct is not None:
            layout = detect_layout_PPStructureV3(ppstruct, rot)
        else:
            layout = {"tables": [], "figures": [], "texts": []}

        # OCR — use text boxes from layout; OCR on cleaned image
        if layout["texts"]:
            tess_words_all: List[Word] = []
            padd_words_all: List[Word] = []
            for tb in layout["texts"]:
                roi_clean = crop(cleaned, tb)
                tess_words_all += tess_words(roi_clean, psm=6)
                padd_words_all += paddle_words(paddle_ocr, roi_clean)
        else:
            tess_words_all = tess_words(cleaned, psm=6)
            padd_words_all = paddle_words(paddle_ocr, cleaned)

        merged_words = reconcile_words(tess_words_all, padd_words_all, digit_bias=True)

        # Save plain text (space-joined; keep simple)
        text_out = " ".join([w.text for w in merged_words])
        try:
            (page_dir / "text.txt").write_text(text_out, encoding="utf-8")
        except Exception as e:
            logging.warning("Failed to write text.txt for page %d: %s", idx, e)

        # Tables (run on COLOR page)
        if ppstruct is not None:
            tables = extract_tables_PPStructureV3(ppstruct, rot, page_dir)
        else:
            tables = []

        # Save simple CSV/XLSX from HTML when present
        for ti, t in enumerate(tables, start=1):
            if not t.html:
                continue
            try:
                dfs = pd.read_html(t.html)
            except Exception as e:
                logging.debug("read_html failed for page %d table %d: %s", idx, ti, e)
                continue
            if not dfs:
                continue
            df = dfs[0]
            try:
                csv_p = page_dir / f"table_{ti}.csv"
                xlsx_p = page_dir / f"table_{ti}.xlsx"
                df.to_csv(csv_p, index=False)
                df.to_excel(xlsx_p, index=False)
                t.csv_paths = [str(csv_p)]
                t.xlsx_paths = [str(xlsx_p)]
            except Exception as e:
                logging.warning("Failed to save table files for page %d table %d: %s", idx, ti, e)

        # Figures
        figures = save_figures(rot, layout["figures"], page_dir)

        # Metrics
        avg_conf = float(np.mean([w.conf for w in merged_words])) if merged_words else 0.0
        blur = page_quality(rot)

        page_json = {
            "page": idx,
            "avg_conf": avg_conf,
            "blur": blur,
            "words": [asdict(w) for w in merged_words],
            "tables": [asdict(t) for t in tables],
            "figures": [asdict(f) for f in figures],
            "layout_counts": {k: len(v) for k, v in layout.items()},
        }
        try:
            (page_dir / "page.json").write_text(json.dumps(page_json, indent=2), encoding="utf-8")
        except Exception as e:
            logging.warning("Failed to write page.json for page %d: %s", idx, e)

        report["pages"].append({
            "page": idx,
            "avg_conf": avg_conf,
            "blur": blur,
            "n_words": len(merged_words),
            "n_tables": len(tables),
            "n_figures": len(figures),
        })

    # Write top-level report
    try:
        (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as e:
        logging.warning("Failed to write report.json: %s", e)

    logging.info("Processing complete. Output directory: %s", outdir.resolve())
    print(f"Done. See: {outdir.resolve()}")

# --------------------------- CLI -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render scanned PDFs and extract text/tables/figures with OCR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    parser.add_argument("--dpi", type=int, default=350, help="Render DPI for page images")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase log verbosity (-v = INFO, -vv = DEBUG)")
    return parser.parse_args(argv)

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    pdf_path = Path(args.pdf).expanduser().resolve()
    outdir = Path("results") / Path(args.out)
    outdir = outdir.expanduser().resolve()

    try:
        run_pipeline(pdf_path, outdir, args.dpi)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(2)
    except RuntimeError as e:
        logging.error("Runtime error: %s", e)
        sys.exit(3)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.exception("Unexpected failure: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
