#!/usr/bin/env python3
"""
extract_all_from_scanned_pdf.py

Scanned-PDF → text, figures, TABLES→CSV/XLSX (robust), with multi-page table stitching.

What’s new vs. previous:
- Better PNG→CSV via grid extractor (Hough lines → grid → per-cell OCR).
- Multi-page tables stitched across pages using header continuity + column alignment.
- Keeps PP-Structure HTML path first (best), then grid extractor, then contour heuristic.
- Version-safe PaddleOCR/PP-Structure inits.

CLI:
  python extract_all_from_scanned_pdf.py input.pdf --out out --dpi 400 -v
  (use --no-tatr / --no-opencv / --no-stitch to disable specific features)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
import pypdfium2 as pdfium
import pytesseract
from paddleocr import PaddleOCR, PPStructureV3
from rapidfuzz import fuzz
import pandas as pd

# Optional deps
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    import ocrmypdf
    HAVE_OCRMYPDF = True
except Exception:
    HAVE_OCRMYPDF = False

HAVE_TATR = False
try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # noqa: F401
    import torch  # noqa: F401
    HAVE_TATR = True
except Exception:
    pass


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
    image_path: Optional[str] = None  # crop path
    df_path: Optional[str] = None     # CSV produced (single)
    part_id: Optional[str] = None     # for stitching
    stitched_group: Optional[str] = None

@dataclass
class FigureResult:
    bbox: Tuple[int, int, int, int]
    path: str
    engine: str = "layout"

@dataclass
class TablePiece:
    page: int
    page_size: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    image_path: Optional[str]
    df: Optional[pd.DataFrame]
    header_sig: Optional[str]
    n_cols: Optional[int]
    at_top: bool
    at_bottom: bool
    part_id: str  # unique id like p001_t002


# --------------------------- Logging -------------------------------

def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# --------------------------- Utilities ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    m = hashlib.md5(); m.update(b); return m.hexdigest()

def _render_with_pdfium(pdf_path: Path, dpi: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    scale = dpi / 72.0
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

def render_pdf_pages(pdf_path: Path, dpi: int = 350) -> List[Image.Image]:
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
    try:
        img = np.array(pil_img)
        osd = pytesseract.image_to_osd(img)
        angle = 0
        for line in osd.splitlines():
            if "Rotate:" in line:
                try: angle = int(line.split(":")[1].strip())
                except Exception: angle = 0
                break
        if angle:
            return pil_img.rotate(-angle, expand=True)
    except Exception as e:
        logging.debug("OSD rotation skipped: %s", e)
    return pil_img

def basic_clean(pil_img: Image.Image) -> Image.Image:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 8)
    return Image.fromarray(th)

def crop(pil_img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
    x, y, w, h = bbox
    return pil_img.crop((x, y, x + w, y + h))

def resize_long_edge(pil_img: Image.Image, long_edge: int = 2400) -> Image.Image:
    w, h = pil_img.size
    scale = long_edge / max(w, h)
    if scale <= 1.0:
        return pil_img
    return pil_img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

def page_quality(pil_img: Image.Image) -> float:
    try:
        lap = cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var()
        return float(lap)
    except Exception:
        return 0.0


# --------------------------- OCR words ----------------------------

def tess_words(pil_img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> List[Word]:
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    try:
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=cfg)
    except pytesseract.pytesseract.TesseractNotFoundError as e:
        raise RuntimeError("Tesseract not found on PATH.") from e
    except Exception as e:
        logging.error("Tesseract OCR failed: %s", e)
        return []
    words: List[Word] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = str(data["text"][i]).strip()
        if not txt: continue
        conf_str = data.get("conf", ["-1"] * n)[i]
        try: conf = float(conf_str) if conf_str != "-1" else 0.0
        except Exception: conf = 0.0
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append(Word(txt, conf, (x, y, w, h), engine="tesseract"))
    return words

def paddle_words(paddle_ocr: PaddleOCR, pil_img: Image.Image) -> List[Word]:
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
            except Exception:
                result = None
        if result is None:
            result = paddle_ocr.ocr(img)
    except Exception as e:
        logging.error("PaddleOCR failed: %s", e)
        return []

    words: List[Word] = []
    try:
        for line in result or []:
            if not line: continue
            for item in line:
                if not item or len(item) < 2: continue
                box, info = item[0], item[1]
                if not isinstance(box, (list, tuple)) or not isinstance(info, (list, tuple)) or len(info) < 2:
                    continue
                txt, conf = info[0], info[1]
                (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
                x = int(min(x1, x2, x3, x4)); y = int(min(y1, y2, y3, y4))
                w = int(max(x1, x2, x3, x4)) - x; h = int(max(y1, y2, y3, y4)) - y
                words.append(Word(str(txt).strip(), float(conf) * 100.0, (x, y, w, h), engine="paddleocr"))
    except Exception as e:
        logging.debug("Failed to normalize PaddleOCR result: %s", e)
    return words

def iou(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if xb <= xa or yb <= ya: return 0.0
    inter = (xb - xa) * (yb - ya)
    return inter / float(w1*h1 + w2*h2 - inter)

def reconcile_words(tess: List[Word], padd: List[Word], digit_bias: bool = True) -> List[Word]:
    from rapidfuzz import fuzz as rf
    merged: List[Word] = []; used: set[int] = set()
    for tw in tess:
        best, best_j, best_iou = None, -1, 0.0
        for j, pw in enumerate(padd):
            if j in used: continue
            ov = iou(tw.bbox, pw.bbox)
            if ov > best_iou: best_iou, best, best_j = ov, pw, j
        candidate = tw
        if best and best_iou > 0.2:
            isnum_t = any(ch.isdigit() for ch in tw.text) if digit_bias else False
            isnum_p = any(ch.isdigit() for ch in best.text) if digit_bias else False
            sim = rf.token_set_ratio(tw.text, best.text)
            if (best.conf > tw.conf + 5) or (isnum_p and not isnum_t) or (sim < 70 and best.conf > tw.conf):
                candidate = best; used.add(best_j)
        merged.append(candidate)
    for j, pw in enumerate(padd):
        if j not in used: merged.append(pw)
    return merged


# -------------------- PP-Structure helpers & detection --------------------

def to_bgr3(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2: arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4: arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else: arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def ppstruct_run(ppstruct: PPStructureV3, img_bgr: np.ndarray):
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("ppstruct_run expects a 3-channel BGR image.")
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)
    try:
        return ppstruct(img_bgr)
    except Exception:
        pass
    if hasattr(ppstruct, "predict"): return ppstruct.predict(img_bgr)
    if hasattr(ppstruct, "structure"): return ppstruct.structure(img_bgr)
    if hasattr(ppstruct, "inference"): return ppstruct.inference(img_bgr)
    raise RuntimeError("Unsupported PPStructureV3 interface.")

def detect_layout_PPStructureV3(ppstruct: PPStructureV3, pil_img: Image.Image) -> Dict[str, List[Tuple[int,int,int,int]]]:
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
        if not isinstance(r, dict): continue
        t = r.get("type") or r.get("label") or "text"
        bbox_raw = r.get("bbox") or r.get("box") or r.get("rect")
        if bbox_raw is None: continue
        try:
            bbox = tuple(map(int, bbox_raw))
            if t == "table": layout["tables"].append(bbox)
            elif str(t).lower() in ("figure","image","picture","pic","chart"):
                layout["figures"].append(bbox)
            else: layout["texts"].append(bbox)
        except Exception:
            logging.debug("Skipping malformed layout record: %s", r)
    return layout

def extract_tables_PPStructureV3(ppstruct: PPStructureV3, pil_img: Image.Image) -> List[TableResult]:
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
        if not isinstance(r, dict): continue
        t = r.get("type") or r.get("label")
        if t != "table": continue
        bbox_raw = r.get("bbox") or r.get("box") or r.get("rect")
        if bbox_raw is None: continue
        try: bbox = tuple(map(int, bbox_raw))
        except Exception: continue
        meta = {k: r[k] for k in r if k not in ("img", "res", "html")}
        html = None
        res = r.get("res")
        if isinstance(res, dict):
            html = res.get("html") or res.get("table", {}).get("html")
        elif isinstance(r.get("html"), str):
            html = r.get("html")
        out.append(TableResult(bbox=bbox, engine="pp-structure", html=html, meta=meta, csv_paths=[], xlsx_paths=[]))
    return out


# -------------------- Union detection + fallbacks --------------------

def de_dup_boxes(boxes: List[Tuple[int,int,int,int]], iou_thresh: float = 0.5) -> List[Tuple[int,int,int,int]]:
    merged: List[Tuple[int,int,int,int]] = []
    for b in boxes:
        if not any(iou(b, m) > iou_thresh for m in merged):
            merged.append(b)
    return merged

def detect_tables_union(ppstruct: Optional[PPStructureV3], pil_color: Image.Image, pil_clean: Image.Image) -> List[Tuple[int,int,int,int]]:
    if ppstruct is None: return []
    boxes: List[Tuple[int,int,int,int]] = []
    for src in (pil_color, pil_clean):
        up = resize_long_edge(src, 2400)
        layout = detect_layout_PPStructureV3(ppstruct, up)
        boxes.extend(layout.get("tables", []))
    return de_dup_boxes(boxes, 0.5)

def detect_tables_opencv(pil_img: Image.Image) -> List[Tuple[int,int,int,int]]:
    img = np.array(pil_img.convert("L"))
    thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)), iterations=1)
    vert  = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)), iterations=1)
    grid = cv2.add(horiz, vert)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 5000 or w < 100 or h < 60: continue
        boxes.append((x,y,w,h))
    out: List[Tuple[int,int,int,int]] = []
    for b in sorted(boxes, key=lambda bb: bb[2]*bb[3], reverse=True):
        if not any(iou(b, o) > 0.5 for o in out):
            out.append(b)
    return out

class TATRDetector:
    def __init__(self, model_name: str = "microsoft/table-transformer-detection", device: Optional[str] = None):
        if not HAVE_TATR:
            raise RuntimeError("Transformers/torch not available for TATR.")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        import torch as _torch
        self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def detect(self, pil_img: Image.Image, score_thresh: float = 0.7) -> List[Tuple[int,int,int,int]]:
        import torch as _torch
        with _torch.inference_mode():
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_object_detection(
                outputs, threshold=score_thresh, target_sizes=[pil_img.size[::-1]]
            )[0]
            boxes = []
            for box, label_id, score in zip(results["boxes"], results["labels"], results["scores"]):
                label = self.model.config.id2label[int(label_id)]
                if str(label).lower() == "table":
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes


# ---------------------- Table HTML from crop (PP-Structure) ----------------------

def pp_table_html_from_crop(ppstruct: Optional[PPStructureV3], pil_crop: Image.Image) -> Optional[str]:
    if ppstruct is None:
        return None
    img_bgr = to_bgr3(pil_crop)
    try:
        results = ppstruct_run(ppstruct, img_bgr)
    except Exception:
        return None
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("res", []) or []
    for r in results or []:
        if not isinstance(r, dict): continue
        res = r.get("res")
        if isinstance(res, dict):
            html = res.get("html") or res.get("table", {}).get("html")
            if isinstance(html, str) and "<table" in html.lower():
                return html
        if isinstance(r.get("html"), str) and "<table" in r["html"].lower():
            return r["html"]
    return None


# ---------------------- Grid-based PNG→CSV (Hough) ----------------------

def _cluster_positions(vals: List[int], tol: int) -> List[int]:
    """Cluster 1D positions by proximity; returns representative centers."""
    if not vals: return []
    vals = sorted(vals)
    clusters = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [int(np.median(c)) for c in clusters]

def grid_extract_dataframe(pil_crop: Image.Image) -> Optional[pd.DataFrame]:
    """
    Detect horizontal/vertical lines with Hough, build a grid, OCR each cell.
    Returns DataFrame or None.
    """
    img_rgb = np.array(pil_crop.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Binarize (OTSU) and invert
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bi

    # Emphasize lines
    w, h = pil_crop.size
    ks_h = max(20, w // 40)
    ks_v = max(20, h // 40)
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ks_h, 1)), iterations=1)
    vert  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ks_v)), iterations=1)

    # Hough transform to get precise lines
    lines_h = cv2.HoughLinesP(horiz, 1, np.pi/180, threshold=100, minLineLength=max(30, w//5), maxLineGap=10)
    lines_v = cv2.HoughLinesP(vert, 1, np.pi/180, threshold=100, minLineLength=max(30, h//5), maxLineGap=10)

    xs, ys = [], []
    if lines_v is not None:
        for l in lines_v[:,0,:]:
            x1, y1, x2, y2 = l
            if abs(x1 - x2) < 5:
                xs.append(int((x1 + x2)//2))
    if lines_h is not None:
        for l in lines_h[:,0,:]:
            x1, y1, x2, y2 = l
            if abs(y1 - y2) < 5:
                ys.append(int((y1 + y2)//2))

    # Cluster to unique grid lines
    xs = _cluster_positions(xs, tol=max(8, w//100))
    ys = _cluster_positions(ys, tol=max(8, h//100))

    # Require at least 2 lines each
    if len(xs) < 2 or len(ys) < 2:
        return None

    xs = sorted(list(set([0] + xs + [w-1])))
    ys = sorted(list(set([0] + ys + [h-1])))

    rows_text: List[List[str]] = []
    for yi in range(len(ys)-1):
        row_vals: List[str] = []
        y1, y2 = ys[yi], ys[yi+1]
        if y2 - y1 < 12:  # too small
            continue
        for xi in range(len(xs)-1):
            x1, x2 = xs[xi], xs[xi+1]
            if x2 - x1 < 10:
                row_vals.append("")
                continue
            patch = bi[y1:y2, x1:x2]
            try:
                txt = pytesseract.image_to_string(Image.fromarray(patch), config="--psm 7").strip()
            except Exception:
                txt = ""
            row_vals.append(txt)
        rows_text.append(row_vals)

    # Post: remove empty leading/trailing rows
    def empty_row(r): return all((not c or c.isspace()) for c in r)
    while rows_text and empty_row(rows_text[0]): rows_text.pop(0)
    while rows_text and empty_row(rows_text[-1]): rows_text.pop()

    if not rows_text:
        return None

    # Normalize columns to median count
    med_cols = int(np.median([len(r) for r in rows_text]))
    norm_rows = []
    for r in rows_text:
        if len(r) < med_cols: r = r + [""]*(med_cols-len(r))
        elif len(r) > med_cols: r = r[:med_cols]
        norm_rows.append(r)

    df = pd.DataFrame(norm_rows)
    return df


# ---------------------- Heuristic contour fallback ----------------------

def ocr_table_grid_to_dataframe(pil_crop: Image.Image) -> Optional[pd.DataFrame]:
    img = np.array(pil_crop.convert("L"))
    img_blur = cv2.GaussianBlur(img, (3,3), 0)
    _, bi = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = 255 - bi
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, inv.shape[1]//40), 1)), iterations=1)
    vert  = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, inv.shape[0]//40))), iterations=1)
    grid = cv2.add(horiz, vert)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 1500 or w < 20 or h < 15:  # ignore tiny
            continue
        cells.append((x,y,w,h))
    if not cells: return None
    rows: List[List[Tuple[int,int,int,int]]] = []
    cells_sorted = sorted(cells, key=lambda b: (b[1] + b[3]//2, b[0]))
    row_tol = max(8, int(0.02 * inv.shape[0]))
    for b in cells_sorted:
        yc = b[1] + b[3]//2
        placed = False
        for row in rows:
            yc0 = row[0][1] + row[0][3]//2
            if abs(yc - yc0) <= row_tol:
                row.append(b); placed = True; break
        if not placed:
            rows.append([b])
    for r in rows: r.sort(key=lambda bb: bb[0])
    col_counts = [len(r) for r in rows]
    if not col_counts: return None
    target_cols = int(np.median(col_counts))
    table_text: List[List[str]] = []
    for r in rows:
        vals: List[str] = []
        for (x, y, w, h) in r:
            patch = bi[y:y+h, x:x+w]
            try: val = pytesseract.image_to_string(Image.fromarray(patch), config="--psm 7").strip()
            except Exception: val = ""
            vals.append(val)
        if len(vals) < target_cols: vals += [""] * (target_cols - len(vals))
        elif len(vals) > target_cols: vals = vals[:target_cols]
        table_text.append(vals)
    def row_empty(rv): return all((not c or c.isspace()) for c in rv)
    while table_text and row_empty(table_text[0]): table_text.pop(0)
    while table_text and row_empty(table_text[-1]): table_text.pop()
    if not table_text: return None
    return pd.DataFrame(table_text)


# --------------------------- Stitch multi-page ---------------------------

def header_signature_from_df(df: pd.DataFrame) -> str:
    # Prefer column names; fall back to first row
    cols = [str(c).strip().lower() for c in df.columns]
    if any(c for c in cols):
        return "|".join(cols)
    if len(df) > 0:
        row0 = [str(x).strip().lower() for x in df.iloc[0].tolist()]
        return "|".join(row0)
    return ""

def stitch_tables(pieces: List[TablePiece], out_root: Path, group_prefix: str = "tablegrp") -> None:
    """
    Merge tables that continue across pages:
    - Continuation heuristic: bottom-of-page piece followed by top-of-next-page piece,
      similar header signature (fuzzy >= 85) and near-equal column count (±1).
    - Drops duplicate header rows in subsequent parts if detected.
    Saves merged CSV as group file; keeps part CSVs too.
    """
    from rapidfuzz import fuzz as rf

    # Index by page for adjacency search
    by_page: Dict[int, List[TablePiece]] = {}
    for p in pieces:
        by_page.setdefault(p.page, []).append(p)
    for lst in by_page.values():
        lst.sort(key=lambda t: t.bbox[1])  # sort by y

    visited: set[str] = set()
    group_id = 1

    for p in pieces:
        if p.part_id in visited:
            continue

        group = [p]
        visited.add(p.part_id)
        cur = p

        # Greedy forward stitching
        while True:
            next_page = cur.page + 1
            candidates = by_page.get(next_page, [])
            if not candidates:
                break
            # Choose top-of-next-page candidates only
            tops = [c for c in candidates if c.at_top and c.part_id not in visited]
            if not tops:
                break
            # Pick best header/column match
            best, best_score = None, -1.0
            for c in tops:
                if cur.header_sig and c.header_sig:
                    sim = rf.token_set_ratio(cur.header_sig, c.header_sig)
                else:
                    sim = 0
                col_ok = (cur.n_cols is not None and c.n_cols is not None and abs(cur.n_cols - c.n_cols) <= 1)
                score = sim + (10 if col_ok else 0)
                if score > best_score:
                    best, best_score = c, score
            if best is None or best_score < 80:  # threshold; tune as needed
                break
            group.append(best)
            visited.add(best.part_id)
            cur = best

        if len(group) == 1:
            # nothing to stitch; skip here (their per-part CSV already saved)
            continue

        # Merge DataFrames
        merged_df: Optional[pd.DataFrame] = None
        # Find first non-empty df as base
        for g in group:
            if g.df is not None and not g.df.empty:
                merged_df = g.df.copy()
                break
        if merged_df is None:
            continue

        # Determine header row to drop in subsequent parts
        base_sig = header_signature_from_df(merged_df)
        for g in group[1:]:
            if g.df is None or g.df.empty:
                continue
            df_part = g.df.copy()
            # Drop header row if duplicated
            part_sig_cols = "|".join([str(c).strip().lower() for c in df_part.columns])
            if base_sig and part_sig_cols and fuzz.token_set_ratio(base_sig, part_sig_cols) >= 85:
                # Headers already in columns; nothing to drop
                pass
            else:
                # Check first row equals base header; if yes, drop it
                if len(df_part) > 0:
                    row0_sig = "|".join([str(x).strip().lower() for x in df_part.iloc[0].tolist()])
                    if base_sig and fuzz.token_set_ratio(base_sig, row0_sig) >= 85:
                        df_part = df_part.iloc[1:].reset_index(drop=True)
            # Align columns by count if off by 1 (pad)
            if df_part.shape[1] < merged_df.shape[1]:
                for _ in range(merged_df.shape[1] - df_part.shape[1]):
                    df_part[df_part.shape[1]] = ""
            elif df_part.shape[1] > merged_df.shape[1]:
                df_part = df_part.iloc[:, :merged_df.shape[1]]
            merged_df = pd.concat([merged_df, df_part], ignore_index=True)

        # Save merged
        grp_name = f"{group_prefix}_{group_id:03d}.csv"
        (out_root / grp_name).write_text(merged_df.to_csv(index=False), encoding="utf-8")
        group_id += 1


# --------------------------- Pipeline -----------------------------

def run_pipeline(
    pdf_path: Path,
    outdir: Path,
    dpi: int,
    enable_tatr: bool = True,
    enable_opencv: bool = True,
    enable_stitch: bool = True,
) -> None:
    if not pdf_path.exists() or not pdf_path.is_file():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    ensure_dir(outdir)

    # Optional searchable PDF + sidecar
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

    # PaddleOCR init (version-safe)
    try:
        try:
            paddle_ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang="en",
                show_log=False,
            )
        except Exception:
            paddle_ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang="en",
            )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e

    # PP-Structure init (version-safe)
    try:
        try:
            ppstruct = PPStructureV3(layout=True, table=True, ocr=False, show_log=False, lang="en")
        except Exception:
            try:
                ppstruct = PPStructureV3(layout=True, table=True, ocr=False, lang="en")
            except Exception:
                ppstruct = PPStructureV3()
                logging.warning("PPStructureV3 running with defaults (no explicit table/layout hints).")
    except Exception as e:
        logging.warning("PP-Structure initialization failed: %s (layout disabled)", e)
        ppstruct = None  # type: ignore

    # TATR init (lazy)
    tatr_detector: Optional[TATRDetector] = None
    if enable_tatr and HAVE_TATR:
        try:
            tatr_detector = TATRDetector()
        except Exception as e:
            logging.warning("TATR unavailable: %s", e)
            tatr_detector = None

    # Render pages
    pages = render_pdf_pages(pdf_path, dpi=dpi)
    if not pages:
        raise RuntimeError("No pages rendered from input PDF.")

    report: Dict[str, Any] = {"pages": []}
    all_pieces: List[TablePiece] = []

    for idx, page in enumerate(pages, start=1):
        page_dir = outdir / "pages" / f"page_{idx:03d}"
        ensure_dir(page_dir)

        rot = auto_rotate_with_osd(page)  # color
        cleaned = basic_clean(rot)        # binarized

        # Layout on color
        if ppstruct is not None:
            layout = detect_layout_PPStructureV3(ppstruct, rot)
        else:
            layout = {"tables": [], "figures": [], "texts": []}

        # OCR text using text regions if available
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

        # Save page text
        try:
            (page_dir / "text.txt").write_text(" ".join([w.text for w in merged_words]), encoding="utf-8")
        except Exception as e:
            logging.warning("Failed to write text.txt for page %d: %s", idx, e)

        # Robust table detection
        table_boxes: List[Tuple[int,int,int,int]] = detect_tables_union(ppstruct, rot, cleaned)

        if not table_boxes and tatr_detector is not None:
            try:
                table_boxes = tatr_detector.detect(resize_long_edge(rot, 2400), score_thresh=0.7)
                if table_boxes:
                    logging.info("TATR fallback found %d table(s) on page %d.", len(table_boxes), idx)
            except Exception as e:
                logging.warning("TATR fallback failed on page %d: %s", idx, e)

        if not table_boxes and enable_opencv:
            table_boxes = detect_tables_opencv(resize_long_edge(cleaned, 2400))
            if table_boxes:
                logging.info("OpenCV heuristic found %d table(s) on page %d.", len(table_boxes), idx)

        # For PP HTML mapping
        pp_tables = extract_tables_PPStructureV3(ppstruct, rot) if ppstruct is not None else []

        # Save crops
        crop_paths = []
        for i, bb in enumerate(table_boxes, start=1):
            try:
                cp = page_dir / f"table_{i}.png"
                crop(rot, bb).save(cp)
                crop_paths.append(str(cp))
            except Exception:
                crop_paths.append(None)

        # Build TableResult list + convert to CSV/XLSX
        tables: List[TableResult] = []
        W, H = rot.size
        for i, bb in enumerate(table_boxes, start=1):
            # attach PP HTML if IoU matches
            best = None; best_iou = 0.0
            for t in pp_tables:
                ov = iou(bb, t.bbox)
                if ov > best_iou:
                    best_iou = ov; best = t
            tr = TableResult(
                bbox=bb,
                engine=(best.engine if best else "detector-union"),
                html=(best.html if best else None),
                meta=(best.meta if best else None),
                csv_paths=[],
                xlsx_paths=[],
                image_path=crop_paths[i-1],
                part_id=f"p{idx:03d}_t{i:03d}",
            )

            # 1) Try PP-Structure on the CROP for HTML
            df: Optional[pd.DataFrame] = None
            if tr.image_path:
                try:
                    crop_img = Image.open(tr.image_path).convert("RGB")
                    # Upscale before structure recon for better HTML recall
                    crop_up = resize_long_edge(crop_img, 2400)
                    if not tr.html and ppstruct is not None:
                        html = pp_table_html_from_crop(ppstruct, crop_up)
                        if html and "<table" in html.lower():
                            tr.html = html
                    if tr.html:
                        try:
                            dfs = pd.read_html(tr.html)
                            if dfs:
                                df = dfs[0]
                                csv_p = page_dir / f"{tr.part_id}.csv"
                                xlsx_p = page_dir / f"{tr.part_id}.xlsx"
                                df.to_csv(csv_p, index=False)
                                df.to_excel(xlsx_p, index=False)
                                tr.csv_paths = [str(csv_p)]; tr.xlsx_paths = [str(xlsx_p)]
                                tr.df_path = str(csv_p)
                        except Exception as e:
                            logging.debug("read_html/export failed (page %d %s): %s", idx, tr.part_id, e)
                except Exception as e:
                    logging.debug("PP structure on crop failed (page %d %s): %s", idx, tr.part_id, e)

            # 2) Grid extractor (Hough) if still no df
            if tr.df_path is None and tr.image_path:
                try:
                    df2 = grid_extract_dataframe(Image.open(tr.image_path).convert("RGB"))
                    if df2 is not None and not df2.empty:
                        csv_p = page_dir / f"{tr.part_id}.csv"
                        df2.to_csv(csv_p, index=False, header=False)
                        tr.csv_paths = [str(csv_p)]
                        tr.df_path = str(csv_p)
                        df = df2
                except Exception as e:
                    logging.debug("Grid extractor failed (page %d %s): %s", idx, tr.part_id, e)

            # 3) Contour fallback
            if tr.df_path is None and tr.image_path:
                try:
                    df3 = ocr_table_grid_to_dataframe(Image.open(tr.image_path).convert("RGB"))
                    if df3 is not None and not df3.empty:
                        csv_p = page_dir / f"{tr.part_id}.csv"
                        df3.to_csv(csv_p, index=False, header=False)
                        tr.csv_paths = [str(csv_p)]
                        tr.df_path = str(csv_p)
                        df = df3
                except Exception as e:
                    logging.debug("Contour fallback failed (page %d %s): %s", idx, tr.part_id, e)

            tables.append(tr)

            # Collect piece for stitching
            # Build header signature and meta for continuity
            at_top = (bb[1] < int(0.08 * H))
            at_bottom = (bb[1] + bb[3] > int(0.92 * H))
            header_sig = None
            n_cols = None
            if df is not None and not df.empty:
                header_sig = header_signature_from_df(df)
                n_cols = df.shape[1]
            all_pieces.append(TablePiece(
                page=idx, page_size=(W, H), bbox=bb, image_path=tr.image_path, df=df,
                header_sig=header_sig, n_cols=n_cols, at_top=at_top, at_bottom=at_bottom,
                part_id=tr.part_id
            ))

        # Figures
        figures = []
        for j, fb in enumerate(layout["figures"], start=1):
            try:
                cp = page_dir / f"figure_{j}.png"
                crop(rot, fb).save(cp)
                figures.append(FigureResult(fb, str(cp)))
            except Exception:
                pass

        # Metrics + page.json
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
            "page": idx, "avg_conf": avg_conf, "blur": blur,
            "n_words": len(merged_words), "n_tables": len(tables), "n_figures": len(figures),
        })

    # Stitch multi-page tables
    if enable_stitch and all_pieces:
        try:
            stitch_tables(all_pieces, outdir)
        except Exception as e:
            logging.warning("Stitching failed: %s", e)

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
        description="Render scanned PDFs and extract text/tables/figures with OCR (robust tables + multi-page stitch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=str, help="Path to input PDF file")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI for page images")
    parser.add_argument("--no-tatr", action="store_true", help="Disable TATR fallback (transformers/torch)")
    parser.add_argument("--no-opencv", action="store_true", help="Disable OpenCV line-heuristic fallback")
    parser.add_argument("--no-stitch", action="store_true", help="Disable multi-page table stitching")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v/-vv)")
    return parser.parse_args(argv)

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    pdf_path = Path(args.pdf).expanduser().resolve()
    outdir = Path("results") / Path(args.out)
    outdir = outdir.expanduser().resolve()

    # Render pages (pdfium -> PyMuPDF fallback is inside helper)
    try:
        run_pipeline(
            pdf_path,
            outdir,
            dpi=args.dpi,
            enable_tatr=not args.no_tatr,
            enable_opencv=not args.no_opencv,
            enable_stitch=not args.no_stitch,
        )
    except FileNotFoundError as e:
        logging.error(str(e)); sys.exit(2)
    except RuntimeError as e:
        logging.error("Runtime error: %s", e); sys.exit(3)
    except KeyboardInterrupt:
        logging.warning("Interrupted by user."); sys.exit(130)
    except Exception as e:
        logging.exception("Unexpected failure: %s", e); sys.exit(1)

if __name__ == "__main__":
    main()
