import os, sys, json, hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

# Rendering (primary)
import pypdfium2 as pdfium  # pip install pypdfium2

# OCR engines
import pytesseract          # pip install pytesseract (requires Tesseract installed)
from paddleocr import PaddleOCR, PPStructureV3  # pip install paddleocr

# Layout / reconciliation
from rapidfuzz import fuzz
import pandas as pd

# Optional
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    import layoutparser as lp  # optional
    HAVE_LAYOUTPARSER = True
except Exception:
    HAVE_LAYOUTPARSER = False

# Optional: Hugging Face TATR (second opinion for tables)
HAVE_TATR = False
try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    import torch
    HAVE_TATR = True
except Exception:
    pass

# Optional: OCRmyPDF for searchable PDF + sidecar
try:
    import ocrmypdf
    HAVE_OCRMYPDF = True
except Exception:
    HAVE_OCRMYPDF = False  # fixed typo

@dataclass
class Word:
    text: str
    conf: float
    bbox: Tuple[int,int,int,int]  # x, y, w, h
    engine: str

@dataclass
class TableResult:
    bbox: Tuple[int,int,int,int]
    engine: str
    html: Optional[str] = None
    csv_paths: Optional[List[str]] = None
    xlsx_paths: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class FigureResult:
    bbox: Tuple[int,int,int,int]
    path: str
    engine: str = "layout"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def md5_bytes(b: bytes):
    m = hashlib.md5(); m.update(b); return m.hexdigest()

def render_pdf_pages(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """Render all pages to PIL images with pypdfium2."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    images = []
    scale = dpi / 72.0  # PDF point is 1/72 inch
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        images.append(pil.convert("RGB"))
    return images

def auto_rotate_with_osd(pil_img: Image.Image) -> Image.Image:
    """Use Tesseract OSD to auto-rotate page (safe if OSD fails)."""
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
            pil_img = pil_img.rotate(-angle, expand=True)
    except Exception:
        pass
    return pil_img

def basic_clean(pil_img: Image.Image) -> Image.Image:
    """Light denoise + binarize; keep gentle to avoid killing fine lines."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 50, 50)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    return Image.fromarray(th)

def tess_words(pil_img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> List[Word]:
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=cfg)
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        if not txt: continue
        conf = float(data["conf"][i]) if data["conf"][i] != '-1' else 0.0
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append(Word(txt, conf, (x, y, w, h), engine="tesseract"))
    return words

def paddle_words(paddle_ocr: PaddleOCR, pil_img: Image.Image) -> List[Word]:
    # Ensure 3-channel RGB for PaddleOCR
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Try the new API first (no kwargs), fall back to legacy .ocr(img)
    result = None
    if hasattr(paddle_ocr, "predict"):
        try:
            result = paddle_ocr.predict(img)  # new API, no 'cls' kw
        except TypeError:
            result = None
    if result is None:
        result = paddle_ocr.ocr(img)  # legacy API, no kwargs

    words = []
    # Normalize typical PaddleOCR result structure:
    # result ~ [ [ (box_points), (text, conf) ], ... ] per line
    for line in result:
        if not line:
            continue
        for item in line:
            if not item or len(item) < 2:
                continue
            box, info = item[0], item[1]
            if not isinstance(box, (list, tuple)) or not isinstance(info, (list, tuple)) or len(info) < 2:
                continue
            txt, conf = info[0], info[1]
            # box is 4 points: ((x1,y1),(x2,y2),(x3,y3),(x4,y4))
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
            x = int(min(x1, x2, x3, x4)); y = int(min(y1, y2, y3, y4))
            w = int(max(x1, x2, x3, x4)) - x; h = int(max(y1, y2, y3, y4)) - y
            words.append(Word(str(txt).strip(), float(conf) * 100.0, (x, y, w, h), engine="paddleocr"))
    return words

def iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa=max(x1,x2); ya=max(y1,y2); xb=min(x1+w1,x2+w2); yb=min(y1+h1,y2+h2)
    if xb<=xa or yb<=ya: return 0.0
    inter=(xb-xa)*(yb-ya)
    a1=w1*h1; a2=w2*h2
    return inter / float(a1 + a2 - inter)

def reconcile_words(tess: List[Word], padd: List[Word], digit_bias=True) -> List[Word]:
    """Merge two word lists; prefer higher conf; for digit-heavy tokens, prefer the one with more digits and better similarity to the other."""
    merged=[]
    used=set()
    for tw in tess:
        best=None; best_j=-1; best_iou=0
        for j, pw in enumerate(padd):
            if j in used: continue
            ov=iou(tw.bbox, pw.bbox)
            if ov>best_iou:
                best_iou=ov; best=(pw); best_j=j
        candidate = tw
        if best and best_iou>0.2:
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

# ---- NEW: ensure 3-channel BGR for PP-Structure ----
def to_bgr3(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    if arr.ndim == 2:
        # grayscale -> BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        # RGBA -> BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        # RGB -> BGR
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

# ---------- Version-agnostic PPStructureV3 runner ----------
def ppstruct_run(ppstruct, img_bgr):
    """
    Call PPStructureV3 regardless of interface differences across versions.
    Input must be BGR 3-channel uint8.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("ppstruct_run expects a 3-channel BGR image.")
    if img_bgr.dtype != np.uint8:
        img_bgr = img_bgr.astype(np.uint8)

    # 1) Try callable instance
    try:
        return ppstruct(img_bgr)
    except TypeError:
        pass
    except Exception:
        raise

    # 2) Try .predict
    if hasattr(ppstruct, "predict"):
        return ppstruct.predict(img_bgr)
    # 3) Try .structure
    if hasattr(ppstruct, "structure"):
        return ppstruct.structure(img_bgr)
    # 4) Try .inference
    if hasattr(ppstruct, "inference"):
        return ppstruct.inference(img_bgr)
    raise RuntimeError("Unsupported PPStructureV3 interface: not callable and no known predict/structure/inference method found.")

def detect_layout_PPStructureV3(ppstruct, pil_img: Image.Image):
    """
    Use PP-Structure to get regions (table/picture/text) via ppstruct_run().
    Always feed BGR 3-channel.
    """
    img_bgr = to_bgr3(pil_img)
    results = ppstruct_run(ppstruct, img_bgr)
    layout = {"tables": [], "figures": [], "texts": []}
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("res", []) or []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get('type') or r.get('label') or 'text'
        bbox_raw = r.get('bbox') or r.get('box') or r.get('rect')
        if bbox_raw is None:
            continue
        bbox = tuple(map(int, bbox_raw))
        if t == 'table':
            layout["tables"].append(bbox)
        elif str(t).lower() in ('figure','image','picture','pic','chart'):
            layout["figures"].append(bbox)
        else:
            layout["texts"].append(bbox)
    return layout

def crop(pil_img: Image.Image, bbox):
    x,y,w,h = bbox
    return pil_img.crop((x,y,x+w,y+h))

def extract_tables_PPStructureV3(ppstruct, pil_img: Image.Image, page_dir: Path) -> List[TableResult]:
    """
    Reuse the same instance and pull out only table items.
    """
    img_bgr = to_bgr3(pil_img)
    results = ppstruct_run(ppstruct, img_bgr)
    out = []
    if isinstance(results, dict):
        results = results.get("results", []) or results.get("res", []) or []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get('type') or r.get('label')
        if t != 'table':
            continue
        bbox_raw = r.get('bbox') or r.get('box') or r.get('rect')
        if bbox_raw is None:
            continue
        bbox = tuple(map(int, bbox_raw))
        meta = {k:r[k] for k in r if k not in ('img','res','html')}
        html = None
        res = r.get('res')
        if isinstance(res, dict):
            html = res.get('html') or res.get('table', {}).get('html')
        elif isinstance(r.get('html'), str):
            html = r.get('html')
        tr = TableResult(bbox=bbox, engine="pp-structure", html=html, meta=meta,
                         csv_paths=[], xlsx_paths=[])
        out.append(tr)
    return out

def save_figures(pil_img: Image.Image, figure_bboxes: List[Tuple[int,int,int,int]], page_dir: Path) -> List[FigureResult]:
    figs=[]
    for i, bb in enumerate(figure_bboxes, start=1):
        crop_img = crop(pil_img, bb)
        p = page_dir / f"figure_{i}.png"
        crop_img.save(p)
        figs.append(FigureResult(bb, str(p)))
    return figs

def page_quality(pil_img: Image.Image):
    """Simple blur score via variance of Laplacian; higher is sharper."""
    lap = cv2.Laplacian(np.array(pil_img.convert("L")), cv2.CV_64F).var()
    return float(lap)

def main(pdf_path: str, outdir: str = "out", dpi: int = 350):
    pdf_path = Path(pdf_path); out = Path(outdir); ensure_dir(out)

    # Optional searchable PDF + sidecar (prepass)
    if HAVE_OCRMYPDF:
        try:
            ocrmypdf.ocr(
                str(pdf_path), str(out / "searchable.pdf"),
                force_ocr=True, deskew=True, rotate_pages=True, clean=True,
                sidecar=str(out/"sidecar.txt")
            )
        except Exception:
            pass

    # Initialize OCR engines
    paddle_ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        lang='en'
    )
    # Single PPStructure instance (no kwargs -> compatible across versions)
    ppstruct = PPStructureV3()

    # Render pages
    pages = render_pdf_pages(pdf_path, dpi=dpi)

    report = {"pages": []}
    for idx, page in enumerate(pages, start=1):
        page_dir = out / "pages" / f"page_{idx:03d}"
        ensure_dir(page_dir)

        # Orientation + cleaning
        rot = auto_rotate_with_osd(page)   # keep color
        cleaned = basic_clean(rot)         # binarized (grayscale)

        # Layout detection on COLOR page (not cleaned)
        layout = detect_layout_PPStructureV3(ppstruct, rot)

        # OCR – use text boxes from layout; OCR on cleaned for quality
        if layout["texts"]:
            tess_words_all=[]; padd_words_all=[]
            for tb in layout["texts"]:
                roi_clean = crop(cleaned, tb)
                tess_words_all += tess_words(roi_clean, psm=6)
                padd_words_all += paddle_words(paddle_ocr, roi_clean)
        else:
            tess_words_all = tess_words(cleaned, psm=6)
            padd_words_all = paddle_words(paddle_ocr, cleaned)

        merged_words = reconcile_words(tess_words_all, padd_words_all, digit_bias=True)

        # Save plain text
        text_out = " ".join([w.text for w in merged_words])
        (page_dir / "text.txt").write_text(text_out, encoding="utf-8")

        # Tables (also run on COLOR page)
        tables = extract_tables_PPStructureV3(ppstruct, rot, page_dir)

        # Save simple CSVs from HTML if present
        for ti, t in enumerate(tables, start=1):
            if t.html:
                try:
                    dfs = pd.read_html(t.html)
                    if len(dfs):
                        df = dfs[0]
                        csv_p = page_dir / f"table_{ti}.csv"
                        xlsx_p = page_dir / f"table_{ti}.xlsx"
                        df.to_csv(csv_p, index=False)
                        df.to_excel(xlsx_p, index=False)
                        t.csv_paths = [str(csv_p)]; t.xlsx_paths = [str(xlsx_p)]
                except Exception:
                    pass

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
        (page_dir / "page.json").write_text(json.dumps(page_json, indent=2), encoding="utf-8")
        report["pages"].append({
            "page": idx,
            "avg_conf": avg_conf,
            "blur": blur,
            "n_words": len(merged_words),
            "n_tables": len(tables),
            "n_figures": len(figures)
        })

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done. See: {out.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_all_from_scanned_pdf.py <input.pdf> [outdir] [dpi]")
        sys.exit(1)
    pdf = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "out"
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 350
    main(pdf, outdir, dpi)
