# Extract All From Scanned PDF

A robust, production-ready CLI pipeline to turn scanned PDFs into structured artifacts:

- High-quality page rendering (pypdfium2; fallback to PyMuPDF).
- Auto-rotation via Tesseract OSD.
- Gentle denoise + adaptive threshold for OCR.
- Dual OCR (Tesseract + PaddleOCR) with smart reconciliation.
- Layout detection (PP-Structure V3) → tables & figures.
- Table HTML → CSV/XLSX via `pandas.read_html`.
- Optional searchable PDF + sidecar via `ocrmypdf`.
- Per-page artifacts (`text.txt`, `page.json`, `figure_*.png`, `table_*.csv/.xlsx`) and a top-level `report.json`.

## Output structure

```
out/
  report.json
  searchable.pdf           # if OCRmyPDF is available
  sidecar.txt              # if OCRmyPDF is available
  pages/
    page_001/
      text.txt
      page.json
      figure_1.png
      table_1.csv
      table_1.xlsx
    page_002/
      ...
```

---

## Installation

> **Python**: 3.9 – 3.11 recommended

1) **Create & activate a virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) **Install Python packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) **Install native dependencies**
- **Tesseract OCR (required for Tesseract engine + OSD)**
  - **Windows**: Install Tesseract and ensure the `tesseract.exe` directory is on your `PATH`.
  - **macOS**: `brew install tesseract`
  - **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`
- **OCRmyPDF (optional, but recommended for searchable PDFs)**
  - Requires Ghostscript and related libs.
  - macOS (Homebrew): `brew install ocrmypdf`
  - Debian/Ubuntu: `sudo apt-get install ocrmypdf`
  - Windows: Use prebuilt binaries from the project or `pip install ocrmypdf` (ensure dependencies available).
- **(Optional) PyMuPDF**: Installed via `pip`; no extra system deps for most platforms.

> If `paddleocr` downloads models on first run, it will cache under your user directory (e.g., `~/.paddleocr` or platform equivalent).

---

## Usage

```bash
python extract_all_from_scanned_pdf.py input.pdf --out out --dpi 350 -v
```

- `--out` (default: `out`) – output directory
- `--dpi` (default: `350`) – render resolution; 300–400 is usually ideal for OCR
- `-v` (info logs) / `-vv` (debug logs)

### Examples

```bash
# Basic
python extract_all_from_scanned_pdf.py ./docs/invoice_scan.pdf

# Custom output dir and verbose logging
python extract_all_from_scanned_pdf.py ./docs/report.pdf --out ./trial_out -vv

# Higher DPI for tiny print
python extract_all_from_scanned_pdf.py ./docs/bank_statement.pdf --dpi 400
```

---

## Notes & Tips

- **Tesseract path**: If Tesseract is installed but not on PATH (common on Windows),
  set it in code before importing `pytesseract`:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- **Performance**:
  - Increase DPI for better accuracy on small text (costs more RAM/CPU).
  - PaddleOCR model downloads once; subsequent runs are faster.
- **Layout detection**:
  - If PP-Structure fails to initialize, the pipeline still runs OCR and figure extraction is skipped; a warning is logged.
- **Tables**:
  - When PP-Structure returns HTML, the script exports `CSV` and `XLSX` using the first `pandas.read_html` dataframe.
  - Complex tables may need post-processing.
- **Figures**:
  - Regions classified as image/picture/figure are cropped and saved as PNGs.

---

## Troubleshooting

- **`TesseractNotFoundError`**  
  Install Tesseract and ensure it’s on PATH (or set `pytesseract.pytesseract.tesseract_cmd` as shown above).

- **`pypdfium2` render fails**  
  The script will try PyMuPDF if installed. Otherwise, install PyMuPDF or resolve the original error.

- **`OCRmyPDF` not found**  
  It’s optional. The pipeline continues without it; you just won’t get `searchable.pdf` and `sidecar.txt`.

- **First run is slow**  
  PaddleOCR downloads models on first use.

---

## Development

- Code style: `ruff` / `black` friendly.
- Python typing: standard `typing` hints + `dataclasses`.
