"""Temporary Mistral OCR scanner for scanned textbooks.

Uses Mistral OCR 3 (`mistral-ocr-2512`), the older/cheaper OCR model still
available. OCR 4 (`mistral-ocr-latest`) is more expensive and is not used.

Output is text-only Markdown (no embedded images) for Claude uploads.
Progress is saved so a long run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import fitz
from dotenv import load_dotenv

try:
    from mistralai import Mistral
except ImportError:  # newer generated SDK layout
    from mistralai.client import Mistral

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Older OCR 3 — cheaper than OCR 4 / mistral-ocr-latest.
OCR_MODEL = "mistral-ocr-2512"
DEFAULT_PDF = ROOT / "Periwinkle G5 Science.pdf"
DEFAULT_OUTPUT = ROOT / "Periwinkle G5 Science.txt"
PROGRESS_PATH = ROOT / "ocr_progress.json"
BATCH_PAGES = 5
MAX_RETRIES = 6
OCR_FILE_LIMIT_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = 48 * 1024 * 1024

IMAGE_MARKDOWN_RE = re.compile(r"!\[.*?\]\([^)]*\)", re.DOTALL)
HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE | re.DOTALL)
DATA_URI_RE = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+",
    re.IGNORECASE,
)
IMG_FILENAME_RE = re.compile(r"\bimg-\d+\.(?:jpe?g|png|gif|webp)\b", re.IGNORECASE)
TABLE_FILE_RE = re.compile(r"\[tbl-\d+\.(?:html|md)\]\(tbl-\d+\.(?:html|md)\)")


def log(message: str) -> None:
    print(message, flush=True)


def require_api_key() -> str:
    api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
    if not api_key:
        log(
            "Missing MISTRAL_API_KEY. Create a .env file in this folder "
            "(see .env.example) or set the environment variable."
        )
        sys.exit(1)
    return api_key


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def progress_path_for(pdf_path: Path) -> Path:
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in pdf_path.stem)
    return ROOT / f"ocr_progress_{safe_stem}.json"


def load_progress(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_progress(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def strip_images(markdown: str) -> str:
    """Keep OCR text only. Drop image markup, binary data, and file placeholders."""
    text = markdown or ""
    text = DATA_URI_RE.sub("", text)
    text = IMAGE_MARKDOWN_RE.sub("", text)
    text = HTML_IMG_RE.sub("", text)
    text = IMG_FILENAME_RE.sub("", text)
    text = TABLE_FILE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compress_pages_to_jpeg(pdf_path: Path, start: int, end: int, dest: Path) -> None:
    """Rasterize pages until the PDF is under the OCR upload limit."""
    src = fitz.open(pdf_path)
    try:
        for dpi, quality in ((150, 75), (120, 70), (100, 60), (72, 50)):
            out = fitz.open()
            try:
                for i in range(start, end):
                    page = src[i]
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(dpi / 72, dpi / 72),
                        colorspace=fitz.csRGB,
                        annots=True,
                    )
                    image_bytes = pix.tobytes("jpeg", jpg_quality=quality)
                    new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                    new_page.insert_image(page.rect, stream=image_bytes)
                dest.parent.mkdir(parents=True, exist_ok=True)
                out.save(dest, garbage=4, deflate=True)
            finally:
                out.close()
            size = dest.stat().st_size
            log(f"Compressed pages {start + 1}-{end} at {dpi}dpi q={quality}: {size / 1_048_576:.1f} MB")
            if size <= MAX_UPLOAD_BYTES:
                return
        raise RuntimeError(
            f"Could not compress {pdf_path.name} pages {start + 1}-{end} under 48 MB"
        )
    finally:
        src.close()


def make_uploadable_chunk(pdf_path: Path, start: int, end: int, dest: Path) -> None:
    extract_page_range(pdf_path, start, end, dest)
    size = dest.stat().st_size
    if size <= MAX_UPLOAD_BYTES:
        return
    log(f"{dest.name} is {size / 1_048_576:.1f} MB; compressing for OCR upload...")
    compress_pages_to_jpeg(pdf_path, start, end, dest)


def extract_page_range(pdf_path: Path, start: int, end: int, dest: Path) -> None:
    """Write pages [start, end) to dest. Page indexes are 0-based."""
    src = fitz.open(pdf_path)
    out = fitz.open()
    try:
        out.insert_pdf(src, from_page=start, to_page=end - 1)
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()


def upload_pdf(client: Mistral, pdf_path: Path) -> str:
    log(f"Uploading {pdf_path.name} ({pdf_path.stat().st_size / 1_048_576:.1f} MB)...")
    uploaded = client.files.upload(
        file={
            "file_name": pdf_path.name,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )
    file_id = getattr(uploaded, "id", None) or uploaded.get("id")
    log(f"Uploaded file_id={file_id}")
    return file_id


def signed_url_for(client: Mistral, file_id: str) -> str:
    signed = client.files.get_signed_url(file_id=file_id)
    return getattr(signed, "url", None) or signed.get("url")


def ocr_document(
    client: Mistral,
    document_url: str,
    pages: str | None = None,
):
    kwargs = {
        "model": OCR_MODEL,
        "document": {
            "type": "document_url",
            "document_url": document_url,
        },
        "include_image_base64": False,
        "image_limit": 0,  # do not extract figures; text only
    }
    if pages is not None:
        kwargs["pages"] = pages
    return client.ocr.process(**kwargs)


def with_retries(fn, *, label: str):
    delay = 4
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — surface API/network errors and retry
            last_error = exc
            message = str(exc)
            retryable = any(
                token in message.lower()
                for token in ("429", "rate", "timeout", "503", "502", "500", "temporarily")
            )
            log(f"{label} failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
            if attempt == MAX_RETRIES or not retryable:
                raise
            log(f"Retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise last_error


def pages_markdown(ocr_response) -> list[tuple[int, str]]:
    results = []
    for page in ocr_response.pages:
        index = getattr(page, "index", 0)
        markdown = getattr(page, "markdown", "") or ""
        # Ignore page.images entirely — never write binary or figure files.
        results.append((index, strip_images(markdown)))
    return results


def append_output(output_path: Path, chunks: list[tuple[int, str]], offset: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, text in chunks:
            page_no = offset + index + 1
            handle.write(f"\n\n===== PAGE {page_no} =====\n\n")
            handle.write(text if text else "[no text extracted]")
            handle.write("\n")


def ocr_via_page_param(
    client: Mistral,
    file_id: str,
    start: int,
    end: int,
) -> list[tuple[int, str]]:
    url = signed_url_for(client, file_id)
    pages = f"{start}-{end - 1}"
    response = with_retries(
        lambda: ocr_document(client, url, pages=pages),
        label=f"OCR pages {start + 1}-{end}",
    )
    return pages_markdown(response)


def ocr_via_split(
    client: Mistral,
    pdf_path: Path,
    start: int,
    end: int,
    tmp_dir: Path,
) -> list[tuple[int, str]]:
    chunk_path = tmp_dir / f"chunk_{start + 1:03d}_{end:03d}.pdf"
    make_uploadable_chunk(pdf_path, start, end, chunk_path)
    file_id = upload_pdf(client, chunk_path)
    url = signed_url_for(client, file_id)
    response = with_retries(
        lambda: ocr_document(client, url),
        label=f"OCR split pages {start + 1}-{end}",
    )
    try:
        client.files.delete(file_id=file_id)
    except Exception:  # noqa: BLE001
        pass
    chunk_path.unlink(missing_ok=True)
    return pages_markdown(response)


def run(pdf_path: Path, output_path: Path, start_page: int, end_page: int | None) -> None:
    api_key = require_api_key()
    total = page_count(pdf_path)
    start = max(0, start_page)
    end = total if end_page is None else min(end_page, total)
    if start >= end:
        log(f"Nothing to do: start={start}, end={end}, total={total}")
        return

    log(f"PDF: {pdf_path}")
    log(f"Pages: {total} (scanning {start + 1} to {end})")
    log(f"Model: {OCR_MODEL}")
    log(f"Output: {output_path}")

    client = Mistral(api_key=api_key)
    progress_file = progress_path_for(pdf_path)
    progress = load_progress(progress_file)
    resume_from = progress.get("next_page", start)
    if progress.get("pdf") == str(pdf_path) and resume_from > start:
        start = resume_from
        log(f"Resuming from page {start + 1}")
    elif output_path.exists() and start == 0 and resume_from == 0:
        output_path.write_text("", encoding="utf-8")

    tmp_dir = ROOT / "_ocr_chunks"
    tmp_dir.mkdir(exist_ok=True)

    source_size = pdf_path.stat().st_size
    bytes_per_page = source_size / max(total, 1)
    batch_pages = 1 if bytes_per_page > 8 * 1024 * 1024 else BATCH_PAGES

    file_id = progress.get("file_id")
    use_page_param = progress.get("use_page_param", True)
    if source_size >= OCR_FILE_LIMIT_BYTES:
        log("Source PDF is over 50 MB; using page splits instead of a full-file upload.")
        use_page_param = False
        file_id = None
    elif use_page_param and not file_id:
        try:
            file_id = upload_pdf(client, pdf_path)
            use_page_param = True
        except Exception as exc:  # noqa: BLE001
            log(f"Full-file upload failed ({exc}); falling back to page splits.")
            use_page_param = False
            file_id = None

    if start == 0 and not output_path.exists():
        output_path.write_text(
            f"# {pdf_path.stem}\n\nExtracted with {OCR_MODEL}. Images omitted.\n",
            encoding="utf-8",
        )

    current = start
    while current < end:
        batch_end = min(current + batch_pages, end)
        log(f"Scanning pages {current + 1}-{batch_end} / {end}...")
        try:
            if use_page_param and file_id:
                chunks = ocr_via_page_param(client, file_id, current, batch_end)
                # API returns page.index relative to the requested set or the
                # original document. Normalize against this batch.
                if chunks and chunks[0][0] >= current:
                    chunks = [(idx - current, text) for idx, text in chunks]
            else:
                chunks = ocr_via_split(client, pdf_path, current, batch_end, tmp_dir)
        except Exception as exc:  # noqa: BLE001
            if use_page_param:
                log(f"Page-range OCR failed ({exc}); switching to split PDFs.")
                use_page_param = False
                continue
            raise

        append_output(output_path, chunks, offset=current)
        current = batch_end
        save_progress(
            progress_file,
            {
                "pdf": str(pdf_path),
                "next_page": current,
                "total_pages": total,
                "file_id": file_id,
                "use_page_param": use_page_param,
                "model": OCR_MODEL,
            },
        )
        log(f"Wrote through page {current}.")
        time.sleep(1)

    if file_id:
        try:
            client.files.delete(file_id=file_id)
        except Exception:  # noqa: BLE001
            pass

    size_kb = output_path.stat().st_size / 1024
    log(f"Done. {output_path.name} is {size_kb:.1f} KB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR a scanned PDF with Mistral OCR 3.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-page", type=int, default=0, help="0-based start page")
    parser.add_argument("--end-page", type=int, default=None, help="0-based exclusive end page")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.pdf, args.output, args.start_page, args.end_page)
