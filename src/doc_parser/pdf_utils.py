"""PDF and image decoding helpers used by every OCR pyfunc.

Why pypdfium2: zero system-level dependencies (no poppler), permissive license,
fast C++ rasterizer. PIL handles raster images. Together they let the pyfunc
accept PDFs/PNGs/JPEGs uniformly via base64-encoded bytes.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

PDF_MAGIC = b"%PDF"
DEFAULT_PDF_DPI = 200
PDF_RENDER_SCALE = DEFAULT_PDF_DPI / 72.0


def decode_b64(image_b64: str) -> bytes:
    """Decode base64 string (with or without data URL prefix) to raw bytes."""
    if image_b64 is None:
        raise ValueError("image_b64 is None")
    s = image_b64.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s, validate=False)


def is_pdf(data: bytes) -> bool:
    return len(data) >= 4 and data[:4] == PDF_MAGIC


def pdf_bytes_to_images(
    data: bytes,
    *,
    dpi: int = DEFAULT_PDF_DPI,
    max_pages: int | None = None,
) -> list[Image]:
    """Rasterize a PDF byte string into a list of PIL.Image (RGB) pages."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        n = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        scale = dpi / 72.0
        pages: list[Image] = []
        for i in range(n):
            page = pdf[i]
            try:
                pil_image = page.render(scale=scale).to_pil().convert("RGB")
                pages.append(pil_image)
            finally:
                page.close()
        return pages
    finally:
        pdf.close()


def image_bytes_to_image(data: bytes) -> Image:
    """Open raster image bytes (PNG/JPEG/TIFF/etc.) as an RGB PIL.Image."""
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(data))
    return img.convert("RGB")


def decode_pages(
    image_b64: str,
    *,
    dpi: int = DEFAULT_PDF_DPI,
    max_pages: int | None = None,
) -> list[Image]:
    """Decode one base64 input (PDF or single image) to a list of PIL.Image pages."""
    raw = decode_b64(image_b64)
    if is_pdf(raw):
        return pdf_bytes_to_images(raw, dpi=dpi, max_pages=max_pages)
    return [image_bytes_to_image(raw)]
