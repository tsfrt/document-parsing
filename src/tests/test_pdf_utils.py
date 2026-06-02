"""Tests for pdf_utils: base64 decoding, PDF rasterization, PNG/JPEG handling."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from doc_parser import pdf_utils


def _png_bytes(size=(64, 32), color=(0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(48, 24), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_pdf(num_pages: int = 2) -> bytes:
    pypdfium2 = pytest.importorskip("pypdfium2")
    pdf = pypdfium2.PdfDocument.new()
    try:
        for _ in range(num_pages):
            pdf.new_page(200, 100)
        out = io.BytesIO()
        pdf.save(out)
        return out.getvalue()
    finally:
        pdf.close()


class TestDecodeB64:
    def test_plain_base64_roundtrip(self):
        raw = b"hello bytes"
        encoded = base64.b64encode(raw).decode()
        assert pdf_utils.decode_b64(encoded) == raw

    def test_strips_data_url_prefix(self):
        raw = b"abc123"
        encoded = "data:image/png;base64," + base64.b64encode(raw).decode()
        assert pdf_utils.decode_b64(encoded) == raw

    def test_none_input_raises(self):
        with pytest.raises(ValueError):
            pdf_utils.decode_b64(None)  # type: ignore[arg-type]


class TestIsPdf:
    def test_png_is_not_pdf(self):
        assert not pdf_utils.is_pdf(_png_bytes())

    def test_pdf_magic_detected(self):
        assert pdf_utils.is_pdf(b"%PDF-1.7\n%...\n")

    def test_short_input_handled(self):
        assert not pdf_utils.is_pdf(b"")
        assert not pdf_utils.is_pdf(b"%PD")


class TestImageBytesToImage:
    def test_png_decodes_to_rgb(self):
        img = pdf_utils.image_bytes_to_image(_png_bytes(size=(10, 5)))
        assert img.mode == "RGB"
        assert img.size == (10, 5)

    def test_jpeg_decodes_to_rgb(self):
        img = pdf_utils.image_bytes_to_image(_jpeg_bytes(size=(7, 9)))
        assert img.mode == "RGB"
        assert img.size == (7, 9)


class TestDecodePages:
    def test_decode_single_image(self):
        b64 = base64.b64encode(_png_bytes()).decode()
        pages = pdf_utils.decode_pages(b64)
        assert len(pages) == 1
        assert pages[0].mode == "RGB"

    def test_decode_pdf_returns_one_image_per_page(self):
        pypdfium2 = pytest.importorskip("pypdfium2")
        del pypdfium2  # ensure import side-effects ran
        pdf_bytes = _make_pdf(num_pages=3)
        b64 = base64.b64encode(pdf_bytes).decode()
        pages = pdf_utils.decode_pages(b64, dpi=72)
        assert len(pages) == 3
        for p in pages:
            assert p.mode == "RGB"

    def test_decode_pdf_max_pages(self):
        pytest.importorskip("pypdfium2")
        pdf_bytes = _make_pdf(num_pages=4)
        b64 = base64.b64encode(pdf_bytes).decode()
        pages = pdf_utils.decode_pages(b64, dpi=72, max_pages=2)
        assert len(pages) == 2
