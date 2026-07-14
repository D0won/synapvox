import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

import fitz
from PIL import Image

import backend.stt.pdf_extractor as pdf_extractor
from backend.stt.pdf_extractor import extract_pdf


def _make_text_pdf(path, text="REST API 개요"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def _make_pdf_with_image(path, text="REST API 개요"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)

    image_buf = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(image_buf, format="PNG")
    page.insert_image(fitz.Rect(72, 100, 172, 200), stream=image_buf.getvalue())

    doc.save(path)
    doc.close()


def test_extract_pdf_returns_text_per_page(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_text_pdf(pdf_path)

    result = extract_pdf(str(pdf_path), describe_images=False)

    assert result["source"] == "sample.pdf"
    assert len(result["pages"]) == 1
    assert result["pages"][0]["page_number"] == 1
    assert "REST API" in result["pages"][0]["text"]


def test_extract_pdf_describes_embedded_images(tmp_path, monkeypatch):
    pdf_path = tmp_path / "with_image.pdf"
    _make_pdf_with_image(pdf_path)

    monkeypatch.setattr(pdf_extractor, "describe_image", lambda blob, mime_type: "2026년 예산 차트")

    result = extract_pdf(str(pdf_path))

    assert "REST API" in result["pages"][0]["text"]
    assert "[이미지 설명: 2026년 예산 차트]" in result["pages"][0]["text"]


def test_extract_pdf_skips_image_description_when_disabled(tmp_path, monkeypatch):
    pdf_path = tmp_path / "with_image.pdf"
    _make_pdf_with_image(pdf_path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("describe_image should not be called when describe_images=False")

    monkeypatch.setattr(pdf_extractor, "describe_image", _fail_if_called)

    result = extract_pdf(str(pdf_path), describe_images=False)

    assert "[이미지 설명" not in result["pages"][0]["text"]
