import argparse
import json
from pathlib import Path

import fitz
import pdfplumber

from .image_description import describe_image


def extract_pdf(path: str, describe_images: bool = True) -> dict:
    """Text via pdfplumber (preferred over pypdf — pypdf drops inter-word spacing on
    Korean text, see synapvox_Local/test_data/Congress/README.md). Images via PyMuPDF
    (fitz), since pdfplumber doesn't expose embedded image bytes directly.
    describe_images=True calls a vision LLM (image_description.describe_image, costs
    money) for every embedded image — set False to skip images entirely (text-only,
    free)."""
    with pdfplumber.open(path) as pdf:
        page_texts = [(page.extract_text() or "").strip() for page in pdf.pages]

    if describe_images:
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            descriptions = []
            for xref, *_ in page.get_images(full=True):
                extracted = doc.extract_image(xref)
                descriptions.append(describe_image(extracted["image"], f"image/{extracted['ext']}"))
            if descriptions:
                image_block = "\n".join(f"[이미지 설명: {d}]" for d in descriptions)
                page_texts[i] = f"{page_texts[i]}\n{image_block}".strip() if page_texts[i] else image_block
        doc.close()

    pages = [{"page_number": i + 1, "text": text} for i, text in enumerate(page_texts)]
    return {"source": Path(path).name, "pages": pages}


def main():
    parser = argparse.ArgumentParser(description="Extract text (+ image descriptions) from a .pdf file")
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--no-image-description",
        action="store_true",
        help="Skip vision-LLM image descriptions (text-only, free, old behavior)",
    )
    parser.add_argument("-o", "--output", help="Write JSON to this path instead of stdout")
    args = parser.parse_args()

    result = extract_pdf(args.pdf_path, describe_images=not args.no_image_description)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
