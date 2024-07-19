import pytest
from src.pdf_extractor import extract_text_from_pdf

def test_extract_text_from_pdf():
    text = extract_text_from_pdf('../data/sample_pdf.pdf')
    assert len(text) > 0
    assert isinstance(text, str)