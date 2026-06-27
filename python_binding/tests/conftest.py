"""Shared pytest fixtures for pipeline tests."""
from pathlib import Path
import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory, auto-cleaned."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


@pytest.fixture
def test_image_path():
    """Path to a test image if available; skip otherwise."""
    candidates = [
        Path("/home/nianliu/tmp/pp/pdf_render_demo/pymupdf_page00.png"),
        Path("data/vision/test.png"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    pytest.skip("No test image available")
