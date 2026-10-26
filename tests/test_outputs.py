from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.generator import store_image
from app.schemas import GenerationRequest


def test_store_image_uses_date_based_relative_url(tmp_path: Path):
    settings = Settings(output_dir=tmp_path, save_metadata=True)
    url = store_image(Image.new("RGB", (8, 8), "white"), "png", 42, GenerationRequest(prompt="test"), settings)
    assert url.startswith("/outputs/")
    image_path = tmp_path / url.removeprefix("/outputs/")
    assert image_path.is_file()
    assert image_path.with_suffix(".png.json").is_file()
