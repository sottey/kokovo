import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.generator import ImageGenerator, latest_output_url, store_image
from app.main import home
from app.schemas import GenerationRequest


def test_store_image_uses_date_based_relative_url(tmp_path: Path):
    settings = Settings(output_dir=tmp_path, save_metadata=True)
    url = store_image(Image.new("RGB", (8, 8), "white"), "png", 42, GenerationRequest(prompt="test"), settings)
    assert url.startswith("/outputs/")
    image_path = tmp_path / url.removeprefix("/outputs/")
    assert image_path.is_file()
    assert image_path.with_suffix(".png.json").is_file()


def test_latest_output_ignores_metadata_and_returns_newest_image(tmp_path: Path):
    settings = Settings(output_dir=tmp_path, save_metadata=True)
    first_url = store_image(Image.new("RGB", (8, 8), "red"), "png", 1, GenerationRequest(prompt="one"), settings)
    second_url = store_image(Image.new("RGB", (8, 8), "blue"), "png", 2, GenerationRequest(prompt="two"), settings)
    assert latest_output_url(tmp_path) == second_url
    assert first_url.startswith("/outputs/")


def test_home_page_embeds_the_latest_image_url(tmp_path: Path):
    settings = Settings(output_dir=tmp_path)
    url = store_image(Image.new("RGB", (8, 8), "green"), "png", 3, GenerationRequest(prompt="three"), settings)
    assert url in home(settings).body.decode("utf-8")


def test_reference_image_is_decoded_and_fitted_to_requested_size(tmp_path: Path):
    buffer = io.BytesIO()
    Image.new("RGB", (160, 80), "purple").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    generator = ImageGenerator(Settings(output_dir=tmp_path))

    reference = generator._reference_image(encoded, 64, 64)

    assert reference.size == (64, 64)
