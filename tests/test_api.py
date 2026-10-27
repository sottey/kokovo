import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.auth import require_api_key
from app.config import Settings
from app.schemas import GenerationRequest


def test_generation_dimensions_must_be_divisible_by_eight():
    with pytest.raises(ValidationError, match="divisible by 8"):
        GenerationRequest(prompt="test", width=513)


def test_generation_request_allows_supported_format():
    request = GenerationRequest(prompt="test", width=512, height=512, output_format="webp")
    assert request.output_format == "webp"


def test_reference_image_requires_enough_effective_steps():
    with pytest.raises(ValidationError, match="reference_strength"):
        GenerationRequest(prompt="test", reference_image="dGVzdA==", reference_strength=0.5, num_inference_steps=1)


def test_reference_image_allows_enough_effective_steps():
    request = GenerationRequest(prompt="test", reference_image="dGVzdA==", reference_strength=0.5, num_inference_steps=2)
    assert request.reference_strength == 0.5


def test_api_key_rejects_missing_credentials():
    with pytest.raises(HTTPException) as error:
        require_api_key(credentials=None, settings=Settings(api_key="secret"))
    assert error.value.status_code == 401


def test_api_key_can_be_explicitly_disabled():
    require_api_key(credentials=None, settings=Settings(require_api_key=False))
