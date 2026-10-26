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


def test_api_key_rejects_missing_credentials():
    with pytest.raises(HTTPException) as error:
        require_api_key(credentials=None, settings=Settings(api_key="secret"))
    assert error.value.status_code == 401


def test_api_key_can_be_explicitly_disabled():
    require_api_key(credentials=None, settings=Settings(require_api_key=False))
