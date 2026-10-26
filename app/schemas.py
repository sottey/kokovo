from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    negative_prompt: str | None = Field(default=None, max_length=4_000)
    prompt_2: str | None = Field(default=None, max_length=4_000)
    negative_prompt_2: str | None = Field(default=None, max_length=4_000)
    width: int | None = Field(default=None, ge=64)
    height: int | None = Field(default=None, ge=64)
    num_inference_steps: int = Field(default=4, ge=1)
    guidance_scale: float = Field(default=0.0, ge=0, le=30)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    num_images: int = Field(default=1, ge=1)
    output_format: Literal["png", "jpeg", "webp"] = "png"
    response_format: Literal["b64_json", "url"] = "b64_json"
    scheduler: str | None = Field(default=None, max_length=80)
    clip_skip: int | None = Field(default=None, ge=1, le=12)
    high_noise_frac: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def validate_dimensions(self) -> "GenerationRequest":
        for name, value in (("width", self.width), ("height", self.height)):
            if value is not None and value % 8:
                raise ValueError(f"{name} must be divisible by 8")
        return self


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    seed: int
    mime_type: str


class GenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


class HealthResponse(BaseModel):
    status: str
    service: str
    model_id: str
    device: str
    ready: bool
    version: str
