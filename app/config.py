from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = "image-gen-api"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_id: str = "stabilityai/sdxl-turbo"
    device: str = "cuda"
    torch_dtype: str = "float16"
    hf_token: str | None = None
    model_cache_dir: Path = Path("/app/models")
    output_dir: Path = Path("/app/outputs")
    enable_cpu_offload: bool = False
    attention_slicing: bool = True
    xformers_enabled: bool = False
    safety_checker_disabled: bool = False
    api_key: str | None = None
    require_api_key: bool = True
    default_width: int = Field(default=1024, ge=64)
    default_height: int = Field(default=1024, ge=64)
    max_width: int = Field(default=1536, ge=64)
    max_height: int = Field(default=1536, ge=64)
    max_steps: int = Field(default=50, ge=1)
    max_images: int = Field(default=4, ge=1)
    save_metadata: bool = True

    @field_validator("torch_dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        if value not in {"float16", "float32", "bfloat16"}:
            raise ValueError("TORCH_DTYPE must be float16, float32, or bfloat16")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
