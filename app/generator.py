import asyncio
import base64
import io
import json
import logging
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .schemas import GenerationRequest

logger = logging.getLogger(__name__)

FORMAT_DETAILS = {
    "png": ("PNG", "image/png", "png"),
    "jpeg": ("JPEG", "image/jpeg", "jpg"),
    "webp": ("WEBP", "image/webp", "webp"),
}


class GeneratorError(RuntimeError):
    pass


@dataclass
class GeneratedImage:
    image: Image.Image
    seed: int


class ImageGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: Any | None = None
        self.ready = False
        self._lock = asyncio.Lock()

    def load(self) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as error:
            raise GeneratorError("Diffusers dependencies are not installed") from error
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise GeneratorError("DEVICE=cuda but CUDA is unavailable")
        kwargs: dict[str, Any] = {"torch_dtype": getattr(torch, self.settings.torch_dtype), "cache_dir": str(self.settings.model_cache_dir)}
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        if self.settings.safety_checker_disabled:
            kwargs["safety_checker"] = None
        logger.info("loading model model_id=%s device=%s", self.settings.model_id, self.settings.device)
        pipeline = AutoPipelineForText2Image.from_pretrained(self.settings.model_id, **kwargs)
        if self.settings.enable_cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.settings.device)
        if self.settings.attention_slicing:
            pipeline.enable_attention_slicing()
        if self.settings.xformers_enabled:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception as error:
                raise GeneratorError("XFORMERS_ENABLED=true but xformers could not be enabled") from error
        self.pipeline = pipeline
        self.ready = True
        logger.info("model loaded model_id=%s", self.settings.model_id)

    async def generate(self, request: GenerationRequest) -> list[GeneratedImage]:
        if not self.ready or self.pipeline is None:
            raise GeneratorError("Model is not ready")
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, request)

    def _generate_sync(self, request: GenerationRequest) -> list[GeneratedImage]:
        import torch

        width = request.width or self.settings.default_width
        height = request.height or self.settings.default_height
        if width > self.settings.max_width or height > self.settings.max_height:
            raise GeneratorError(f"Dimensions must not exceed {self.settings.max_width}x{self.settings.max_height}")
        if request.num_inference_steps > self.settings.max_steps:
            raise GeneratorError(f"num_inference_steps must not exceed {self.settings.max_steps}")
        if request.num_images > self.settings.max_images:
            raise GeneratorError(f"num_images must not exceed {self.settings.max_images}")
        base_seed = request.seed if request.seed is not None else secrets.randbelow(2**63 - 1)
        generators = [torch.Generator(device=self.settings.device).manual_seed(base_seed + index) for index in range(request.num_images)]
        if request.scheduler:
            self._set_scheduler(request.scheduler)
        kwargs: dict[str, Any] = {"prompt": request.prompt, "negative_prompt": request.negative_prompt, "width": width, "height": height, "num_inference_steps": request.num_inference_steps, "guidance_scale": request.guidance_scale, "num_images_per_prompt": request.num_images, "generator": generators}
        for field in ("prompt_2", "negative_prompt_2", "clip_skip", "high_noise_frac"):
            if (value := getattr(request, field)) is not None:
                kwargs[field] = value
        try:
            with torch.inference_mode():
                result = self.pipeline(**kwargs)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise GeneratorError("CUDA out of memory; reduce dimensions, steps, or image count") from error
        except Exception as error:
            raise GeneratorError("Image generation failed") from error
        return [GeneratedImage(image=image, seed=base_seed + index) for index, image in enumerate(result.images)]

    def _set_scheduler(self, name: str) -> None:
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler

        scheduler_types = {
            "ddim": DDIMScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
        }
        scheduler_type = scheduler_types.get(name.lower())
        if scheduler_type is None:
            allowed = ", ".join(sorted(scheduler_types))
            raise GeneratorError(f"Unsupported scheduler {name!r}; choose one of: {allowed}")
        self.pipeline.scheduler = scheduler_type.from_config(self.pipeline.scheduler.config)


def image_bytes(image: Image.Image, output_format: str) -> tuple[bytes, str]:
    pillow_format, mime_type, _ = FORMAT_DETAILS[output_format]
    if pillow_format == "JPEG" and image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=pillow_format)
    return buffer.getvalue(), mime_type


def store_image(image: Image.Image, output_format: str, seed: int, request: GenerationRequest, settings: Settings) -> str:
    _, _, extension = FORMAT_DETAILS[output_format]
    now = datetime.now(UTC)
    relative_dir = Path(now.strftime("%Y/%m/%d"))
    target_dir = settings.output_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.{extension}"
    target = target_dir / filename
    payload, _ = image_bytes(image, output_format)
    target.write_bytes(payload)
    if settings.save_metadata:
        metadata = {"created_at": now.isoformat(), "model_id": settings.model_id, "seed": seed, "prompt": request.prompt, "negative_prompt": request.negative_prompt, "width": request.width or settings.default_width, "height": request.height or settings.default_height, "num_inference_steps": request.num_inference_steps, "guidance_scale": request.guidance_scale}
        target.with_suffix(target.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return f"/outputs/{relative_dir.as_posix()}/{filename}"


def as_base64(image: Image.Image, output_format: str) -> tuple[str, str]:
    payload, mime_type = image_bytes(image, output_format)
    return base64.b64encode(payload).decode("ascii"), mime_type


def latest_output_url(output_dir: Path) -> str | None:
    """Return the newest generated image as a URL relative to this service."""
    if not output_dir.is_dir():
        return None
    images = (
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    newest = max(images, key=lambda path: path.stat().st_mtime, default=None)
    if newest is None:
        return None
    return f"/outputs/{newest.relative_to(output_dir).as_posix()}"
