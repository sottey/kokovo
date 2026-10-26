import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from .auth import require_api_key
from .config import Settings, get_settings
from .generator import FORMAT_DETAILS, GeneratorError, ImageGenerator, as_base64, store_image
from .schemas import GenerationRequest, GenerationResponse, HealthResponse, ImageData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.getLogger().setLevel(settings.log_level.upper())
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    generator = ImageGenerator(settings)
    generator.load()
    app.state.generator = generator
    yield


app = FastAPI(title="Kokovo", version="0.1.0", lifespan=lifespan)


def settings_dependency() -> Settings:
    return get_settings()


@app.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(settings_dependency)) -> HealthResponse:
    generator = getattr(app.state, "generator", None)
    return HealthResponse(status="ok" if generator and generator.ready else "starting", service=settings.service_name, model_id=settings.model_id, device=settings.device, ready=bool(generator and generator.ready), version=settings.version)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
def models(settings: Settings = Depends(settings_dependency)) -> dict:
    return {"data": [{"id": settings.model_id, "object": "model", "active": True}]}


@app.post("/v1/images/generations", response_model=GenerationResponse, dependencies=[Depends(require_api_key)])
async def generate(request: GenerationRequest, settings: Settings = Depends(settings_dependency)) -> GenerationResponse:
    started = time.monotonic()
    generator: ImageGenerator | None = getattr(app.state, "generator", None)
    if generator is None or not generator.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready")
    logger.info("generation start seed=%s images=%s", request.seed, request.num_images)
    try:
        generated = await generator.generate(request)
    except GeneratorError as error:
        logger.exception("generation error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error)) from error
    data: list[ImageData] = []
    for item in generated:
        if request.response_format == "b64_json":
            encoded, mime_type = await run_in_threadpool(as_base64, item.image, request.output_format)
            data.append(ImageData(b64_json=encoded, seed=item.seed, mime_type=mime_type))
        else:
            url = await run_in_threadpool(store_image, item.image, request.output_format, item.seed, request, settings)
            data.append(ImageData(url=url, seed=item.seed, mime_type=FORMAT_DETAILS[request.output_format][1]))
    logger.info("generation complete duration_ms=%d seed=%s", int((time.monotonic() - started) * 1000), request.seed)
    return GenerationResponse(created=int(time.time()), data=data)


@app.get("/outputs/{output_path:path}")
def output(output_path: str, settings: Settings = Depends(settings_dependency)) -> FileResponse:
    root = settings.output_dir.resolve()
    candidate = (root / output_path).resolve()
    if root not in candidate.parents or candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or not candidate.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Output not found")
    return FileResponse(candidate)
