# Kokovo

Kokovo is a small, self-hosted REST API for text-to-image generation with Hugging Face Diffusers. It is designed for one Nvidia-equipped homelab machine: no workflow UI, database, queues, or background workers.

## What it provides

- FastAPI/OpenAPI service (`/docs`)
- One warm, configurable Diffusers text-to-image pipeline
- Bearer-key protection for generation and model endpoints
- Synchronous, serialized generation for predictable GPU use
- PNG, JPEG, or WebP outputs as base64 JSON or locally served URLs
- Output images and optional JSON metadata in a mounted directory

## Requirements

- Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An Nvidia GPU with a CUDA-capable driver
- Access to the configured Hugging Face model. Set `HF_TOKEN` for gated models.

## Run with Docker

```bash
cp .env.example .env
# Edit .env: set API_KEY and, if necessary, MODEL_ID/HF_TOKEN.
docker compose up --build -d
curl http://localhost:8000/health
```

Images persist under `./outputs`. The first start downloads the model into the `./models` cache and can take a while.

## API examples

```bash
curl http://localhost:8000/health
```

```bash
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_API_KEY' \
  -d '{
    "prompt": "a cozy moonlit bedroom, soft shadows, calm atmosphere, cinematic",
    "negative_prompt": "text, watermark, blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 42,
    "num_images": 1,
    "output_format": "png",
    "response_format": "url"
  }'
```

`stabilityai/sdxl-turbo` is the default and works best with low step counts and `guidance_scale: 0`. Change `MODEL_ID` to use another Diffusers-compatible text-to-image model. Request limits are environment-configurable.

The optional `scheduler` accepts `ddim`, `dpmpp_2m`, `euler`, or `euler_a`. Scheduler compatibility depends on the selected model.

## Reference images

With the default SDXL Turbo model, Kokovo can use an uploaded image as an image-to-image reference. In the browser, select a PNG, JPEG, or WebP file and choose its reference strength. The uploaded source is used only for that request and is not stored; generated-image metadata records only whether a reference was used and its strength.

API clients may provide `reference_image` as either a base64 string or a base64 data URL, plus `reference_strength` between `0` and `1`. For SDXL Turbo, `num_inference_steps * reference_strength` must be at least `1`. Reference images are center-cropped to the selected output ratio. `MAX_REFERENCE_IMAGE_BYTES` and `MAX_REFERENCE_IMAGE_PIXELS` limit accepted uploads.

When `response_format` is `url`, use the returned relative URL at this service, for example `http://localhost:8000/outputs/2026/10/26/<uuid>.png`.

## Local development

Python 3.11+ is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
pytest
```

For CPU-only development, set `DEVICE=cpu` and `TORCH_DTYPE=float32`. CPU generation is intentionally not a practical production configuration.

## Security and limitations

- Set a strong `API_KEY`; it is required by default for generation and model endpoints. `/health` and output URLs remain public by design.
- This is a single-process personal service. Requests are serialized in-process to prevent concurrent GPU allocations.
- The service has no rate limiting, user accounts, job persistence, image-to-image, or content moderation layer. Review your chosen model's license and safety requirements before exposing it beyond a trusted network.
- The configured model loads at application startup; startup fails clearly if it cannot be downloaded or initialized.

