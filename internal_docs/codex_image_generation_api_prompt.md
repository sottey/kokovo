Use this as your **Codex prompt**.

---

# Prompt for Codex

Build a **self-hosted image generation API service** for my homelab that is conceptually similar to Kokoro-FastAPI for TTS:

- simple
- lightweight
- API-first
- Dockerized
- Nvidia GPU capable
- no heavy workflow UI
- no ComfyUI
- no Automatic1111
- suitable for being called from apps and scripts

The service should be a **single-purpose REST API** around **Hugging Face Diffusers** for text-to-image generation, with room for future image-to-image support.

## Primary goals

Design and implement a production-usable MVP with these properties:

- **FastAPI** backend
- **Diffusers** for inference
- **PyTorch with CUDA**
- **Dockerized**
- **docker-compose friendly**
- **OpenAPI/Swagger docs**
- **simple JSON API**
- **predictable file/output handling**
- **API key support**
- **healthcheck endpoint**
- **model configurable**
- **negative prompt support**
- **basic generation parameter support**
- **multiple output formats if reasonable**
- **clear logging**
- **works as a headless service**
- **easy to run on a single Nvidia GPU homelab machine**

## Important constraints

- Do **not** build or embed a graphical workflow engine
- Do **not** build a complex frontend
- A tiny optional HTML test page is acceptable, but API-first is the priority
- Keep dependencies as small and explicit as practical
- Prefer clarity and maintainability over clever abstractions
- The code should be organized for a solo developer to understand quickly
- Avoid overengineering
- Avoid introducing Celery, Redis, Kafka, or any large distributed components unless absolutely necessary
- The first version should run in a **single container**
- The service should be suitable for self-hosted personal use, not multi-tenant enterprise SaaS

## Technical stack

Use:

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- diffusers
- torch
- transformers if needed
- accelerate if needed
- Pillow for image handling

Prefer standard libraries where possible.

## Model requirements

The service should support loading a text-to-image model from configuration, initially targeting a practical Stable Diffusion XL or similar Diffusers-compatible model.

Design for configurable model selection via environment variables, for example:

- `MODEL_ID`
- `TORCH_DTYPE`
- `DEVICE`
- `ENABLE_CPU_OFFLOAD`
- `ATTENTION_SLICING`
- `XFORMERS_ENABLED` if applicable
- `SAFETY_CHECKER_DISABLED`
- `HF_TOKEN` if needed for gated models

Make model loading happen once at startup and keep the pipeline warm.

## API requirements

Implement these endpoints.

### 1. `GET /health`
Return a simple JSON payload indicating service health, model loaded state, device, and version.

Example response:
```json
{
  "status": "ok",
  "service": "image-gen-api",
  "model_id": "stabilityai/sdxl-turbo",
  "device": "cuda",
  "ready": true,
  "version": "0.1.0"
}
```

### 2. `GET /v1/models`
Return a simple list of the active model or available configured models.

### 3. `POST /v1/images/generations`
Primary text-to-image endpoint.

Accept JSON similar in spirit to OpenAI image generation, but optimized for self-hosted Diffusers.

Example request:
```json
{
  "prompt": "a peaceful desert landscape at dusk, cinematic, soft moonlight, highly detailed",
  "negative_prompt": "text, watermark, blurry, distorted, ugly",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 20,
  "guidance_scale": 6.5,
  "seed": 12345,
  "num_images": 1,
  "output_format": "png",
  "response_format": "b64_json"
}
```

Support these fields:
- `prompt` required
- `negative_prompt` optional
- `width`
- `height`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `num_images`
- `output_format`
- `response_format`
- optional `scheduler`
- optional `clip_skip` only if practical
- optional `high_noise_frac` only if model requires it
- optional `prompt_2` and `negative_prompt_2` if using SDXL and practical

Validate inputs carefully.

### Response formats

Support at least:
- `b64_json`
- `url` referencing a locally served saved file

If `response_format` is `b64_json`, return base64-encoded images in JSON.

If `response_format` is `url`, save output files to a local outputs directory and return URLs.

Example response:
```json
{
  "created": 1730000000,
  "data": [
    {
      "b64_json": "....",
      "seed": 12345,
      "mime_type": "image/png"
    }
  ]
}
```

or

```json
{
  "created": 1730000000,
  "data": [
    {
      "url": "/outputs/2026/10/26/abcd1234.png",
      "seed": 12345,
      "mime_type": "image/png"
    }
  ]
}
```

### 4. `GET /outputs/{path}`
Serve generated files safely from the outputs directory.
Prevent directory traversal.

### 5. Optional: `POST /v1/images/variations`
Only add this if it can be done cleanly. MVP priority is text-to-image.

## Non-functional behavior

### Startup
- Load configuration from environment variables
- Initialize model pipeline once
- Log startup clearly
- Fail fast with useful errors if the model cannot be loaded

### Performance
- Support GPU execution
- Use inference mode / no grad
- Keep memory usage reasonable
- Consider enabling memory optimizations via config
- Make generation requests synchronous for MVP
- If concurrency is a problem, serialize generation using a simple lock or queue within the process

### File handling
- Save files in a predictable structure like:
  - `/app/outputs/YYYY/MM/DD/<uuid>.png`
- Include metadata if practical:
  - prompt
  - seed
  - model
  - timestamp
- If sidecar JSON is useful, store it next to the image
- Make output directory configurable via env var

### Security
Implement optional API key auth via environment variable:
- `API_KEY`

If configured:
- require `Authorization: Bearer <key>` for generation endpoints
- health endpoint may remain open
- outputs may be open or protected; choose a simple sane default and document it

### Errors
Return structured JSON errors with useful messages.
Examples:
- invalid dimensions
- unsupported output format
- model not ready
- CUDA unavailable
- out of memory
- generation failure

### Logging
Implement structured-ish logging that is readable in Docker logs.
Log:
- startup config summary
- request start
- request end
- duration
- seed
- errors

Do not log full API keys.

## Code organization

Use a clean project structure, for example:

- `app/main.py`
- `app/config.py`
- `app/models.py`
- `app/schemas.py`
- `app/auth.py`
- `app/generator.py`
- `app/utils.py`

This does not have to be exact, but keep concerns separated.

## Config behavior

Support environment variables like:

- `HOST=0.0.0.0`
- `PORT=8000`
- `MODEL_ID=stabilityai/sdxl-turbo`
- `DEVICE=cuda`
- `TORCH_DTYPE=float16`
- `OUTPUT_DIR=/app/outputs`
- `API_KEY=...`
- `DEFAULT_WIDTH=1024`
- `DEFAULT_HEIGHT=1024`
- `MAX_WIDTH=1536`
- `MAX_HEIGHT=1536`
- `MAX_STEPS=50`
- `MAX_IMAGES=4`
- `LOG_LEVEL=INFO`
- `SAVE_METADATA=true`
- `REQUIRE_API_KEY=true`

Add sane defaults.

## Deliverables

Provide all of the following:

### 1. Full source code
Complete runnable implementation.

### 2. `requirements.txt`
Pinned or reasonably constrained dependencies.

### 3. `Dockerfile`
Use a CUDA-capable base image appropriate for PyTorch inference.

### 4. `docker-compose.yml`
A ready-to-run compose file for Nvidia GPU usage.

### 5. `.env.example`
Include all supported configuration values with comments if helpful.

### 6. `README.md`
Document:
- what the service does
- how to run it
- required Nvidia setup
- how to configure the model
- curl examples
- response examples
- known limitations

## Docker requirements

The Docker setup should:

- run the FastAPI app
- expose port 8000
- mount outputs as a volume
- support Nvidia GPU access
- be easy to launch with `docker compose up -d`

Include a compose example using modern GPU reservation syntax if practical.

## Testing requirements

At minimum, include:
- one or two unit tests for request validation or auth behavior
- a simple smoke test approach in the README

## API examples to include

Include curl examples like:

### Health
```bash
curl http://localhost:8000/health
```

### Generate image
```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "a cozy moonlit bedroom, soft shadows, calm atmosphere, cinematic",
    "negative_prompt": "text, watermark, blurry, low quality",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 6.5,
    "seed": 42,
    "num_images": 1,
    "output_format": "png",
    "response_format": "url"
  }'
```

## Design preference notes

Optimize this for a developer who prefers:
- explicit code
- small moving parts
- Docker deployment
- API-first services
- no bloated GUI stack
- maintainable homelab infrastructure

Keep it simple, robust, and easy to inspect.

## Nice-to-have features if they do not add much complexity

Only include these if straightforward:
- selectable scheduler
- image metadata sidecar JSON
- `/docs` from FastAPI
- request IDs
- automatic seed generation when omitted
- graceful CUDA OOM handling
- model warmup on startup

## Avoid
- ComfyUI integration
- Automatic1111 integration
- databases
- background workers
- websockets
- accounts/users system
- frontend frameworks
- plugin systems
- giant abstraction layers

## Final output expectation

Return:
1. the complete codebase
2. Docker assets
3. README
4. concise explanation of architecture and tradeoffs

---

## My note to you
- I’d also add this **one-liner above the prompt** when you paste it into Codex:

**“Prefer a minimal, explicit implementation that actually runs over a feature-rich but fragile one.”**

- If you want, I can next turn this into:
  - a **shorter Codex prompt**
  - or an **even better spec doc** with acceptance criteria and exact endpoint schemas.
