# Intelligent Routing Layer

A small FastAPI service that classifies a request and returns a validated destination from your configured model/SDK registry. **It does not execute inference or call the destination SDK.**

## Requirements

- Python 3.10+
- A Gemini API key for LLM-based classification (optional for the deterministic fallback)
- Your own existing inference services / 100x SDK integration

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set the environment variables from `.env.example` in your shell or your deployment environment. The app does not automatically load `.env`; use your shell, process manager, or add `python-dotenv` if you want automatic loading.

The Gemini OpenAI-compatible endpoint has the form:

`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`

Check the exact model IDs and API compatibility enabled for your Google project before use. The router model can be changed with `ROUTER_LLM_MODEL`.

## Configure destinations

Edit `config/models.yaml`. Only entries in `routes` are eligible destinations. Replace `REPLACE_WITH_VERIFIED_VIDEO_MODEL_ID` with the exact video model ID you have access to. The `100x` entries are SDK capability destinations; your application must interpret the returned `capability` and invoke the relevant SDK function.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8085 --reload
```

Health check:

```bash
curl http://127.0.0.1:8085/health
```

Route a request:

```bash
curl -X POST http://127.0.0.1:8085/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a PowerPoint presentation about AI adoption",
    "requirements": {
      "quality": "high",
      "latency": "normal",
      "cost": "medium"
    }
  }'
```

Open interactive API docs at `http://127.0.0.1:8085/docs`.

## How it works

1. If `ROUTER_LLM_API_KEY`, `ROUTER_LLM_BASE_URL`, and `ROUTER_LLM_MODEL` are configured, the analyzer asks the routing LLM for JSON classification.
2. The returned task must be one of the route keys in `config/models.yaml`. Invalid tasks are rejected and sent to the deterministic fallback.
3. The API returns the selected destination and analysis metadata.
4. The calling gateway executes the selected model/SDK. This service does not execute it.

## Important production notes

- The example uses a small deterministic fallback. It is not a replacement for LLM classification and will not understand every phrasing.
- Verify that your Gemini model supports the configured OpenAI-compatible chat-completions endpoint and JSON response mode. If not, adapt `_classify_with_llm` to your enabled Gemini API.
- Add an explicit modality/capability check based on trusted request metadata or attachment metadata. Do not infer that an image/video/audio is present from prompt text alone.
- Add authentication, request-size limits, structured logs, metrics, timeouts, and rate limits before exposing the service publicly.
- Add provider health and latency/cost ranking only when you have reliable measurements. This MVP routes by task/category and explicit user requirements; it does not dynamically benchmark models.
- Never send API keys in request payloads. Store secrets in environment variables or a secret manager.
- `confidence` is a classifier estimate, not a calibrated probability until evaluated on a labeled dataset.

## Tests

```bash
pytest -q
```
