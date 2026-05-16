# Knorus Price API

FastAPI service for book price prediction by the Knorus publisher. The used model and pipeline structure are described [here](https://github.com/pymlex/knorus-price-regressor).

## Layout

- `app/` API source
- `artifacts/` model bundle and preprocessing files
- `scripts/` run and profiling helpers

## Endpoints

### GET `/health`

```bash
curl http://127.0.0.1:8000/health
````

### POST `/predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title":"Deep Learning with Python","publisher":"Manning","year":2021,"pages":520}'
```

```json
{
  "predicted_price": 1440.0
}
```

## Create .env

```bash
cp .env.example .env
```

## Model download

```bash
mkdir -p artifacts
wget -P artifacts https://huggingface.co/pymlex/knorus-price-regressor/resolve/main/final_model.joblib
wget -P artifacts https://huggingface.co/pymlex/knorus-price-regressor/resolve/main/preprocess_bundle.joblib
```

The API loads the Qwen embedding model at runtime.

## Local run with venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

With Docker Compose:

```bash
docker compose up --build
```

## Prediction request

```bash
python -m scripts/sample_request.py
```

## Py-Spy profiling

Run the server under `py-spy`:

```bash
mkdir -p profiles
py-spy record -o profiles/knorus_api.svg --format svg --rate 100 -- \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker profiling needs `SYS_PTRACE` and an unconfined seccomp profile:

```bash
docker run --rm -it \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/profiles:/app/profiles" \
  knorus-price-api bash -lc 'py-spy record -o /app/profiles/knorus_api.svg --format svg --rate 100 -- uvicorn app.main:app --host 0.0.0.0 --port 8000'
```