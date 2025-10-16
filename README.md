# Solar Design Service

A FastAPI service intended for deployment on Cloud Run that orchestrates Google Solar API's building insights with mixed panel layout planning.

## Endpoints

- `POST /solar/design` – Accepts a design request and returns the maximum DC kW configuration, including panel mix details and a rendered PNG image encoded in Base64.
- `GET /healthz` – Simple health probe endpoint.

## Environment

Set the `GOOGLE_API_KEY` environment variable via Cloud Run or Secret Manager. The service never stores the key in source code.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your-key
uvicorn app.main:app --reload --port 8080
```

## Deployment

Build the container and deploy to Cloud Run:

```bash
gcloud builds submit --tag gcr.io/PROJECT/solar-design
gcloud run deploy solar-design --image gcr.io/PROJECT/solar-design --allow-unauthenticated
```
