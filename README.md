# Solar Design Service

A FastAPI service intended for deployment on Cloud Run that orchestrates Google Solar API's building insights with mixed panel layout planning.

## Endpoints

- `POST /square_image` – Returns a square Google Maps Static API satellite image as a Base64 data URI together with the resolved map center and meters-per-pixel figure.
- `POST /layout_panels` – Packs rectangular panel specifications into provided roof polygons and returns a rendered layout image plus aggregate counts.
- `POST /solar/design` – Accepts a design request and returns the maximum DC kW configuration, including panel mix details and a rendered PNG image encoded in Base64.
- `GET /healthz` – Simple health probe endpoint.

## Request payload expectations

- `/square_image` always responds with `image_data_uri`, `square_size_px`, and a `meters_per_pixel` (m/px) figure.
- `/layout_panels` expects each `roofs[].polygon` entry to be supplied in metres (local X/Y space) while `panels[].w_mm`, `h_mm`, and `gap_mm` are in millimetres. `orientation_mode` accepts `auto`, `portrait`, or `landscape`, and `min_walkway_m` defaults to `0.4` when omitted. `max_total` and `max_per_face` are optional.

## Environment

Set the `GOOGLE_MAPS_API_KEY` environment variable via Cloud Run or Secret Manager. For backwards compatibility `GOOGLE_API_KEY` is also accepted. The service never stores the key in source code.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_MAPS_API_KEY=your-key
uvicorn app.main:app --reload --port 8080
```

## Deployment

Build the container and deploy to Cloud Run:

```bash
gcloud builds submit --tag gcr.io/PROJECT/solar-design
gcloud run deploy solar-design --image gcr.io/PROJECT/solar-design --allow-unauthenticated
```
