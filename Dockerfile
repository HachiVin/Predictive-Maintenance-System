# REST API image: FastAPI + the model, no dashboard libraries.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 app

COPY pdm ./pdm
COPY app/__init__.py app/main.py app/schemas.py ./app/
COPY models ./models
COPY docs/example_request.json ./docs/

USER app
EXPOSE 8000
# Healthy only when the model is loaded, not merely when the process answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import json,urllib.request as u; assert json.load(u.urlopen('http://127.0.0.1:8000/health'))['model_loaded']"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
