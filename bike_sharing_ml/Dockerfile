FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir .

EXPOSE 8000 8501

CMD ["uvicorn", "bike_sharing.apps.api:app", "--host", "0.0.0.0", "--port", "8000"]
