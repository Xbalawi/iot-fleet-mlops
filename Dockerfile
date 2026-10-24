FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Libraries first: this layer is cached until requirements-serve.txt changes.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --default-timeout=120 -r requirements-serve.txt

COPY src ./src
COPY model_store ./model_store

ENV MODEL_DIR=/app/model_store

# Do not run as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# One worker on purpose: the scorer keeps each sensor's recent readings in memory.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]