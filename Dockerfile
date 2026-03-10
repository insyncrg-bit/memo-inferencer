# ─── Memo Inferencer Service ───
# Multi-stage build for a lean Python image.
# Cloud Run sets PORT=8080 automatically.

FROM python:3.13-slim AS base

# Don't buffer stdout/stderr (important for Cloud Run logging)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY deck_distiller.py .
COPY memo_inferencer.py .
COPY llm_client.py .
COPY schema.py .

# Cloud Run always sets PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run with uvicorn — no --reload in production
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
