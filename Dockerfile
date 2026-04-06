# ─── Stage 1: Builder ───
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: Runner ───
FROM python:3.11-slim AS runner
WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

RUN adduser --system --no-create-home appuser
USER appuser

EXPOSE 9002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9002"]
