# Runner/agent image: python + project deps + the docker CLI.
# It talks to the HOST docker daemon through the mounted socket (docker-out-of-docker),
# so it needs the client binary only — no dockerd inside.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates docker-cli \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so source edits don't bust the layer cache.
COPY pyproject.toml ./
RUN mkdir -p src && touch src/__init__.py
RUN uv pip install --system --no-cache .

COPY src/ ./src/

CMD ["python", "-m", "src.main"]
