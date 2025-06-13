# ──────────────────────────────────────────────────────────────
# docker/Qdrant.Dockerfile
# ──────────────────────────────────────────────────────────────
FROM qdrant/qdrant:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates bash && \
    rm -rf /var/lib/apt/lists/*

# Expose is just documentary; docker-compose does the publishing.
EXPOSE 6333 6334
