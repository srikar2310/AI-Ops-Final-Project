# Pin the base image tag: the Stockfish version comes from this distro's apt repo,
# so changing the tag can change the engine and therefore the evaluation data.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Stockfish installs to /usr/games/stockfish (matches ENGINE_PATH in the script)
RUN apt-get update \
    && apt-get install -y --no-install-recommends stockfish git \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/games:${PATH}"

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application code (data is NOT baked in; mount it as a volume or pull with DVC)
COPY Stockfish_train_eval.py params.yaml ./

# Run as a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Log the engine version at build time so it is visible in the build output
RUN stockfish <<< "quit" | head -n 1 || true

# Default: run the evaluation. Override args at `docker run` time.
ENTRYPOINT ["python", "Stockfish_train_eval.py"]
CMD ["--input", "dataset_clean/train", "--output", "dataset_clean/train_evaluated"]