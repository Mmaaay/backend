# Stage 1: Build dependencies
FROM python:3.12-slim-bullseye AS build

# Set up environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies and set up virtual environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

# Copy model and application files
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY models/embeddings /app/models/embeddings
COPY src ./src
COPY tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Set permissions and switch to non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port and set entrypoint
EXPOSE 3000
CMD ["python", "src/server.py"]
