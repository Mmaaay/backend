# Stage 0: Generate requirements
FROM python:3.12-slim-bookworm AS requirements

# Install pip-tools
RUN pip install pip-tools

# Copy requirements file
COPY requirements.txt .

# Generate locked requirements
RUN pip-compile requirements.txt --output-file requirements.lock

# Stage 1: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements and create virtual environment
COPY --from=requirements requirements.lock .
RUN python3.11 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn

# Copy application files
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser .env .env
COPY --chown=appuser:appuser tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Set PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"

# Switch to non-root user
USER appuser

# Expose port and set entry point
EXPOSE 3000

# Use the venv Python explicitly
CMD ["/opt/venv/bin/python", "src/server.py"]