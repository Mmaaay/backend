# Stage 0: Generate requirements
ARG PYTHON_VERSION=3.12-slim-bullseye AS requirements
FROM python:${PYTHON_VERSION}

#set up a virtual environment
RUN python -m venv /opt/venv

# Set environment variables
ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY = $GOOGLE_API_KEY 

ARG MONGODB_URI
ENV MONGODB_URI = $MONGODB_URI

ARG MONGODB_PASSWORD
ENV MONGODB_PASSWORD = $MONGODB_PASSWORD

ARG SUPABASE_API_KEY
ENV SUPABASE_API_KEY = $SUPABASE_API_KEY

ARG SUPABASE_PROJECT_URL
ENV SUPABASE_PROJECT_URL = $SUPABASE_PROJECT_URL

# Create a non-root user
RUN useradd -m appuser && \
    chown -R appuser /opt/venv

# Set PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive
    

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


# Set working directory
RUN mkdir -p /app
WORKDIR /app

# Install pip-tools
RUN pip install pip-tools

# Copy requirements file
COPY requirements.txt .

# Generate locked requirements
RUN pip-compile requirements.txt --output-file requirements.lock

# Stage 1: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime


# Copy requirements and create virtual environment
COPY --from=requirements requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.lock && \
    pip install --no-cache-dir torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn

# Copy application files
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser .env .env
COPY --chown=appuser:appuser tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Set PATH to use venv


# Switch to non-root user
USER appuser

# Expose port and set entry point
EXPOSE 3000

# Use the venv Python explicitly
CMD ["/opt/venv/bin/python", "src/server.py"]