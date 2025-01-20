FROM python:3.12-slim-bullseye AS model

WORKDIR /model
# Remove downloading model
# RUN pip install --no-cache-dir transformers torch
# RUN python -c "from transformers import AutoModel, AutoTokenizer; model_name='sentence-transformers/paraphrase-albert-small-v2'; model = AutoModel.from_pretrained(model_name); tokenizer = AutoTokenizer.from_pretrained(model_name); model.save_pretrained('/model'); tokenizer.save_pretrained('/model')"

# Copy pre-downloaded model files
COPY models/embeddings /model

# Stage 1: Build requirements
FROM python:3.12-slim-bullseye AS build

# Set up a virtual environment
RUN python -m venv /opt/venv

# Set environment variables
ARG ENV_RELOAD
ENV ENV_RELOAD=Production

ARG GEMENI_API_KEY
ENV GEMENI_API_KEY=$GEMENI_API_KEY

ARG GOOGLE_API_KEY
ENV GOOGLE_API_KEY=$GOOGLE_API_KEY 

ARG MONGODB_URI
ENV MONGODB_URI=$MONGODB_URI

ARG MONGODB_PASSWORD
ENV MONGODB_PASSWORD=$MONGODB_PASSWORD

ARG SUPABASE_API_KEY
ENV SUPABASE_API_KEY=$SUPABASE_API_KEY

ARG SUPABASE_PROJECT_URL
ENV SUPABASE_PROJECT_URL=$SUPABASE_PROJECT_URL

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
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

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Install Python and create appuser
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser

# Set up virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the new environment
RUN /opt/venv/bin/pip install --upgrade pip

# Copy requirements and install dependencies
COPY --from=build /app/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && \
    pip install --no-cache-dir torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn

# Create app directory and copy files
WORKDIR /app
COPY --from=model /model /app/models/embeddings
COPY src ./src
COPY tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Set permissions and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 3000
CMD ["python", "src/server.py"]