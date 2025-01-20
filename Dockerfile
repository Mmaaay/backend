FROM python:3.12-slim-bullseye AS model

WORKDIR /model
# Remove downloading model
# RUN pip install --no-cache-dir transformers torch
# RUN python -c "from transformers import AutoModel, AutoTokenizer; model_name='sentence-transformers/paraphrase-albert-small-v2'; model = AutoModel.from_pretrained(model_name); tokenizer = AutoTokenizer.from_pretrained(model_name); model.save_pretrained('/model'); tokenizer.save_pretrained('/model')"

# Copy pre-downloaded model files
COPY models/embeddings /model

# Stage 0: Generate requirements
FROM python:3.12-slim AS requirements

# Set up a virtual environment

ENV VENV_PATH=/opt/venv

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

# Remove pip-compile step
# RUN pip-compile requirements.txt --output-file requirements.lock

# Simply copy the pre-generated file
COPY requirements.lock .

# Stage 1: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Install build requirements for Python 3.12
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Build and install Python 3.12 from source
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && \
    tar -xf Python-3.12.0.tgz && \
    cd Python-3.12.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get install -y tzdata

# Create appuser
RUN useradd -m appuser

# Copy model files
COPY --from=model /model /app/models/embeddings

# Copy the virtual environment from the requirements stage
COPY --from=requirements /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy locked requirements
COPY --from=requirements /app/requirements.lock /tmp/requirements.lock

# Install Python dependencies using the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn

# Copy application files
COPY src ./src

COPY tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Switch to non-root user
USER appuser

# Remove the apt-get install line for PACKAGE_NAME




# Set correct permissions
RUN chown -R appuser:appuser /app/models

# Expose port and set entry point
EXPOSE 3000

# Use the venv Python explicitly
CMD ["/opt/venv/bin/python", "src/server.py"]