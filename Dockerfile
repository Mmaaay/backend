FROM python:3.12-slim-bullseye AS model

WORKDIR /model
# Remove downloading model
# RUN pip install --no-cache-dir transformers torch
# RUN python -c "from transformers import AutoModel, AutoTokenizer; model_name='sentence-transformers/paraphrase-albert-small-v2'; model = AutoModel.from_pretrained(model_name); tokenizer = AutoTokenizer.from_pretrained(model_name); model.save_pretrained('/model'); tokenizer.save_pretrained('/model')"

# Copy pre-downloaded model files
COPY path/to/local/model /model


# Stage 0: Generate requirements
ARG PYTHON_VERSION=3.12-slim-bullseye AS requirements
FROM python:${PYTHON_VERSION}

# Set up a virtual environment
RUN python -m venv /opt/venv

# Set environment variables
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

# Stage 1: Runtime image
FROM nvcr.io/nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# Copy model files
COPY --from=model /model /app/models/embeddings

# Copy the virtual environment from the requirements stage
COPY --from=requirements /opt/venv /opt/venv


# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy locked requirements
COPY --from=requirements /app/requirements.lock /tmp/requirements.lock

# Install Python dependencies using the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.lock && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.1.0 faiss-gpu-cu12==1.9.0.0 uvicorn


# Copy application files
COPY src ./src
COPY .env .env
COPY tafasir_quran_faiss_vectorstore ./tafasir_quran_faiss_vectorstore

# Switch to non-root user
USER appuser

# Set correct permissions
RUN chown -R appuser:appuser /app/models

# Create appuser
RUN useradd -m appuser
# Expose port and set entry point
EXPOSE 3000

# Use the venv Python explicitly
CMD ["/opt/venv/bin/python", "src/server.py"]
