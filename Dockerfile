FROM python:3.12-slim-bullseye AS model

WORKDIR /model
# Remove downloading model
# RUN pip install --no-cache-dir transformers torch
# RUN python -c "from transformers import AutoModel, AutoTokenizer; model_name='sentence-transformers/paraphrase-albert-small-v2'; model = AutoModel.from_pretrained(model_name); tokenizer = AutoTokenizer.from_pretrained(model_name); model.save_pretrained('/model'); tokenizer.save_pretrained('/model')"

# Copy pre-downloaded model files
COPY models/embeddings /model

# Stage 0: Generate requirements
FROM python:3.12-slim-bullseye AS requirements

# Install uv first
RUN pip install uv

# Set up a virtual environment using uv
ENV VENV_PATH=/opt/venv
RUN uv venv /opt/venv --python 3.12 --seed  --relocatable  
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Set PATH to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables
ARG ENV_RELOAD
ENV ENV_RELOAD="Production"

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

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC


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


RUN pip install uv 

# Copy requirements file
COPY requirements.txt .


# Simply copy the pre-generated file
COPY requirements.lock .

# Stage 1: Runtime image
FROM nvcr.io/nvidia/cuda:12.5.1-runtime-ubuntu22.04 AS runtime

# Set CUDA environment variables
RUN apt-get update && apt-get install -y curl gnupg \
  && mkdir -p /usr/share/keyrings \
  && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update && apt-get install -y nvidia-container-toolkit \
  && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# Install build requirements for Python 3.12
WORKDIR /app/build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libssl-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    xz-utils \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit


# Install Python 3.12 from source with correct checksum
WORKDIR /app/build
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && \
    echo "51412956d24a1ef7c97f1cb5f70e185c13e3de1f50d131c0aac6338080687afb  Python-3.12.0.tgz" | sha256sum -c - && \
    tar -xf Python-3.12.0.tgz && \
    cd Python-3.12.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Create appuser
RUN useradd -m appuser

# Copy model files with ownership set to appuser
COPY --chown=appuser:appuser --from=model /model ./models/embeddings

# Copy the virtual environment from the requirements stage
COPY --from=requirements /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/uv pip install --upgrade pip setuptools wheel uv

# Set PATH to use the virtual environment

# Copy locked requirements
COPY --from=requirements /app/requirements.lock /tmp/requirements.lock
COPY --from=requirements /app/requirements.txt /tmp/requirements.txt
# set noninteractive installation
# Install Python dependencies using the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
RUN /opt/venv/bin/uv pip install --upgrade pip && /opt/venv/bin/uv pip install --verbose -r /tmp/requirements.lock
    
# Copy application files
COPY src ./src

COPY tafasir_quran_faiss_vectorstore /app/build/tafasir_quran_faiss_vectorstore
RUN chmod -R 775 /app/build/tafasir_quran_faiss_vectorstore && \
    chown -R appuser:appuser /app/build/tafasir_quran_faiss_vectorstore

# Switch to non-root user
USER appuser

# Remove the chown command
# RUN chown -R appuser:appuser /app/models

# Expose port and set entry point
EXPOSE 3000

# Use the venv Python explicitly
CMD ["/opt/venv/bin/python", "src/server.py"]

