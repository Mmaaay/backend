# Stage 0: Download or prepare the model
FROM python:3.12-slim-bullseye AS model

WORKDIR /opt/model
# Copy pre-downloaded model files
COPY models/embeddings /opt/model

# Stage 1: Generate requirements
FROM python:3.12-slim-bullseye AS requirements

WORKDIR /opt/app

# Install uv first
RUN pip install uv

# Set up a virtual environment using uv
ENV VENV_PATH=/opt/venv
RUN uv venv /opt/venv --python 3.12 --seed --relocatable
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip

# Copy requirements files
COPY requirements.txt .
COPY requirements.lock .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.lock

# Stage 2: Runtime image
FROM nvcr.io/nvidia/cuda:12.5.1-runtime-ubuntu22.04 AS runtime

WORKDIR /opt/app

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

# Install Python 3.12 from source
RUN wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz && \
    echo "51412956d24a1ef7c97f1cb5f70e185c13e3de1f50d131c0aac6338080687afb  Python-3.12.0.tgz" | sha256sum -c - && \
    tar -xf Python-3.12.0.tgz && \
    cd Python-3.12.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Create appuser
RUN useradd -m appuser

# Copy virtual environment from the requirements stage
COPY --from=requirements /opt/venv /opt/venv

# Copy the model from the model stage
COPY --from=model /opt/model /opt/app/models/embeddings

# Copy the application code
COPY src /opt/app/src

# Copy FAISS vectorstore
COPY tafasir_quran_faiss_vectorstore /opt/app/tafasir_quran_faiss_vectorstore
RUN chmod -R 775 /opt/app/tafasir_quran_faiss_vectorstore && \
    chown -R appuser:appuser /opt/app/tafasir_quran_faiss_vectorstore

# Set up Python environment
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# Switch to non-root user
USER appuser

# Expose port and set entry point
EXPOSE 3000
CMD ["/opt/venv/bin/python", "/opt/app/src/server.py"]
