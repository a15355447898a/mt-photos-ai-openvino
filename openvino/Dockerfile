# Use the official OpenVINO runtime image as a parent image
FROM openvino/ubuntu22_runtime:latest
USER root

# Set up APT sources for faster downloads
COPY ./sources.list /etc/apt/sources.list

# Install permanent system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libgomp1 \
    ccache \
    supervisor && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Pre-create insightface model directory so local weights can be used directly
RUN mkdir -p /app/insightface_models/models
# Pre-create log directory for supervisor
RUN mkdir -p /var/log/supervisor

# Install build dependencies, Python packages, and then remove build dependencies in a single layer
# This minimizes image size and avoids "no space left on device" errors during build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip3 install --no-cache-dir -r requirements.txt --index-url=https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code, including server.py, utils, models, and fonts
COPY . .

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Grant execution permissions
RUN chmod +x /app/entrypoint.sh && chmod +x /app/watchdog.sh

# Set environment variables
ENV NO_ALBUMENTATIONS_UPDATE=1
ENV API_AUTH_KEY=mt_photos_ai_extra
ENV INSIGHTFACE_MODEL_ROOT=/app/insightface_models

# Expose the port the app runs on
EXPOSE 8060

# Set the entrypoint to our custom script.
# This script will pre-load models and then start the main application.
ENTRYPOINT ["/app/entrypoint.sh"]
