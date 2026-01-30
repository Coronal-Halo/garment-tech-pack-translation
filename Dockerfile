# Tech Pack Image Translation System
# Crystal International - Technical Assessment
# Docker Configuration

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    # Font support
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    # Build tools
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Update font cache
RUN fc-cache -fv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create directories and download models
RUN mkdir -p /root/.cache/huggingface && \
    python3 -c "from transformers import MarianMTModel, MarianTokenizer; \
                model_name = 'Helsinki-NLP/opus-mt-en-zh'; \
                MarianTokenizer.from_pretrained(model_name); \
                MarianMTModel.from_pretrained(model_name)"

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY run.py .

# Create directories and set permissions
RUN mkdir -p inputs outputs assets/fonts && \
    chmod -R 777 outputs

# Copy font files if available (use . to copy directory contents)
COPY assets/fonts/. ./assets/fonts/

# Set default command
ENTRYPOINT ["python", "run.py"]
CMD ["--help"]


# Labels
LABEL maintainer="Yuxiang Huang"
LABEL description="Tech Pack Image Translation System for Crystal International"
LABEL version="1.0.0"
