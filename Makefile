# Tech Pack Image Translation System
# Crystal International - Technical Assessment
# Makefile for common operations

.PHONY: all install run run-docker build-docker clean test help

# Default target
all: help

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

# Run translation on the sample image
run:
	@echo "Running tech pack translation..."
	python run.py --input "techpack_img 1.png" --verbose

# Run with custom input
run-custom:
	@echo "Usage: make run-custom INPUT=<image_path>"
	@if [ -n "$(INPUT)" ]; then \
		python run.py --input "$(INPUT)" --verbose; \
	else \
		echo "Error: Please specify INPUT=<image_path>"; \
	fi

# Build Docker image
build-docker:
	@echo "Building Docker image..."
	docker build -t techpack-translator:latest .
	@echo "Docker image built successfully!"

# Run with Docker
run-docker:
	@echo "Running with Docker..."
	docker run --rm \
		-v $(PWD):/app/inputs \
		-v $(PWD)/outputs:/app/outputs \
		techpack-translator:latest \
		--input "inputs/techpack_img 1.png" --verbose

# Run with Docker Compose
run-compose:
	@echo "Running with Docker Compose..."
	docker compose run --rm techpack-translator \
		--input "inputs/techpack_img 1.png" --verbose

# Development shell with Docker
dev-shell:
	@echo "Starting development shell..."
	docker compose --profile dev run --rm techpack-dev

# Create necessary directories
setup-dirs:
	@echo "Creating directories..."
	mkdir -p inputs outputs assets/fonts
	@echo "Directories created!"

# Copy sample image to inputs
setup-input:
	@echo "Setting up input..."
	cp "techpack_img 1.png" inputs/ 2>/dev/null || echo "Sample image not found or already in inputs/"

# Download Noto Sans CJK font (for Chinese text rendering)
download-font:
	@echo "Downloading Noto Sans SC font..."
	mkdir -p assets/fonts
	curl -L "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf" \
		-o assets/fonts/NotoSansSC-Regular.otf 2>/dev/null || \
		echo "Font download failed. Please download manually from https://fonts.google.com/noto"
	@echo "Font downloaded!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf outputs/*
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf *.pyc src/*.pyc tests/*.pyc
	rm -rf .pytest_cache
	@echo "Clean complete!"

# Clean Docker resources
clean-docker:
	@echo "Cleaning Docker resources..."
	docker-compose down --rmi local 2>/dev/null || true
	docker rmi techpack-translator:latest 2>/dev/null || true
	@echo "Docker cleanup complete!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Format code
format:
	@echo "Formatting code..."
	black src/ run.py
	@echo "Formatting complete!"

# Lint code
lint:
	@echo "Linting code..."
	flake8 src/ run.py --max-line-length=100

# Full setup (install + directories + font)
setup: install setup-dirs download-font
	@echo "Full setup complete!"

# Help
help:
	@echo "Tech Pack Image Translation System"
	@echo "==================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install Python dependencies"
	@echo "  run           - Run translation on sample image"
	@echo "  run-custom    - Run with custom input (INPUT=<path>)"
	@echo "  build-docker  - Build Docker image"
	@echo "  run-docker    - Run translation with Docker"
	@echo "  run-compose   - Run with Docker Compose"
	@echo "  dev-shell     - Start development shell in Docker"
	@echo "  setup         - Full setup (install + dirs + font)"
	@echo "  setup-dirs    - Create necessary directories"
	@echo "  download-font - Download Chinese font"
	@echo "  clean         - Clean generated files"
	@echo "  clean-docker  - Clean Docker resources"
	@echo "  test          - Run unit tests"
	@echo "  format        - Format code with black"
	@echo "  lint          - Lint code with flake8"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make install"
	@echo "  2. make run"
	@echo ""
	@echo "Docker quick start:"
	@echo "  1. make build-docker"
	@echo "  2. make run-docker"
