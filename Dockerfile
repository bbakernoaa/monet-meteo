FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install development dependencies
RUN pip install --no-cache-dir build pytest pytest-cov black flake8 mypy pre-commit

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Run tests to verify the installation
RUN pytest tests/unit -v

# Set the default command
CMD ["python", "-c", "import monet_meteo; print('monet_meteo version:', monet_meteo.__version__)"]