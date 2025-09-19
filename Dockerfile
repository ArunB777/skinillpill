# -------------------------
# Base image
# -------------------------
FROM python:3.11-slim

# -------------------------
# Set work directory
# -------------------------
WORKDIR /app

# -------------------------
# Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Copy requirements and install Python dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy project files
# -------------------------
COPY . .

# -------------------------
# Expose the port for FastAPI
# -------------------------
EXPOSE 8000

# -------------------------
# Command to run FastAPI with Uvicorn (Render uses $PORT)
# -------------------------
CMD ["sh", "-c", "uvicorn fast:app --host 0.0.0.0 --port ${PORT:-8000}"]
