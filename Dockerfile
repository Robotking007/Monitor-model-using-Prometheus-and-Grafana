# Dockerfile for MLOps Model Serving API
# Student: Siddhant Udavant (Roll No: 20, PRN: 12414447, Branch: AI&DS, Div: F)

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained model artifacts
COPY app.py .
COPY model/ model/

# Expose FastAPI application port
EXPOSE 8000

# Healthcheck definition
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with single worker (standard for Prometheus client metrics isolation)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

