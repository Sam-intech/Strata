FROM python:3.11-slim

WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . /app

# Expose backend
EXPOSE 8000

# Start server
# If you use FastAPI:
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
#
# If you currently run main.py:
# CMD ["python", "main.py"]
