# Stage 1: Build stage
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

CMD ["python", "app.py"]