# ---- Base image ----
FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System deps (audio + websockets stability) ----
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Install Python deps first (better caching) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy app ----
COPY . .

# ---- Expose port ----
EXPOSE 8003

# ---- Run server ----
CMD ["uvicorn", "appv10:app", "--host", "0.0.0.0", "--port", "8003"]
