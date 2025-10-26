# Gunakan image Python resmi
FROM python:3.11-slim

# Install dependensi sistem (libGL, libglib, dsb)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /app

# Copy semua file proyek
COPY . .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan aplikasi Flask via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
