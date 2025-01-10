# Use the NVIDIA CUDA base image with Python 3.8
#FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-pip \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the rest of your application code into the container
COPY . .

# Set the default command to execute when running the container
#CMD ["python3", "detector_app.py"]
CMD ["gunicorn","--workers","2","--timeout","600","--bind","0.0.0.0:7001","wsgi_detector_app:app"]
#gunicorn --workers 2 --timeout 600 --bind 0.0.0.0:7001 wsgi_detector_app:app
