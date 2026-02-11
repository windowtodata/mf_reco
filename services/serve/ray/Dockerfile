# Use Ray ML image with GPU support
FROM rayproject/ray:2.51.0-py312-gpu

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include the source code directory
# This allows 'import recommend' to work from anywhere
ENV PYTHONPATH="${PYTHONPATH:-}:/workspace/src/main/python"

# Default environment variables
ENV MODEL_DIR="/workspace/models/latest"

# Configure Ray Serve to listen on all interfaces
ENV RAY_SERVE_HTTP_OPTIONS='{"host": "0.0.0.0", "port": 8000}'

# Expose Ray Serve HTTP port
EXPOSE 8000

# Start Ray Serve
CMD ray start --head --dashboard-host=0.0.0.0 --num-gpus=1 --disable-usage-stats && \
    serve start --http-host 0.0.0.0 --http-port 8000 && \
    serve run serve_recommender:deployment