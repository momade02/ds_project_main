# ==============================================================================
# DOCKERFILE — Streamlit UI Container (Tankstoppfinder)
# ==============================================================================
#
# Purpose
# - Builds a lightweight runtime image for the multi-page Streamlit application.
# - Designed to run in Azure Container Apps with port 8501 exposed and Streamlit bound to 0.0.0.0.
#
# Image characteristics
# - Base: python:3.11-slim (small footprint; sufficient for Streamlit + scientific Python stack).
# - Installs minimal system tooling only (build-essential) to support compilation of Python wheels
#   if a dependency does not provide prebuilt binaries for the target platform.
#
# Build flow
# 1) Set working directory to /app
# 2) Install minimal OS packages (build-essential) then clean apt lists to reduce image size
# 3) Copy requirements.txt and install Python dependencies 
# 4) Copy application source code into the image
#
# Runtime 
# - Exposes port 8501
# - Runs Streamlit with:
#     streamlit run src/app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
# - In Azure Container Apps, environment variables/secrets provide API keys and Redis configuration.
# ==============================================================================

FROM python:3.11-slim

WORKDIR /app

# System packages (keep minimal; add more only if build fails)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
