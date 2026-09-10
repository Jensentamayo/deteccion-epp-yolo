# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# Dependencias de sistema necesarias para opencv y build de paquetes nativos.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (gestor de paquetes/entornos exclusivo del proyecto).
COPY --from=ghcr.io/astral-sh/uv:0.4.30 /uv /usr/local/bin/uv

WORKDIR /app

# Copia solo los manifiestos primero para aprovechar la cache de capas.
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copia el resto del código fuente.
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/

# Genera los stubs gRPC dentro de la imagen.
RUN uv run python -m grpc_tools.protoc \
    -Isrc/grpc_service \
    --python_out=src/grpc_service \
    --grpc_python_out=src/grpc_service \
    src/grpc_service/inference.proto \
    && sed -i 's/^import inference_pb2 as inference__pb2/from src.grpc_service import inference_pb2 as inference__pb2/' \
       src/grpc_service/inference_pb2_grpc.py

ENV PYTHONUNBUFFERED=1 \
    GRPC_PORT=50051

EXPOSE 50051 8501

# Por defecto levanta el servidor gRPC; el servicio Streamlit se orquesta
# como contenedor/proceso separado (ver docker-compose sugerido en README).
CMD ["uv", "run", "python", "-m", "src.grpc_service.server"]
