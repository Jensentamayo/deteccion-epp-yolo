#!/usr/bin/env bash
# Genera inference_pb2.py e inference_pb2_grpc.py a partir del contrato .proto.
# Uso: uv run bash scripts/generate_grpc.sh
set -euo pipefail

PROTO_DIR="src/grpc_service"
OUT_DIR="src/grpc_service"

python -m grpc_tools.protoc \
  -I"${PROTO_DIR}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${PROTO_DIR}/inference.proto"

# Corrige el import relativo que genera protoc por defecto.
sed -i 's/^import inference_pb2 as inference__pb2/from src.grpc_service import inference_pb2 as inference__pb2/' \
  "${OUT_DIR}/inference_pb2_grpc.py"

echo "Stubs generados en ${OUT_DIR}/inference_pb2.py y ${OUT_DIR}/inference_pb2_grpc.py"
