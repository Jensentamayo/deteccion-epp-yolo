"""Interfaz Streamlit para detección de EPP en tiempo real.

Responsabilidad (capa Vista/Cliente): capturar la imagen del usuario,
enviarla al servicio gRPC de inferencia y desplegar la clase predicha,
la probabilidad y las cajas delimitadoras sobre la imagen.

Ejecutar con: uv run streamlit run app/detector_epp.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from src.grpc_service.client import ClientDetectResult, detect
from src.visualizations.draw_boxes import draw_detections
from src.models.predict_model import Detection

st.set_page_config(page_title="Detección de EPP - YOLO11n", layout="wide")
st.title("🦺 Detección de Elementos de Protección Personal (EPP)")
st.caption("Casco y guantes · YOLO11n servido vía gRPC · Talos IA - Módulo 3")

with st.sidebar:
    st.header("Configuración")
    conf_threshold = st.slider("Umbral de confianza", 0.05, 0.95, 0.25, 0.05)
    grpc_address = st.text_input("Dirección del servicio gRPC", value="localhost:50051")

uploaded_file = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image = np.array(Image.open(uploaded_file).convert("RGB"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(original_image, use_container_width=True)

    if st.button("Ejecutar detección", type="primary"):
        with st.spinner("Consultando servicio de inferencia gRPC..."):
            try:
                result: ClientDetectResult = detect(
                    image_bytes=image_bytes,
                    conf_threshold=conf_threshold,
                    address=grpc_address,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo contactar al servicio gRPC: {exc}")
                st.stop()

        detections = [
            Detection(
                class_name=d.class_name,
                confidence=d.confidence,
                x1=d.box[0],
                y1=d.box[1],
                x2=d.box[2],
                y2=d.box[3],
            )
            for d in result.detections
        ]
        annotated = draw_detections(original_image, detections)

        with col2:
            st.subheader("Detecciones")
            st.image(annotated, use_container_width=True)

        st.success(
            f"{len(result.detections)} objeto(s) detectado(s) en "
            f"{result.inference_time_ms:.1f} ms · modelo {result.model_version}"
        )

        if result.detections:
            st.table(
                [
                    {"Clase": d.class_name, "Confianza": f"{d.confidence:.2%}"}
                    for d in result.detections
                ]
            )
        else:
            st.info("No se detectaron elementos por encima del umbral configurado.")
else:
    st.info("Sube una imagen para comenzar.")
