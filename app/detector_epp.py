"""Interfaz Streamlit para detección de EPP en tiempo real.

Responsabilidad (capa Vista/Cliente):
capturar video mediante WebRTC, enviar frames al servicio
gRPC y dibujar las detecciones producidas por YOLO11n.

Ejecutar con:

    uv run streamlit run app/detector_epp.py
"""

from __future__ import annotations

import io
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import (
    WebRtcMode,
    VideoProcessorBase,
    webrtc_streamer,
)

from src.grpc_service.client import EPPGrpcClient
from src.models.predict_model import Detection
from src.visualizations.draw_boxes import draw_detections


st.set_page_config(
    page_title="Detección de EPP - YOLO11n",
    page_icon="🦺",
    layout="wide",
)

st.title(
    "🦺 Detección de Elementos de Protección Personal"
)

st.caption(
    "YOLO11n + gRPC + Streamlit + MLflow · "
    "Talos IA - Módulo 3"
)


MAX_FRAME_SIDE = 1280
JPEG_QUALITY = 80
PROCESS_EVERY_N_FRAMES = 2

DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_GLOVES_THRESHOLD = 0.60


class EPPVideoProcessor(VideoProcessorBase):
    """Procesador de video que ejecuta YOLO11n vía gRPC."""

    def __init__(self) -> None:
        """Inicializa el procesador de video."""
        self._lock = threading.Lock()

        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.gloves_threshold = DEFAULT_GLOVES_THRESHOLD
        self.grpc_address = "localhost:50051"

        self.frame_count = 0
        self.last_inference_ms = 0.0
        self.last_detection_count = 0

        self.client = EPPGrpcClient(
            address=self.grpc_address,
            timeout=10.0,
        )

    def update_config(
        self,
        conf_threshold: float,
        gloves_threshold: float,
        grpc_address: str,
    ) -> None:
        """Actualiza la configuración del procesador."""
        with self._lock:
            address_changed = (
                grpc_address != self.grpc_address
            )

            self.conf_threshold = conf_threshold
            self.gloves_threshold = gloves_threshold
            self.grpc_address = grpc_address

        if address_changed:
            self.client.close()

            self.client = EPPGrpcClient(
                address=grpc_address,
                timeout=10.0,
            )

    @staticmethod
    def _resize_frame(
        image: np.ndarray,
    ) -> np.ndarray:
        """Reduce el frame manteniendo su relación de aspecto."""
        height, width = image.shape[:2]

        longest = max(height, width)

        if longest <= MAX_FRAME_SIDE:
            return image

        scale = MAX_FRAME_SIDE / longest

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode_jpeg(
        image: np.ndarray,
    ) -> bytes:
        """Codifica un frame RGB como JPEG."""
        image_pil = Image.fromarray(image)

        buffer = io.BytesIO()

        image_pil.save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
        )

        return buffer.getvalue()

    def _filter_detections(
        self,
        detections,
    ):
        """Aplica un umbral específico para Gloves."""
        with self._lock:
            gloves_threshold = self.gloves_threshold

        filtered = []

        for detection in detections:
            if (
                detection.class_name.lower() == "gloves"
                and detection.confidence < gloves_threshold
            ):
                continue

            filtered.append(detection)

        return filtered

    def recv(
        self,
        frame: av.VideoFrame,
    ) -> av.VideoFrame:
        """Procesa un frame de video y devuelve el frame anotado."""
        image_bgr = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        if (
            self.frame_count
            % PROCESS_EVERY_N_FRAMES
            != 0
        ):
            return frame

        image_bgr = self._resize_frame(
            image_bgr
        )

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image_bytes = self._encode_jpeg(
            image_rgb
        )

        try:
            with self._lock:
                conf_threshold = self.conf_threshold

            result = self.client.detect(
                image_bytes=image_bytes,
                conf_threshold=conf_threshold,
            )

            detections = self._filter_detections(
                result.detections
            )

            self.last_inference_ms = (
                result.inference_time_ms
            )

            self.last_detection_count = (
                len(detections)
            )

            model_detections = [
                Detection(
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.box[0],
                    y1=detection.box[1],
                    x2=detection.box[2],
                    y2=detection.box[3],
                )
                for detection in detections
            ]

            annotated_rgb = draw_detections(
                image_rgb,
                model_detections,
            )

            annotated_bgr = cv2.cvtColor(
                annotated_rgb,
                cv2.COLOR_RGB2BGR,
            )

            return av.VideoFrame.from_ndarray(
                annotated_bgr,
                format="bgr24",
            )

        except Exception:
            return frame

    def __del__(self) -> None:
        """Cierra el cliente gRPC."""
        try:
            self.client.close()
        except Exception:
            pass


with st.sidebar:
    st.header("⚙️ Configuración")

    conf_threshold = st.slider(
        "Umbral general",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.05,
        help=(
            "Umbral mínimo para las detecciones "
            "generales del modelo."
        ),
    )

    gloves_threshold = st.slider(
        "Umbral para Gloves 🧤",
        min_value=0.25,
        max_value=0.95,
        value=DEFAULT_GLOVES_THRESHOLD,
        step=0.05,
        help=(
            "Umbral específico para reducir "
            "falsos positivos de guantes."
        ),
    )

    grpc_address = st.text_input(
        "Servicio gRPC",
        value="localhost:50051",
    )

    st.divider()

    st.subheader("Rendimiento")

    st.write(
        f"Procesando 1 de cada "
        f"{PROCESS_EVERY_N_FRAMES} frames"
    )

    st.write(
        f"Resolución máxima: "
        f"{MAX_FRAME_SIDE}px"
    )


st.subheader("🎥 Cámara en tiempo real")

webrtc_ctx = webrtc_streamer(
    key="epp-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EPPVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.update_config(
        conf_threshold=conf_threshold,
        gloves_threshold=gloves_threshold,
        grpc_address=grpc_address,
    )


st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Modelo",
        "YOLO11n",
    )

with col2:
    st.metric(
        "Arquitectura",
        "gRPC",
    )

with col3:
    st.metric(
        "Tracking",
        "MLflow",
    )


st.info(
    "Presiona START para activar la cámara. "
    "El video será procesado continuamente por "
    "YOLO11n mediante el servicio gRPC."
)