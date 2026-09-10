"""Preprocesamiento de imágenes de entrada antes de la inferencia.

Responsabilidad (capa Controlador): normalizar el formato de entrada
(bytes o array) a un arreglo RGB apto para YOLO, con validaciones.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 1280  # Límite razonable para evitar imágenes desproporcionadas.


class InvalidImageError(ValueError):
    """Se lanza cuando los bytes de entrada no representan una imagen válida."""


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes (jpg/png) a un arreglo numpy RGB.

    Args:
        image_bytes: Contenido binario de la imagen (ej. subida por Streamlit
            o recibida por gRPC).

    Returns:
        Arreglo numpy (H, W, 3) en formato RGB.

    Raises:
        InvalidImageError: si los bytes no pueden decodificarse como imagen.
    """
    if not image_bytes:
        raise InvalidImageError("Los bytes de la imagen están vacíos.")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError(f"No se pudo decodificar la imagen: {exc}") from exc
    return np.array(pil_image)


def resize_if_needed(image: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """Redimensiona la imagen si su lado mayor supera `max_side`, preservando aspecto."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def preprocess(image_bytes: bytes, max_side: int = MAX_SIDE) -> np.ndarray:
    """Pipeline completo de preprocesamiento: decodificar + redimensionar.

    Args:
        image_bytes: Bytes crudos de la imagen recibida.
        max_side: Lado máximo permitido antes de redimensionar.

    Returns:
        Imagen RGB lista para pasar al modelo YOLO.
    """
    image = decode_image_bytes(image_bytes)
    return resize_if_needed(image, max_side=max_side)
