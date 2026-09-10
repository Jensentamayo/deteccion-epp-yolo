# 🦺 deteccion-epp-yolo

![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-8A2BE2)
![Ruff](https://img.shields.io/badge/lint-ruff-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-en%20desarrollo-yellow)

Detección de Elementos de Protección Personal (**casco** y **guantes**) en imágenes/video en tiempo real, usando **YOLO11n** (Ultralytics), servido vía **gRPC**, con cliente **Streamlit** y tracking de experimentos/inferencias en **MLflow**.

Proyecto académico — Módulo 3 (Talos IA).

## 📐 Arquitectura

```
Streamlit (Vista/Cliente) ──gRPC──▶ Servicio de Inferencia (Controlador + Modelo YOLO11n)
                                            │
                                            ▼
                                       MLflow Tracking
```

Ver diagrama interactivo en [`docs/arquitectura.html`](docs/arquitectura.html) (generado con Archify).

### Patrón MVC

| Capa | Módulo | Responsabilidad |
|---|---|---|
| Modelo | `src/models/load_model.py`, `src/models/predict_model.py` | Carga de YOLO11n y ejecución de inferencia |
| Controlador | `src/data/preprocess_img.py`, `src/visualizations/draw_boxes.py`, `src/integrator.py` | Preprocesamiento, orquestación y anotación de resultados |
| Vista/Cliente | `app/detector_epp.py` | Interfaz Streamlit que consume el servicio gRPC |
| Servicio | `src/grpc_service/` | Contrato `.proto`, servidor y cliente gRPC |
| Tracking | `src/tracking/mlflow_tracker.py` | Registro de runs en MLflow |

## 🧰 Stack

- **Modelo**: [YOLO11n](https://docs.ultralytics.com/models/yolo11/) (Ultralytics)
- **Dataset**: [PPE Detection with Gloves](https://universe.roboflow.com/ppe-detection-82plm/ppe-detection-with-gloves) (Roboflow)
- **Servicio**: gRPC (contratos `.proto`)
- **Frontend**: Streamlit
- **Tracking**: MLflow
- **Gestión de entorno**: uv (Python 3.13)
- **Calidad**: Ruff, pytest

## 🚀 Instalación

```bash
# 1. Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar el repositorio
git clone <url-del-repo>
cd deteccion-epp-yolo

# 3. Sincronizar el entorno (crea .venv automáticamente)
uv sync
```

### Descargar dataset y colocar el modelo

1. Descarga el dataset desde Roboflow (formato YOLO) en `data/raw/`.
2. Entrena o descarga los pesos `.pt` de YOLO11n ajustado y colócalos en `models/epp_yolo11n.pt`
   (ruta configurable con la variable de entorno `EPP_MODEL_PATH`).

### Generar los stubs de gRPC

Los archivos `inference_pb2.py` e `inference_pb2_grpc.py` **no se versionan** (se generan localmente):

```bash
uv run bash scripts/generate_grpc.sh
```

## ▶️ Uso

**1. Levantar MLflow** (en una terminal):

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

**2. Levantar el servicio de inferencia gRPC** (en otra terminal):

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export EPP_MODEL_PATH=models/epp_yolo11n.pt
uv run python -m src.grpc_service.server
```

**3. Levantar la interfaz Streamlit** (en otra terminal):

```bash
uv run streamlit run app/detector_epp.py
```

Abre `http://localhost:8501`, sube una imagen y presiona **"Ejecutar detección"**.

### Con Docker Compose (todo el stack)

```bash
docker compose up --build
```

## 🧪 Pruebas

```bash
uv run pytest
```

La suite cubre: preprocesamiento (`test_preprocess_img.py`), dibujo de detecciones
(`test_draw_boxes.py`), inferencia (`test_predict_model.py`), carga de modelo
(`test_load_model.py`), orquestación (`test_integrator.py`), tracking MLflow
(`test_mlflow_tracker.py`) y el servicio gRPC (`test_grpc_server.py`, requiere
haber generado los stubs). Los tests del modelo/gRPC usan **mocks** de YOLO,
por lo que no requieren los pesos reales para ejecutarse.

> Meta del curso: mínimo 120 pruebas unitarias. La base actual está parametrizada
> para extenderse fácilmente — ver sección "Próximos pasos".

## 🧹 Calidad de código

```bash
uv run ruff check .
uv run ruff format .
```

## 📁 Estructura del repositorio

```
deteccion-epp-yolo/
├── data/{raw,processed,external}
├── src/
│   ├── data/preprocess_img.py
│   ├── models/{load_model.py, predict_model.py}
│   ├── visualizations/draw_boxes.py
│   ├── grpc_service/{inference.proto, server.py, client.py}
│   ├── tracking/mlflow_tracker.py
│   └── integrator.py
├── app/detector_epp.py          # Vista Streamlit
├── tests/                       # pytest
├── scripts/generate_grpc.sh
├── docs/arquitectura.html       # Diagrama Archify
├── Dockerfile / docker-compose.yml
├── pyproject.toml               # uv + ruff + pytest config
└── README.md
```

## 🌳 Gitflow

- `main`: versión estable / entregable.
- `develop`: integración de features.
- `feature/*`: una rama por funcionalidad, mergeada a `develop` vía Pull Request
  usando la plantilla en [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).

## 📊 MLflow: qué se registra

| Señal | Contenido |
|---|---|
| Params | referencia del modelo en el Hub, umbral de confianza, config de preprocesamiento |
| Metrics | nº de detecciones, nº de incumplimientos, latencia de inferencia (ms) |
| Tags | licencia del modelo, ambiente, servicio |

## 🗺️ Próximos pasos

1. **Completar hasta 120 pruebas unitarias**: extender los `@pytest.mark.parametrize`
   existentes con más casos límite (imágenes corruptas, múltiples clases simultáneas,
   umbrales extremos 0.0/1.0, concurrencia del servidor gRPC).
2. **Entrenar/ajustar YOLO11n** sobre el dataset real de Roboflow y registrar la
   evaluación (precision/recall/F1/mAP50) con `MLflowTracker.log_evaluation`.
2. **Video en tiempo real**: extender `app/detector_epp.py` para procesar streams
   de webcam (frame a frame) en vez de solo imágenes estáticas.
3. **CI/CD (Módulo 4)**: pipeline GitLab `test → build → deploy` hacia un Droplet
   de DigitalOcean.
4. **Autenticación/TLS** en el canal gRPC para el despliegue en producción.
5. **Diagrama Archify**: generar `docs/arquitectura.html` (Architecture + Sequence
   para la ruta de inferencia) con `npx skills add tt-a1i/archify -g`.

## 📄 Licencia

MIT — ver [`LICENSE`](LICENSE).
