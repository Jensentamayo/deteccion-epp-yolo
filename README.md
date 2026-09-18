# 🦺 Detección de Incumplimiento de EPP con YOLO11n

![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/gestor-uv-8A2BE2)
![Ruff](https://img.shields.io/badge/lint-ruff-red)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema de visión por computador para la **detección en tiempo real de incumplimientos de elementos de protección personal (EPP)** mediante **YOLO11n**, **gRPC**, **Streamlit**, **MLflow** y **Docker**.

Proyecto académico — Módulo 3 de Inteligencia Artificial y Machine Learning.

---

## 📌 Descripción

El proyecto implementa una solución de visión por computador capaz de analizar imágenes y video provenientes de una cámara para detectar situaciones en las que un trabajador no utiliza determinados elementos de protección personal.

El modelo final está especializado exclusivamente en la detección de dos tipos de incumplimiento:

* `no_helmet`: ausencia de casco de seguridad.
* `no_gloves`: ausencia de guantes de seguridad.

Las detecciones se visualizan mediante cajas de color rojo en la interfaz.

---

## 🎯 Objetivo

Desarrollar una solución de Inteligencia Artificial capaz de identificar incumplimientos de EPP en tiempo real y presentar los resultados mediante una interfaz web.

La solución integra:

* Modelo de detección YOLO11n.
* Servicio de inferencia mediante gRPC.
* Interfaz web con Streamlit.
* Seguimiento de experimentos e inferencias con MLflow.
* Contenerización mediante Docker.
* Pruebas automatizadas con pytest.
* Control de calidad mediante Ruff.

---

## 🔎 Alcance

El modelo final detecta únicamente las siguientes clases:

| Clase       | Descripción                      |
| ----------- | -------------------------------- |
| `no_helmet` | Persona sin casco de seguridad   |
| `no_gloves` | Persona sin guantes de seguridad |

Las siguientes clases **no forman parte del alcance final**:

* `helmet`
* `gloves`
* `mask`
* `vest`
* `goggles`
* `safety_shoe`
* EPP completo

El objetivo del sistema es identificar específicamente **incumplimientos**, no determinar si una persona cumple con todos los elementos de protección.

---

## 🤖 Modelo

**Modelo:** YOLO11n
**Framework:** Ultralytics

Pesos finales:

```text
models/trained/epp_no_compliance_yolo11n_final.pt
```

Clases del modelo:

```text
0 - no_helmet
1 - no_gloves
```

El modelo final fue obtenido mediante un proceso de ajuste del modelo YOLO11n sobre datos de detección de EPP.

> El entrenamiento utilizado corresponde a un proceso de fine-tuning secuencial. No se considera un `resume` real del estado completo del optimizador.

---

## 📊 Resultados del modelo

Resultados obtenidos sobre el conjunto de prueba:

| Métrica   | Resultado |
| --------- | --------: |
| Precision |     79.9% |
| Recall    |     69.2% |
| mAP@50    |     76.5% |
| mAP@50-95 |     37.2% |

### Resultados por clase

| Clase       | Precision | Recall | mAP@50 | mAP@50-95 |
| ----------- | --------: | -----: | -----: | --------: |
| `no_helmet` |     84.2% |  77.2% |  83.6% |     46.2% |
| `no_gloves` |     75.6% |  61.3% |  69.4% |     28.3% |

### Evaluación formal

```text
Accuracy:           95.64%
AUC-ROC:            98.08%
Precision:          84.85%
Recall:             68.30%
F1-score:           75.68%
mAP@50:             75.59%
mAP@50-95:          36.12%
Latencia promedio:  43.65 ms
Latencia P95:       48.34 ms
Throughput:         22.91 FPS
```

---

## 🏗️ Arquitectura

La solución separa la interfaz de usuario, la comunicación y la inferencia del modelo.

```text
                    ┌──────────────────┐
                    │    Streamlit     │
                    │   Interfaz Web   │
                    └────────┬─────────┘
                             │
                            gRPC
                             │
                             ▼
                    ┌──────────────────┐
                    │ Servicio gRPC    │
                    │   de inferencia  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     YOLO11n      │
                    │ Modelo entrenado │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     MLflow       │
                    │     Tracking     │
                    └──────────────────┘
```

El diagrama interactivo de arquitectura se encuentra en:

`docs/arquitectura.html`

---

## 📐 Patrón MVC

El proyecto organiza las responsabilidades siguiendo una separación basada en MVC:

| Capa          | Módulo                           | Responsabilidad                      |
| ------------- | -------------------------------- | ------------------------------------ |
| Modelo        | `src/models/`                    | Carga y ejecución del modelo YOLO11n |
| Controlador   | `src/data/`, `src/integrator.py` | Preprocesamiento y orquestación      |
| Vista         | `app/detector_epp.py`            | Interfaz Streamlit                   |
| Servicio      | `src/grpc_service/`              | Comunicación mediante gRPC           |
| Visualización | `src/visualizations/`            | Dibujo de detecciones                |
| Tracking      | `src/tracking/`                  | Registro de información en MLflow    |

---

## 🔌 Servicio gRPC

La comunicación entre Streamlit y el modelo se realiza mediante un servicio gRPC independiente.

Contrato:

```text
src/grpc_service/inference.proto
```

El servicio permite enviar imágenes al servidor de inferencia y recibir información sobre:

* Clase detectada.
* Coordenadas de las cajas.
* Confianza.
* Número de detecciones.
* Número de incumplimientos.
* Tiempo de inferencia.
* Versión del modelo.

Los archivos generados por Protocol Buffers se generan localmente y no forman parte del control de versiones.

---

## 📈 MLflow

MLflow se utiliza para realizar seguimiento de las inferencias y registrar información relevante del sistema.

### Parámetros

| Parámetro        | Descripción                        |
| ---------------- | ---------------------------------- |
| `model_revision` | Versión o referencia del modelo    |
| `preprocessing`  | Configuración del preprocesamiento |
| `model_hub_ref`  | Referencia del modelo utilizado    |
| `conf_threshold` | Umbral de confianza                |

### Métricas

| Métrica             | Descripción                     |
| ------------------- | ------------------------------- |
| `inference_time_ms` | Tiempo de inferencia            |
| `throughput_fps`    | Imágenes procesadas por segundo |
| `num_violations`    | Número de incumplimientos       |
| `num_detections`    | Número de detecciones           |
| `latency_p95_ms`    | Latencia P95                    |
| `latency_mean_ms`   | Latencia promedio               |

### Tags

Se registran etiquetas relacionadas con:

* Equipo de desarrollo.
* Ambiente.
* Servicio.
* Autor del modelo.
* Licencia del modelo.
* Referencia del issue o tarea.

---

## 🖥️ Streamlit

La aplicación web permite:

* Utilizar la cámara del equipo.
* Capturar imágenes.
* Ejecutar inferencias.
* Visualizar las detecciones.
* Mostrar las cajas de incumplimiento.
* Mostrar el nivel de confianza.
* Configurar la dirección del servicio gRPC.

Archivo principal:

```text
app/detector_epp.py
```

---

## 🐳 Docker

El proyecto utiliza Docker Compose para ejecutar el stack completo:

```text
┌──────────────┐
│   Streamlit  │ :8501
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Inference  │ :50051
│     gRPC     │
└──────┬───────┘
       │
       ▼
    YOLO11n

┌──────────────┐
│    MLflow    │ :5000
└──────────────┘
```

### Iniciar el sistema

```bash
docker compose up -d
```

### Ver los servicios

```bash
docker compose ps
```

### Streamlit

Abrir:

`http://localhost:8501`

### MLflow

Abrir:

`http://localhost:5000`

### Ver logs

```bash
docker compose logs -f inference
```

### Detener

```bash
docker compose down
```

---

## ⚙️ Ejecución local

Instalar las dependencias:

```bash
uv sync
```

Iniciar el servicio gRPC:

```bash
uv run python -m src.grpc_service.server
```

En otra terminal, iniciar Streamlit:

```bash
uv run streamlit run app/detector_epp.py
```

---

## 🧪 Pruebas

La calidad del proyecto se valida mediante pytest.

Ejecutar:

```bash
uv run pytest -q
```

Estado actual:

```text
125 passed
```

Cobertura aproximada:

```text
85%
```

Las pruebas cubren componentes relacionados con:

* Preprocesamiento.
* Carga del modelo.
* Inferencia.
* Integrador.
* Visualización.
* Servicio gRPC.
* Cliente gRPC.
* Tracking MLflow.

Los tests utilizan mocks cuando es necesario para evitar depender de los pesos reales del modelo.

---

## 🧹 Calidad de código

El proyecto utiliza Ruff para análisis estático y formato.

Ejecutar:

```bash
uv run ruff check app src tests scripts
```

Resultado actual:

```text
All checks passed!
```

---

## 📁 Estructura del proyecto

```text
deteccion-epp-yolo/
│
├── app/
│   └── detector_epp.py
│
├── data/
│
├── docs/
│   └── arquitectura.html
│
├── models/
│   └── trained/
│       └── epp_no_compliance_yolo11n_final.pt
│
├── reports/
│
├── runs/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── grpc_service/
│   ├── models/
│   ├── tracking/
│   ├── visualizations/
│   └── integrator.py
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 🔄 Flujo de detección

```text
                 Cámara
                    │
                    ▼
              ┌───────────┐
              │ Streamlit │
              └─────┬─────┘
                    │
                   gRPC
                    │
                    ▼
              ┌───────────┐
              │  YOLO11n  │
              └─────┬─────┘
                    │
                    ▼
             Detecciones
                    │
              ┌─────┴─────┐
              │           │
              ▼           ▼
          no_helmet    no_gloves
              │           │
              └─────┬─────┘
                    ▼
             Visualización
```

---

## 🌳 Gitflow

El proyecto utiliza ramas para organizar el desarrollo:

```text
main
 │
 └── develop
       │
       ├── feature/*
       ├── bugfix/*
       └── docs/*
```

Las funcionalidades y correcciones se integran mediante Pull Requests.

---

## ⚠️ Limitaciones

* El modelo detecta únicamente `no_helmet` y `no_gloves`.
* La iluminación puede afectar el desempeño de la detección.
* La distancia respecto a la cámara puede afectar los resultados.
* La detección de `no_gloves` presenta menor desempeño que `no_helmet`.
* El rendimiento depende del hardware disponible.
* El sistema es un proyecto académico y no constituye por sí mismo un sistema certificado de seguridad industrial.

---

## 🛠️ Tecnologías

| Tecnología       | Uso                                |
| ---------------- | ---------------------------------- |
| Python 3.13      | Lenguaje principal                 |
| YOLO11n          | Detección de objetos               |
| Ultralytics      | Framework de visión por computador |
| gRPC             | Comunicación entre servicios       |
| Protocol Buffers | Contrato del servicio              |
| Streamlit        | Interfaz web                       |
| MLflow           | Tracking y experimentación         |
| Docker           | Contenerización                    |
| Docker Compose   | Orquestación                       |
| pytest           | Pruebas automatizadas              |
| Ruff             | Calidad de código                  |
| uv               | Gestión de dependencias            |

---

## 📚 Fuentes y dependencias externas

El proyecto utiliza tecnologías y recursos desarrollados por terceros.

### YOLO11n / Ultralytics

Modelo y framework de detección proporcionados por Ultralytics.

Documentación:

https://docs.ultralytics.com/

### Dataset

Dataset utilizado durante el desarrollo:

https://universe.roboflow.com/ppe-detection-82plm/ppe-detection-with-gloves

La licencia y condiciones de uso del dataset deben consultarse directamente en su fuente original.

### Dependencias

Las dependencias de software utilizadas por el proyecto pueden tener sus propias licencias y condiciones de uso. La licencia MIT de este repositorio se aplica al código propio del proyecto y no reemplaza las licencias de terceros.

---

## 📄 Licencia

El código desarrollado específicamente para este proyecto se distribuye bajo la **MIT License**.

El texto completo de la licencia se encuentra en:

`LICENSE`

La licencia MIT permite utilizar, modificar y distribuir el código bajo sus condiciones, incluyendo la conservación del aviso de copyright y de la licencia.

Los modelos, datasets, frameworks y demás recursos de terceros permanecen sujetos a sus respectivas licencias y términos de uso.

---

## 👨‍💻 Autor

**Nicolás Bolaños Esterling**

Proyecto académico de Inteligencia Artificial y Machine Learning.
