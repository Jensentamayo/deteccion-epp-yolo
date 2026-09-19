# 🦺 Detección de incumplimiento de EPP con YOLO11n

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLO](https://img.shields.io/badge/YOLO-11n-purple)
![gRPC](https://img.shields.io/badge/gRPC-Architecture-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![Tests](https://img.shields.io/badge/Tests-120%20passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Descripción

Sistema de inteligencia artificial para la **detección en tiempo real de incumplimientos en el uso de elementos de protección personal (EPP)** mediante visión por computador y el modelo **YOLO11n**.

El sistema analiza imágenes provenientes de una cámara y detecta específicamente dos tipos de incumplimiento:

* 🔴 `no_helmet`: ausencia de casco de seguridad.
* 🔴 `no_gloves`: ausencia de guantes de seguridad.

El proyecto implementa una arquitectura desacoplada mediante **gRPC**, seguimiento experimental con **MLflow**, una interfaz interactiva desarrollada con **Streamlit** y despliegue mediante **Docker Compose**.

---

## 🎯 Objetivo

Desarrollar una solución de inteligencia artificial capaz de identificar en tiempo real situaciones de incumplimiento relacionadas con el uso de casco y guantes de seguridad, proporcionando una visualización inmediata de las detecciones.

### Alcance

El modelo final está limitado exclusivamente a las siguientes clases:

| ID | Clase       | Descripción                      |
| -: | ----------- | -------------------------------- |
|  0 | `no_helmet` | Persona sin casco de seguridad   |
|  1 | `no_gloves` | Persona sin guantes de seguridad |

> El sistema no realiza clasificación de EPP completo ni utiliza clases como `helmet`, `gloves`, `mask`, `vest`, `goggles` o `safety_shoe`.

---

## 🤖 Modelo

Modelo utilizado:

```text
YOLO11n
```

Modelo final:

```text
models/trained/epp_no_compliance_yolo11n_final.pt
```

### Métricas de evaluación

| Métrica           |   Resultado |
| ----------------- | ----------: |
| Precision global  |       0.799 |
| Recall global     |       0.692 |
| mAP@50            |       0.765 |
| mAP@50-95         |       0.372 |
| Accuracy          |  95.64498 % |
| AUC-ROC           |  98.07549 % |
| Precision         |     0.84852 |
| Recall            |     0.68301 |
| F1-Score          |     0.75682 |
| mAP@50            |     0.75589 |
| mAP@50-95         |     0.36119 |
| Latencia promedio |   43.651 ms |
| Latencia P95      |   48.338 ms |
| Throughput        | 22.9087 FPS |

### Rendimiento por clase

| Clase       | Precision | Recall | mAP@50 | mAP@50-95 |
| ----------- | --------: | -----: | -----: | --------: |
| `no_helmet` |     0.842 |  0.772 |  0.836 |     0.462 |
| `no_gloves` |     0.756 |  0.613 |  0.694 |     0.283 |

---

## 🏗️ Arquitectura

El sistema utiliza una arquitectura desacoplada donde la interfaz de usuario, la lógica de comunicación y la inferencia del modelo se encuentran separadas.

```text
┌──────────────────────────────┐
│          Streamlit           │
│      Interfaz de usuario     │
└──────────────┬───────────────┘
               │
               │ gRPC
               ▼
┌──────────────────────────────┐
│      Inference Service       │
│       Servicio gRPC          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          YOLO11n             │
│       Modelo de IA/ML        │
└──────────────┬───────────────┘
               │
               ▼
       Detecciones:
       no_helmet
       no_gloves

               │
               ▼
┌──────────────────────────────┐
│           MLflow             │
│ Tracking de experimentos     │
└──────────────────────────────┘
```

### Diagrama interactivo

El diagrama interactivo de arquitectura se encuentra en:

```text
docs/arquitectura.html
```

Puede visualizarse localmente mediante:

```bash
python3 -m http.server 8000 --directory docs
```

y posteriormente accediendo a:

```text
http://localhost:8000/arquitectura.html
```

---

## 🔄 Flujo de inferencia

1. El usuario proporciona una imagen o utiliza una cámara.
2. Streamlit recibe la imagen.
3. Streamlit envía la solicitud mediante gRPC.
4. El servicio de inferencia recibe la imagen.
5. YOLO11n procesa la imagen.
6. Se detectan las clases `no_helmet` y `no_gloves`.
7. El servicio devuelve las detecciones.
8. Streamlit visualiza los resultados.
9. MLflow permite registrar y consultar información relacionada con las ejecuciones.

---

## 🧩 Organización MVC

El proyecto mantiene una separación de responsabilidades basada en el patrón MVC:

### Model

Contiene los componentes relacionados directamente con el modelo de inteligencia artificial:

```text
src/models/
```

Responsabilidades:

* Carga del modelo YOLO11n.
* Ejecución de inferencia.
* Procesamiento de resultados.

### Controller

Contiene la lógica de procesamiento y comunicación:

```text
src/
```

Responsabilidades:

* Comunicación gRPC.
* Procesamiento de imágenes.
* Integración de componentes.
* Lógica de inferencia.

### View

La interfaz de usuario se encuentra en:

```text
app/detector_epp.py
```

Está implementada utilizando Streamlit.

---

## 🌐 Arquitectura gRPC

El proyecto utiliza **gRPC** para separar el cliente de la interfaz de usuario del servicio encargado de realizar la inferencia.

El contrato de comunicación se define mediante Protocol Buffers:

```text
proto/
```

La arquitectura permite separar:

```text
Cliente Streamlit
       │
       │ gRPC
       ▼
Servicio de inferencia
       │
       ▼
Modelo YOLO11n
```

Esta separación facilita el mantenimiento, las pruebas y el despliegue independiente de los componentes.

---

## 📊 MLflow

MLflow se utiliza para el seguimiento de experimentos y evaluaciones del modelo.

El servidor puede iniciarse mediante Docker Compose:

```bash
docker compose up -d
```

La interfaz de MLflow estará disponible en:

```text
http://localhost:5000
```

### Información registrada

El seguimiento contempla información relevante para reproducibilidad y evaluación:

**Parámetros**

* Modelo utilizado.
* Configuración de inferencia.
* Umbral de confianza.
* Configuración de procesamiento.
* Dataset utilizado.
* Entorno de ejecución.

**Métricas**

* Accuracy.
* Precision.
* Recall.
* F1-Score.
* AUC-ROC.
* mAP@50.
* mAP@50-95.
* Latencia promedio.
* Latencia P95.
* Throughput.

**Artefactos**

* Resultados de evaluación.
* Predicciones.
* Gráficas.
* Información relacionada con el modelo y el pipeline.

**Tags**

* Entorno.
* Autor/equipo.
* Información del proyecto.
* Referencias de desarrollo.

---

## 🖥️ Interfaz Streamlit

La aplicación proporciona una interfaz para realizar inferencias mediante:

* 📷 Cámara.
* 🖼️ Imágenes.
* 📦 Servicio gRPC.
* 🔴 Visualización de incumplimientos.

Las detecciones de:

```text
no_helmet
no_gloves
```
Ejemplo de funcionamiento:
Sin casco =
<img width="2312" height="1092" alt="no_helmet" src="https://github.com/user-attachments/assets/e6d02aed-4b91-40ce-82ba-cf9b5e5cdafb" />

Sin guantes = 
<img width="2316" height="1078" alt="no_gloves" src="https://github.com/user-attachments/assets/3da66efc-5188-4e27-a9dd-8372d77d2113" />

son mostradas visualmente sobre la imagen procesada.

---

## 🐳 Docker

El proyecto dispone de un entorno de ejecución mediante Docker Compose.

Servicios principales:

```text
┌────────────────────┐
│      Streamlit     │
│       :8501        │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│      Inference     │
│       :50051       │
└─────────┬──────────┘
          │
          ▼
       YOLO11n

┌────────────────────┐
│       MLflow       │
│       :5000        │
└────────────────────┘
```

### Iniciar el sistema

```bash
docker compose up -d
```

### Servicios

Streamlit:

```text
http://localhost:8501
```

MLflow:

```text
http://localhost:5000
```

### Detener el sistema

```bash
docker compose down
```

### Consultar logs

```bash
docker compose logs -f inference
```

---

## 💻 Ejecución local

### Requisitos

* Python 3.13
* uv
* Git
* Docker
* Docker Compose

### Instalar dependencias

El entorno está administrado mediante `uv`.

```bash
uv sync
```

### Ejecutar Streamlit

```bash
uv run streamlit run app/detector_epp.py
```

### Ejecutar pruebas

```bash
uv run pytest -q
```

Resultado actual:

```text
120 passed
```

### Ejecutar Ruff

```bash
uv run ruff check app src tests scripts
```

Resultado esperado:

```text
All checks passed!
```

---

## 🧪 Pruebas

El proyecto cuenta actualmente con:

```text
120 pruebas unitarias
```

Las pruebas cubren diferentes componentes del sistema, incluyendo:

* Procesamiento de imágenes.
* Comunicación gRPC.
* Inferencia.
* Integración.
* Validación de datos.
* Componentes de la aplicación.

La ejecución se realiza mediante:

```bash
uv run pytest -q
```

La cobertura actual se encuentra aproximadamente en:

```text
89 %
```

---

## 🌿 Gitflow

El desarrollo del proyecto utiliza una estrategia basada en Gitflow, separando las ramas según su propósito.

```text
                    ┌──────────────┐
                    │     main     │
                    │ versión      │
                    │   estable    │
                    └──────▲───────┘
                           │
                           │
                    ┌──────┴───────┐
                    │   develop    │
                    │ integración  │
                    │              │
                    └──────▲───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
      ┌───────┴────────┐       ┌────────┴───────┐
      │    feature/*   │       │    feature/*   │
      │ nueva función  │       │ documentación  │
      └────────────────┘       └────────────────┘
```

### Ramas principales

**`main`**

Contiene la versión estable del proyecto.

**`develop`**

Contiene la versión de integración y desarrollo.

**`feature/*`**

Se utilizan para desarrollar funcionalidades, correcciones o mejoras específicas sin modificar directamente `develop`.

### Flujo de trabajo

```text
feature/*
    │
    │ desarrollo
    ▼
commit
    │
    ▼
push
    │
    ▼
Pull Request
    │
    ▼
develop
    │
    │ versión validada
    ▼
main
```

Las modificaciones se desarrollan en ramas `feature/*`. Después de completar y validar una modificación, se crea un Pull Request hacia `develop` para revisión e integración.

Este flujo permite mantener separadas las funcionalidades en desarrollo de las versiones de integración y estable del proyecto.

---

## 📁 Estructura del proyecto

```text
deteccion-epp-yolo/
│
├── app/
│   └── detector_epp.py
│
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│
├── docs/
│   └── arquitectura.html
│
├── models/
│   └── trained/
│       └── epp_no_compliance_yolo11n_final.pt
│
├── proto/
│   └── *.proto
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── data/
│   ├── features/
```
