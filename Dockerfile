# Imagen base oficial y liviana de Python
FROM python:3.11-slim

# Metadatos del contenedor
LABEL maintainer="MIAV1E414 - Proyecto Final MLOps"
LABEL description="Servicio de inferencia de modelo California Housing entrenado y registrado en MLflow"

# Evitar escritura de archivos .pyc y forzar salida inmediata por stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="model_champion" \
    PORT=8000

# Directorio de trabajo en el contenedor
WORKDIR /app

# Instalar dependencias del sistema minimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivo de requerimientos fijados
COPY requirements.txt .

# Instalar dependencias de Python optimizadas
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el codigo del servicio de inferencia y el modelo empaquetado
COPY app.py .
COPY model_champion/ ./model_champion/

# Puerto expuesto
EXPOSE 8000

# Healthcheck interno del contenedor
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio del microservicio (Uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

