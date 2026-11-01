import os
import subprocess
import json
import streamlit as st
import requests
import pandas as pd
import time

st.set_page_config(
    page_title="California Housing MLOps - Inferencia en Kubernetes",
    page_icon="🏠",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("🏠 California Housing - Servicio de Inferencia en Kubernetes")
st.markdown("""
**Módulo:** MIAV1E414 — Proyecto Final MLOps  
**Infraestructura:** Clúster de Kubernetes con múltiples réplicas detrás de un Service `ClusterIP`.  
El modelo desplegado corresponde a la versión **`5@champion`** (XGBoost GPU) registrada en **MLflow Model Registry**.
""")

# Barra lateral con estado de conexión a Kubernetes
st.sidebar.header("⚙️ Estado de Kubernetes Service")
st.sidebar.text(f"Endpoint: {API_URL}")

try:
    health_resp = requests.get(f"{API_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        h_data = health_resp.json()
        st.sidebar.success(f"🟢 Servicio Activo y Saludable")
        st.sidebar.info(f"Pod conectado: `{h_data.get('pod_name', 'N/A')}`")
    else:
        st.sidebar.warning(f"🟡 Servicio respondió código {health_resp.status_code}")
except Exception as e:
    st.sidebar.error(f"🔴 No se pudo conectar al Service de Kubernetes: {e}")

tabs = st.tabs(["🔮 Inferencia Individual", "⚖️ Auditoría de Balanceo de Carga (Pods)", "ℹ️ Metadatos del Modelo"])

with tabs[0]:
    st.subheader("Predicción del Valor Mediano de Vivienda")
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.slider("MedInc (Ingreso medio en decenas de miles de USD)", 0.5, 15.0, 3.87, 0.1)
        house_age = st.slider("HouseAge (Antigüedad media de la vivienda)", 1.0, 52.0, 28.0, 1.0)
        ave_rooms = st.slider("AveRooms (Habitaciones promedio)", 1.0, 15.0, 5.4, 0.1)
        ave_bedrms = st.slider("AveBedrms (Dormitorios promedio)", 0.5, 5.0, 1.05, 0.05)

    with col2:
        population = st.number_input("Population (Población del bloque)", min_value=10.0, max_value=35000.0, value=1425.0, step=50.0)
        ave_occup = st.slider("AveOccup (Ocupación promedio por hogar)", 1.0, 10.0, 3.0, 0.1)
        latitude = st.slider("Latitude", 32.5, 42.0, 37.88, 0.01)
        longitude = st.slider("Longitude", -124.5, -114.3, -122.23, 0.01)

    if st.button("🚀 Enviar Inferencia al Clúster de Kubernetes", type="primary"):
        payload = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }

        with st.spinner("Consultando Service en Kubernetes..."):
            try:
                t0 = time.time()
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                elapsed_ms = round((time.time() - t0) * 1000, 2)

                if res.status_code == 200:
                    data = res.json()
                    pred_val = data["prediction"]
                    pod_resp = data["pod_name"]
                    version = data["model_version"]

                    st.success("✅ Inferencia completada con éxito")
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Valor Estimado de Vivienda", f"${pred_val * 100000:,.0f} USD", f"{pred_val:.4f} MedVal")
                    mcol2.metric("Pod de Kubernetes que Atendió", pod_resp)
                    mcol3.metric("Tiempo de Respuesta", f"{elapsed_ms} ms", f"Versión: {version}")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Error en la petición HTTP: {e}")

with tabs[1]:
    st.subheader("Auditoría en Tiempo Real de Balanceo de Tráfico entre Réplicas")
    st.markdown("Envía una ráfaga de peticiones sucesivas al Service de Kubernetes para verificar que `kube-proxy` distribuye las solicitudes entre los distintos Pods activos.")
    
    num_reqs = st.slider("Cantidad de solicitudes a enviar", 5, 20, 10)
    
    if st.button("📡 Enviar Ráfaga de Tráfico"):
        results = []
        progress_bar = st.progress(0)
        
        sample_payload = {
            "MedInc": 3.87, "HouseAge": 28.0, "AveRooms": 5.4, "AveBedrms": 1.05,
            "Population": 1425.0, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23
        }

        for i in range(num_reqs):
            resp_json = None
            try:
                # Enviar directo a través de kube-proxy en el clúster para evidenciar el balanceo entre pods
                cmd = [
                    "docker", "exec", "-i", "desktop-control-plane",
                    "curl", "-s", "-X", "POST", "http://10.96.136.6/predict",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(sample_payload)
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if proc.returncode == 0 and proc.stdout.strip():
                    resp_json = json.loads(proc.stdout.strip())
            except Exception:
                pass

            if not resp_json:
                try:
                    r = requests.post(f"{API_URL}/predict", json=sample_payload, timeout=3)
                    if r.status_code == 200:
                        resp_json = r.json()
                except Exception as e:
                    results.append({"Petición": i + 1, "Pod Atendedor": "Error", "Estado": str(e)})

            if resp_json:
                results.append({
                    "Petición": i + 1,
                    "Pod Atendedor": resp_json.get("pod_name", "N/A"),
                    "Predicción": resp_json.get("prediction", 0.0),
                    "Versión": resp_json.get("model_version", "N/A"),
                    "Estado": "OK"
                })
            
            progress_bar.progress((i + 1) / num_reqs)
            time.sleep(0.05)

        if results:
            df_res = pd.DataFrame(results)
            col_t1, col_t2 = st.columns([1, 1])
            with col_t1:
                st.dataframe(df_res, use_container_width=True)
            with col_t2:
                pod_counts = df_res["Pod Atendedor"].value_counts()
                st.bar_chart(pod_counts)
                st.info(f"Total de Pods diferentes que respondieron: **{len(pod_counts)} réplicas**")

with tabs[2]:
    st.subheader("Información y Trazabilidad del Modelo")
    st.markdown("""
    - **Registro en MLflow:** `modelo-inmobiliario-prod`
    - **Alias en Producción:** `@champion`
    - **Algoritmo:** XGBoost Regressor (Acelerado por GPU CUDA en fase de entrenamiento)
    - **Métrica de Selección:** $R^2 = 0.8415$ (Superó a Gradient Boosting y Random Forest)
    - **Contenerización:** Docker multi-arch / Python 3.11-slim
    - **Orquestador:** Kubernetes Deployment con 3 réplicas y Service ClusterIP
    """)

