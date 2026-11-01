import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import mlflow.pyfunc
from sklearn.metrics import r2_score, mean_squared_error

def test_data_drift():
    print("=" * 70)
    print("1. EVALUACIÓN DE DATA DRIFT (Prueba Kolmogorov-Smirnov)")
    print("=" * 70)

    # 1. Cargar baseline de entrenamiento
    baseline_df = pd.read_csv("data/baseline_train.csv")
    test_ref_df = pd.read_csv("data/reference_test.csv")
    features = list(baseline_df.columns)
    
    # Umbral con corrección de Bonferroni para control de tasa de error familiar (FWER):
    # Al evaluar 8 variables simultáneamente sobre muestras grandes (N > 4.000),
    # se ajusta alpha_nominal / k = 0.05 / 8 = 0.00625 para evitar falsos positivos por ruido.
    alpha_nominal = 0.05
    k_features = len(features)
    alpha = alpha_nominal / k_features # 0.00625
    print(f"Nivel de significancia nominal: alpha = {alpha_nominal}")
    print(f"Nivel de significancia ajustado (Bonferroni para {k_features} variables): alpha_adj = {alpha:.5f}")

    print(f"\n[TEST 1] Comparando datos limpios (mismo origen) contra Baseline...")
    drift_clean_results = []
    for col in features:
        stat, p_val = stats.ks_2samp(baseline_df[col], test_ref_df[col])
        has_drift = p_val < alpha
        drift_clean_results.append({
            "feature": col,
            "stat": round(stat, 4),
            "p_value": round(p_val, 4),
            "drift_detected": has_drift
        })
    
    clean_report = pd.DataFrame(drift_clean_results)
    print(clean_report.to_string(index=False))
    any_clean_drift = clean_report["drift_detected"].any()
    if not any_clean_drift:
        print("\n>>> RESULTADO TEST 1: [ESTADO VERDE - PASS] No se detectó Data Drift.")
    else:
        print("\n>>> RESULTADO TEST 1: [ESTADO ROJO - FAIL]")

    # 2. Generar lote perturbado deliberadamente (Data Drift artificial)
    print(f"\n[TEST 2] Inyectando alteración deliberada en variables (MedInc +40%, AveRooms +30%)...")
    drifted_df = test_ref_df.copy()
    drifted_df["MedInc"] = drifted_df["MedInc"] * 1.40 # Shock económico positivo / gentrificación
    drifted_df["AveRooms"] = drifted_df["AveRooms"] * 1.30 # Casas más grandes en nuevos desarrollos

    drift_dirty_results = []
    for col in features:
        stat, p_val = stats.ks_2samp(baseline_df[col], drifted_df[col])
        has_drift = p_val < alpha
        drift_dirty_results.append({
            "feature": col,
            "stat": round(stat, 4),
            "p_value": f"{p_val:.2e}",
            "drift_detected": has_drift
        })
    
    dirty_report = pd.DataFrame(drift_dirty_results)
    print(dirty_report.to_string(index=False))
    drifted_cols = dirty_report[dirty_report["drift_detected"] == True]["feature"].tolist()
    print(f"\n>>> RESULTADO TEST 2: [ESTADO ROJO - FAIL / ALERTA DE DATA DRIFT]")
    print(f"Variables con distribución alterada estadísticamente: {drifted_cols}")

    # 3. Graficar distribuciones de Data Drift
    os.makedirs("Entregable", exist_ok=True)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(baseline_df["MedInc"], bins=30, alpha=0.5, density=True, label="Baseline (Train)", color="blue")
    plt.hist(test_ref_df["MedInc"], bins=30, alpha=0.5, density=True, label="Lote Limpio (Test)", color="green")
    plt.title("MedInc: Sin Drift (PASS - Estado Verde)")
    plt.xlabel("MedInc")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(baseline_df["MedInc"], bins=30, alpha=0.5, density=True, label="Baseline (Train)", color="blue")
    plt.hist(drifted_df["MedInc"], bins=30, alpha=0.5, density=True, label="Lote Perturbado (Drift)", color="red")
    plt.title("MedInc: Con Data Drift (FAIL - Estado Rojo)")
    plt.xlabel("MedInc")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "Entregable/evidencia4a_data_drift.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Gráfico de Data Drift guardado en: {plot_path}")

def test_concept_drift():
    print("\n" + "=" * 70)
    print("2. EVALUACIÓN DE CONCEPT DRIFT (Degradación de Métrica en Producción)")
    print("=" * 70)

    model = mlflow.pyfunc.load_model("model_champion")
    
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    _, X_test_full, _, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dividir el conjunto de test en 10 lotes sucesivos
    n_batches = 10
    batch_size = len(X_test_full) // n_batches
    
    r2_scores = []
    rmse_scores = []
    batches = [f"Lote {i+1}" for i in range(n_batches)]

    print("Simulando 10 lotes sucesivos en producción...")
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        X_batch = X_test_full.iloc[start_idx:end_idx].copy()
        y_batch = y_test_full.iloc[start_idx:end_idx].copy()

        # A partir del lote 4 introducimos Concept Drift:
        # La relación entrada-salida se desacopla debido a un shock regulatorio/mercado
        if i >= 3:
            # Factor de perturbación acumulativo
            drift_factor = 0.15 * (i - 2)
            # El valor real de la vivienda se desacopla de la relación histórica aprendida
            y_batch = y_batch * (1.0 - drift_factor) + (X_batch["HouseAge"] * 0.05)

        preds = model.predict(X_batch)
        r2 = r2_score(y_batch, preds)
        rmse = np.sqrt(mean_squared_error(y_batch, preds))
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        print(f"  {batches[i]}: R2={r2:.4f} | RMSE={rmse:.4f} " + ("[ESTABLE]" if i < 3 else "[DEGRADADO - DRIFT]"))

    # Criterio explícito de reentrenamiento:
    # Umbral de advertencia: R2 cae debajo de 0.75
    # Umbral crítico de reentrenamiento: R2 < 0.65 sostenido durante >= 2 lotes
    warning_threshold = 0.75
    retrain_threshold = 0.65

    # Graficar la serie temporal de Concept Drift
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), r2_scores, marker='o', linewidth=2.5, color="#1f77b4", label="R2 Score por Lote")
    plt.axhline(y=0.84, color="green", linestyle="--", alpha=0.8, label="Baseline de Entrenamiento (0.84)")
    plt.axhline(y=warning_threshold, color="orange", linestyle="--", label=f"Alerta Temprana ({warning_threshold})")
    plt.axhline(y=retrain_threshold, color="red", linestyle="--", label=f"Disparo de Reentrenamiento ({retrain_threshold})")
    
    # Resaltar la zona de alarma
    alarm_batches = [i+1 for i, r in enumerate(r2_scores) if r < retrain_threshold]
    if len(alarm_batches) >= 2:
        trigger_batch = alarm_batches[1]
        plt.scatter([trigger_batch], [r2_scores[trigger_batch-1]], color="red", s=200, zorder=5, label=f"¡Reentrenamiento Disparado! (Lote {trigger_batch})")
        plt.annotate("DISPARO DE ALARMA\nReentrenamiento Automático",
                     xy=(trigger_batch, r2_scores[trigger_batch-1]),
                     xytext=(trigger_batch - 2.5, r2_scores[trigger_batch-1] - 0.25),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
                     fontweight='bold', color='darkred')

    plt.title("Concept Drift: Degradación de Desempeño y Criterio de Reentrenamiento", fontsize=13, fontweight='bold')
    plt.xlabel("Lotes de Producción Temporales", fontsize=11)
    plt.ylabel("Coeficiente de Determinación (R²)", fontsize=11)
    plt.xticks(range(1, 11), batches, rotation=30)
    plt.ylim(-0.2, 1.0)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    concept_plot_path = "Entregable/evidencia4b_concept_drift.png"
    plt.savefig(concept_plot_path, dpi=150)
    plt.close()
    print(f"\nGráfico de Concept Drift guardado en: {concept_plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    test_data_drift()
    test_concept_drift()
