import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from mlflow import MlflowClient

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import os

# Confianza en tipos de sklearn para skops en MLflow 3.x
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

def main():
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    db_path = os.path.abspath("mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment_name = "california-housing-pricing"
    mlflow.set_experiment(experiment_name)

    print("Cargando dataset California Housing...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Partición Train / Test reproducible (semilla fija = 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Guardar baselines para la Fase 4 (Data Drift)
    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/baseline_train.csv", index=False)
    X_test.to_csv("data/reference_test.csv", index=False)
    print("Baseline guardado en data/baseline_train.csv")

    signature = infer_signature(X_train, y_train)

    experiments = [
        {
            "run_name": "Run-1-RandomForest-Base",
            "type": "rf",
            "params": {"n_estimators": 50, "max_depth": 10, "random_state": 42},
            "model_builder": lambda p: RandomForestRegressor(**p)
        },
        {
            "run_name": "Run-2-RandomForest-Deep",
            "type": "rf",
            "params": {"n_estimators": 100, "max_depth": 18, "random_state": 42},
            "model_builder": lambda p: RandomForestRegressor(**p)
        },
        {
            "run_name": "Run-3-GradientBoosting-Light",
            "type": "gb",
            "params": {"n_estimators": 80, "learning_rate": 0.05, "max_depth": 4, "random_state": 42},
            "model_builder": lambda p: GradientBoostingRegressor(**p)
        },
        {
            "run_name": "Run-4-GradientBoosting-Strong",
            "type": "gb",
            "params": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 6, "random_state": 42},
            "model_builder": lambda p: GradientBoostingRegressor(**p)
        },
        {
            "run_name": "Run-5-XGBoost-GPU-Optimized",
            "type": "xgb",
            "params": {
                "n_estimators": 250,
                "learning_rate": 0.08,
                "max_depth": 7,
                "tree_method": "hist",
                "device": "cuda",
                "random_state": 42
            },
            "model_builder": lambda p: xgb.XGBRegressor(**p)
        }
    ]

    model_registry_name = "modelo-inmobiliario-prod"
    best_r2 = -float("inf")
    best_run_id = None
    best_version = None

    print("\nIniciando los 5 experimentos en MLflow...")
    for exp in experiments:
        with mlflow.start_run(run_name=exp["run_name"]) as run:
            run_id = run.info.run_id
            print(f"\n--- Ejecutando {exp['run_name']} (Run ID: {run_id}) ---")
            
            model = exp["model_builder"](exp["params"])
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            print(f"Resultados: RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f}")

            mlflow.log_params(exp["params"])
            mlflow.log_param("model_type", exp["type"])
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2_score": r2})

            if exp["type"] == "xgb":
                mlflow.xgboost.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_test.iloc[:5]
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_test.iloc[:5],
                    skops_trusted_types=TRUSTED_TYPES
                )

            model_uri = f"runs:/{run_id}/model"
            reg_version = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
            print(f"Modelo registrado en Registry '{model_registry_name}' v{reg_version.version}")

            if r2 > best_r2:
                best_r2 = r2
                best_run_id = run_id
                best_version = reg_version.version

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_registry_name,
        alias="champion",
        version=best_version
    )
    print("\n" + "="*70)
    print(f"PROCESO DE EXPERIMENTACIÓN COMPLETADO")
    print(f"Mejor modelo: Versión {best_version} (Run ID: {best_run_id})")
    print(f"Métrica R2: {best_r2:.4f}")
    print(f"Alias asignado: @champion en '{model_registry_name}'")
    print("="*70)

if __name__ == "__main__":
    main()
