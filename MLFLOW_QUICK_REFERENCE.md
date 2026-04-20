# MLflow Quick Reference (Project Notes)

This file is a short guide for how we used MLflow in this project.

## 1) High-level workflow

1. Prepare data and train multiple models (Logistic Regression, Random Forest, XGBoost, XGBoost+SMOTE).
2. Log each model run to MLflow with:

- parameters
- metrics (accuracy, recall, f1)
- model artifact

3. Compare runs in MLflow UI and choose the best model by a target metric.
4. Register the selected model in Model Registry.
5. Promote/copy model to a production model name and assign alias (for example `champion`).

## 2) Start MLflow server/UI

Run from project root (`mlflow_dagshub_demo`):

```bash
source .venv/bin/activate
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Open UI:

```text
http://127.0.0.1:5000
```

## 3) Set tracking in notebook/script

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
experiment_name = "Anomaly Detection"

client = MlflowClient()
exp = client.get_experiment_by_name(experiment_name)
if exp is not None and exp.lifecycle_stage == "deleted":
    client.restore_experiment(exp.experiment_id)

mlflow.set_experiment(experiment_name)
```

## 4) Log runs for different models

```python
for i, (model_name, params, model, train_set, test_set) in enumerate(models):
    X_train, y_train = train_set
    X_test, y_test = test_set

    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": report["accuracy"],
            "recall_class_1": report["1"]["recall"],
            "recall_class_0": report["0"]["recall"],
            "f1_score_macro": report["macro avg"]["f1-score"],
        })

        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")
```

## 5) Compare models (UI + programmatic)

UI method:

1. Open experiment page in MLflow.
2. Sort by `f1_score_macro` (or your target metric).
3. Pick best run.

Programmatic method:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
exp = client.get_experiment_by_name("Anomaly Detection")

best_run = client.search_runs(
    [exp.experiment_id],
    order_by=["metrics.f1_score_macro DESC"],
    max_results=1,
)[0]

best_run_id = best_run.info.run_id
print("Best run:", best_run_id)
print("Best f1_score_macro:", best_run.data.metrics.get("f1_score_macro"))
```

## 6) Register best model

```python
registered_model_name = "XGB-Smote"
model_uri = f"runs:/{best_run_id}/model"

registered = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
model_version = registered.version

print(f"Registered {registered_model_name} version {model_version}")
```

## 7) Promote to production model and alias

```python
production_model_name = "anomaly-detection-prod"
src_model_uri = f"models:/{registered_model_name}/{model_version}"

client = mlflow.MlflowClient()
copied = client.copy_model_version(src_model_uri=src_model_uri, dst_name=production_model_name)

client.set_registered_model_alias(
    name=production_model_name,
    alias="champion",
    version=copied.version,
)

print("Production alias set:", production_model_name, "@champion")
```

## 8) Load production model for inference

```python
prod_model_uri = "models:/anomaly-detection-prod@champion"
loaded_model = mlflow.xgboost.load_model(prod_model_uri)
y_pred = loaded_model.predict(X_test)
```

## 9) Common issues checklist

- Use `127.0.0.1` consistently (avoid mixing with `localhost`).
- Set `mlflow.set_tracking_uri(...)` before `mlflow.set_experiment(...)`.
- If experiment exists but is deleted, restore it before use.
- Ensure notebook kernel uses the same `.venv` where packages are installed.
