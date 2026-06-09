import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn

# =====================================
# AUTLOG MLFLOW
# =====================================
mlflow.sklearn.autolog()

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop("target", axis=1)
y = df["target"]

# =====================================
# SPLIT DATA
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# MLFLOW EXPERIMENT
# =====================================
mlflow.set_experiment("heart_disease_experiment")

with mlflow.start_run():

    # =====================================
    # TRAINING
    # =====================================
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    # =====================================
    # PREDIKSI
    # =====================================
    y_pred = model.predict(X_test)

    # =====================================
    # EVALUASI
    # =====================================
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # =====================================
    # LOG MANUAL
    # =====================================
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

print("\nTraining selesai.")
print("MLflow berhasil menyimpan metrics dan model.")
