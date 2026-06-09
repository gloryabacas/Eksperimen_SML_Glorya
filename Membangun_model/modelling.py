
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()
# =========================
# 1. LOAD DATASET
# =========================
# pakai hasil preprocessing kamu
df = pd.read_csv("dataset_preprocessing.csv")

# pastikan kolom target sesuai dataset kamu
target_col = "target"

X = df.drop(columns=[target_col])
y = df[target_col]

# =========================
# 2. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. MLflow SETUP
# =========================
mlflow.set_experiment("heart_disease_experiment")

with mlflow.start_run():

    # =========================
    # 4. MODEL TRAINING
    # =========================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # =========================
    # 5. PREDICTION
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # 6. METRICS
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # =========================
    # 7. LOGGING KE MLFLOW
    # =========================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("===== MODEL EVALUATION =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
