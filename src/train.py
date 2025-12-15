from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data_processing import build_preprocessing_pipeline


def load_training_data(
    file_name: str = "train_customers_with_target.csv",
) -> pd.DataFrame:
    """Load the processed training data with is_high_risk target."""
    data_path = (
        Path(__file__).resolve().parents[1] / "data" / "processed" / file_name
    )
    print("Loading training data from:", data_path)
    df = pd.read_csv(data_path)
    return df


def evaluate(y_true, y_pred, y_proba):
    """Compute standard binary classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    return acc, prec, rec, f1, auc


def main() -> None:
    df = load_training_data()

    if "is_high_risk" not in df.columns:
        raise ValueError("Target column 'is_high_risk' not found")

    X_df = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessing_pipeline(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    with mlflow.start_run(run_name="LogisticRegression"):
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)

        y_pred_lr = log_reg.predict(X_test)
        y_proba_lr = log_reg.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, auc = evaluate(y_test, y_pred_lr, y_proba_lr)
        print(
            "LogisticRegression -> "
            f"acc: {acc:.3f}, prec: {prec:.3f}, "
            f"rec: {rec:.3f}, f1: {f1:.3f}, auc: {auc:.3f}",
        )

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(log_reg, artifact_path="model")

    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        y_proba_rf = rf.predict_proba(X_test)[:, 1]

        acc, prec, rec, f1, auc = evaluate(y_test, y_pred_rf, y_proba_rf)
        print(
            "RandomForest -> "
            f"acc: {acc:.3f}, prec: {prec:.3f}, "
            f"rec: {rec:.3f}, f1: {f1:.3f}, auc: {auc:.3f}",
        )

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(rf, artifact_path="model")


if __name__ == "__main__":
    main()
