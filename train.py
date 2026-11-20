"""
train.py
--------
Assignment 8: Monitor a Machine Learning Model using Prometheus and Grafana
Student: Siddhant Udavant (Roll No: 20, PRN: 12414447, Branch: AI&DS, Div: F)

This script trains a Random Forest Classifier on the Iris flower dataset,
evaluates its performance, and serializes both the trained model and
baseline feature statistics for production drift monitoring.
"""

import json
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_and_export():
    print("=" * 60)
    print("Training Machine Learning Model for MLOps Monitoring Pipeline")
    print("=" * 60)

    # 1. Load dataset
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()
    
    # Standardize column names to valid identifiers
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X.columns = feature_names
    class_names = list(iris.target_names)
    
    print(f"Dataset: Iris Classification ({len(X)} samples, 4 features)")
    print(f"Target classes: {class_names}")

    # 2. Train-test split (80% train, 20% test, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    # 3. Model training
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train.values, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Compute baseline statistics for feature drift monitoring
    stats = {
        "features": feature_names,
        "classes": class_names,
        "mean": [float(X_train[col].mean()) for col in feature_names],
        "std": [float(X_train[col].std()) for col in feature_names],
        "min": [float(X_train[col].min()) for col in feature_names],
        "max": [float(X_train[col].max()) for col in feature_names],
        "sample_count": int(len(X_train)),
        "accuracy": float(accuracy),
        "model_type": "RandomForestClassifier",
        "version": "1.0.0"
    }

    # 6. Save model artifacts
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "model.joblib")
    stats_path = os.path.join("model", "training_stats.json")

    joblib.dump(model, model_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print(f"\nTrained model successfully serialized to: {model_path}")
    print(f"Baseline feature statistics exported to: {stats_path}")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export()

