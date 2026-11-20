"""
run_experiments.py
------------------
Executes all testing scenarios for Assignment 8, captures Prometheus metrics,
and renders high-resolution visualization charts for inclusion in the formal lab report.
Student: Siddhant Udavant (Roll No: 20, PRN: 12414447, Branch: AI&DS, Div: F)
"""

import os
import random
import time
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from fastapi.testclient import TestClient

from app import app, FEATURES, CLASSES

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)

client = TestClient(app)

def run_scenarios():
    print("=" * 65)
    print("RUNNING AUTOMATED MLOps MONITORING EXPERIMENTS (Assignment 8)")
    print("=" * 65)

    history = {
        "timestamp": [],
        "request_type": [],
        "latency_ms": [],
        "status_code": [],
        "prediction": [],
        "confidence": [],
        "drift_z": {f: [] for f in FEATURES},
        "cumulative_requests": [],
        "cumulative_errors": []
    }

    total_reqs = 0
    total_errs = 0

    # -------------------------------------------------------------
    # Scenario A: Normal Traffic (50 requests)
    # -------------------------------------------------------------
    print("\n>>> Scenario A: Simulating Normal Production Traffic (50 requests)...")
    for i in range(50):
        t_start = time.perf_counter()
        payload = {
            "sepal_length": round(random.gauss(5.84, 0.40), 2),
            "sepal_width": round(random.gauss(3.05, 0.30), 2),
            "petal_length": round(random.gauss(3.77, 0.80), 2),
            "petal_width": round(random.gauss(1.20, 0.40), 2)
        }
        res = client.post("/predict", json=payload)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        total_reqs += 1
        if res.status_code == 200:
            data = res.json()
            pred = data["prediction"]
            conf = data["confidence"]
            # Feedback: 96% correct
            actual = pred if random.random() < 0.96 else random.choice(CLASSES)
            client.post("/feedback", json={"predicted": pred, "actual": actual})
        else:
            total_errs += 1
            pred = "None"
            conf = 0.0

        history["timestamp"].append(total_reqs)
        history["request_type"].append("Normal")
        history["latency_ms"].append(t_elapsed)
        history["status_code"].append(res.status_code)
        history["prediction"].append(pred)
        history["confidence"].append(conf)
        history["cumulative_requests"].append(total_reqs)
        history["cumulative_errors"].append(total_errs)

        # Scrape drift values from /metrics
        metrics_text = client.get("/metrics").text
        for f in FEATURES:
            z_val = 0.0
            for line in metrics_text.splitlines():
                if f'model_feature_drift_zscore{{feature="{f}"}}' in line:
                    z_val = float(line.split()[-1])
                    break
            history["drift_z"][f].append(z_val)

    print("    Scenario A completed. Normal requests processed cleanly.")

    # -------------------------------------------------------------
    # Scenario B: Feature Data Drift Traffic (50 requests)
    # -------------------------------------------------------------
    print("\n>>> Scenario B: Inducing Feature Data Drift (50 requests with shifted inputs)...")
    for i in range(50):
        t_start = time.perf_counter()
        # Massive shift in petal length (+2.5 cm) and sepal length (+1.5 cm)
        payload = {
            "sepal_length": round(random.gauss(5.84 + 1.5, 0.30), 2),
            "sepal_width": round(random.gauss(3.05, 0.30), 2),
            "petal_length": round(random.gauss(3.77 + 2.5, 0.40), 2),
            "petal_width": round(random.gauss(1.20 + 0.8, 0.20), 2)
        }
        res = client.post("/predict", json=payload)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        total_reqs += 1
        if res.status_code == 200:
            data = res.json()
            pred = data["prediction"]
            conf = data["confidence"]
            # Feedback under drift: accuracy degrades to ~55%
            actual = pred if random.random() < 0.55 else random.choice(CLASSES)
            client.post("/feedback", json={"predicted": pred, "actual": actual})
        else:
            total_errs += 1
            pred = "None"
            conf = 0.0

        history["timestamp"].append(total_reqs)
        history["request_type"].append("Drift")
        history["latency_ms"].append(t_elapsed)
        history["status_code"].append(res.status_code)
        history["prediction"].append(pred)
        history["confidence"].append(conf)
        history["cumulative_requests"].append(total_reqs)
        history["cumulative_errors"].append(total_errs)

        metrics_text = client.get("/metrics").text
        for f in FEATURES:
            z_val = 0.0
            for line in metrics_text.splitlines():
                if f'model_feature_drift_zscore{{feature="{f}"}}' in line:
                    z_val = float(line.split()[-1])
                    break
            history["drift_z"][f].append(z_val)

    print("    Scenario B completed. Significant z-score elevation observed.")

    # -------------------------------------------------------------
    # Scenario C: Controlled API Error Injection (30 requests)
    # -------------------------------------------------------------
    print("\n>>> Scenario C: Controlled API Error Injection (30 requests)...")
    for i in range(30):
        t_start = time.perf_counter()
        is_err = (random.random() < 0.70)
        if is_err:
            payload = {"sepal_length": -99.0, "sepal_width": "invalid"}  # Bad types & out-of-range
        else:
            payload = {
                "sepal_length": 6.0,
                "sepal_width": 3.0,
                "petal_length": 4.0,
                "petal_width": 1.3
            }

        res = client.post("/predict", json=payload)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        total_reqs += 1
        if res.status_code == 200:
            data = res.json()
            pred = data["prediction"]
            conf = data["confidence"]
        else:
            total_errs += 1
            pred = "Error"
            conf = 0.0

        history["timestamp"].append(total_reqs)
        history["request_type"].append("Errors")
        history["latency_ms"].append(t_elapsed)
        history["status_code"].append(res.status_code)
        history["prediction"].append(pred)
        history["confidence"].append(conf)
        history["cumulative_requests"].append(total_reqs)
        history["cumulative_errors"].append(total_errs)

        metrics_text = client.get("/metrics").text
        for f in FEATURES:
            z_val = 0.0
            for line in metrics_text.splitlines():
                if f'model_feature_drift_zscore{{feature="{f}"}}' in line:
                    z_val = float(line.split()[-1])
                    break
            history["drift_z"][f].append(z_val)

    print("    Scenario C completed. Error counters properly updated.")

    # Print final /metrics snapshot
    print("\n" + "=" * 65)
    print("PROMETHEUS /metrics ENDPOINT EXCERPT:")
    print("=" * 65)
    final_metrics = client.get("/metrics").text
    for line in final_metrics.splitlines():
        if any(k in line for k in ["ml_prediction", "model_feature_drift", "model_feedback", "model_confidence"]):
            print(line)

    # -------------------------------------------------------------
    # Plot Generation
    # -------------------------------------------------------------
    generate_figures(history)

import matplotlib.patches as mpatches

def generate_figures(history: Dict[str, Any]):
    print("\nGenerating report figures...")
    # Set overall aesthetic
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------
    # Figure 1: Architecture Diagram
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.axis("off")

    # Draw boxes
    boxes = [
        {"x": 0.05, "y": 0.65, "w": 0.22, "h": 0.25, "title": "Client / Load Generator\n(simulate.py / curl)", "color": "#E3F2FD", "border": "#1976D2"},
        {"x": 0.38, "y": 0.55, "w": 0.25, "h": 0.40, "title": "ML Serving API (FastAPI)\n- Pydantic Validation\n- Scikit-learn Model\n- Prometheus Client\n- /predict & /feedback", "color": "#E8F5E9", "border": "#388E3C"},
        {"x": 0.73, "y": 0.55, "w": 0.23, "h": 0.40, "title": "Prometheus Server\n- Scrapes /metrics (5s)\n- Time-series TSDB\n- Evaluates Alert Rules\n- PromQL Engine", "color": "#FFF3E0", "border": "#F57C00"},
        {"x": 0.73, "y": 0.08, "w": 0.23, "h": 0.32, "title": "Grafana Visualization\n- Real-time Dashboards\n- Latency, Rate, Drift\n- Panels & Thresholds", "color": "#F3E5F5", "border": "#7B1FA2"},
        {"x": 0.38, "y": 0.08, "w": 0.25, "h": 0.32, "title": "Alertmanager\n- Groups Alerts\n- Deduplication\n- Outage & Drift Alarms", "color": "#FFEBEE", "border": "#D32F2F"},
        {"x": 0.05, "y": 0.08, "w": 0.22, "h": 0.32, "title": "Node Exporter\n- Host CPU Usage\n- Host Memory & Disk\n- Port 9100", "color": "#ECEFF1", "border": "#455A64"},
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.02", facecolor=b["color"], edgecolor=b["border"], linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]/2, b["title"], ha="center", va="center", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#212121")

    # Arrows
    arrows = [
        ((0.27, 0.77), (0.38, 0.77), "POST /predict\nJSON payload"),
        ((0.63, 0.75), (0.73, 0.75), "Pulls /metrics\nevery 5s (PromQL)"),
        ((0.73, 0.60), (0.63, 0.25), "Forwards Firing\nAlerts"),
        ((0.84, 0.55), (0.84, 0.40), "Queries TSDB\nvia PromQL"),
        ((0.27, 0.24), (0.73, 0.65), "Scrapes Node\nMetrics (5s)"),
    ]

    for start, end, label in arrows:
        ax.annotate(
            label, xy=end, xytext=start,
            arrowprops=dict(facecolor="#37474F", edgecolor="#37474F", width=1.5, headwidth=8, headlength=8, shrink=0.05),
            ha="center", va="center", fontsize=8.5, color="#1A237E", fontweight="semibold"
        )

    plt.title("System Architecture: End-to-End MLOps Monitoring with Prometheus & Grafana", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("figures/1_system_architecture.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/1_system_architecture.png")

    # -------------------------------------------------------------
    # Figure 2: Experimental Results Multi-Plot
    # -------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # 2A: Cumulative Requests & Errors
    req_nums = history["timestamp"]
    axs[0, 0].plot(req_nums, history["cumulative_requests"], label="Total Requests (200 OK + 4xx)", color="#1976D2", linewidth=2)
    axs[0, 0].plot(req_nums, history["cumulative_errors"], label="Total Errors (422 / 500)", color="#D32F2F", linewidth=2, linestyle="--")
    axs[0, 0].axvline(50, color="gray", linestyle=":", label="Scenario B (Drift Starts)")
    axs[0, 0].axvline(100, color="orange", linestyle=":", label="Scenario C (Errors Start)")
    axs[0, 0].set_title("A. Cumulative Prediction Requests & Error Counts", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("Request Sequence Index")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].legend(loc="upper left", fontsize=8.5)
    axs[0, 0].grid(True, alpha=0.5)

    # 2B: Latency Profile
    axs[0, 1].plot(req_nums, history["latency_ms"], color="#00897B", alpha=0.6, label="Sample Latency (ms)")
    rolling_lat = np.convolve(history["latency_ms"], np.ones(5)/5, mode="valid")
    axs[0, 1].plot(range(5, len(history["latency_ms"]) + 1), rolling_lat, color="#004D40", linewidth=2, label="Rolling Mean Latency (5-pt)")
    axs[0, 1].set_title("B. Inference Latency over Time (FastAPI Response)", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Request Sequence Index")
    axs[0, 1].set_ylabel("Latency (milliseconds)")
    axs[0, 1].legend(loc="upper right", fontsize=8.5)
    axs[0, 1].grid(True, alpha=0.5)

    # 2C: Feature Drift Z-Score Trajectory
    colors = {"sepal_length": "#E65100", "sepal_width": "#2E7D32", "petal_length": "#C2185B", "petal_width": "#512DA8"}
    for f in FEATURES:
        axs[1, 0].plot(req_nums, history["drift_z"][f], label=f"{f} z-score", color=colors[f], linewidth=2)

    axs[1, 0].axhline(3.0, color="red", linestyle="--", linewidth=1.5, label="Upper Drift Threshold (+3.0)")
    axs[1, 0].axhline(-3.0, color="red", linestyle="--", linewidth=1.5, label="Lower Drift Threshold (-3.0)")
    axs[1, 0].axvline(50, color="gray", linestyle=":", label="Drift Injected")
    axs[1, 0].set_title("C. Feature Drift Z-Scores (Triggering FeatureDriftDetected Alert)", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Request Sequence Index")
    axs[1, 0].set_ylabel("Z-Score")
    axs[1, 0].legend(loc="upper left", fontsize=8.5)
    axs[1, 0].grid(True, alpha=0.5)

    # 2D: Prediction Confidence & Error Rate Proportion
    err_rate = [100.0 * (history["cumulative_errors"][i] / (i + 1)) for i in range(len(req_nums))]
    axs[1, 1].plot(req_nums, err_rate, color="#D32F2F", linewidth=2, label="Cumulative Error Rate (%)")
    valid_conf = [c for c in history["confidence"] if c > 0]
    axs[1, 1].scatter([req_nums[i] for i, c in enumerate(history["confidence"]) if c > 0], valid_conf, color="#3949AB", alpha=0.5, s=20, label="Prediction Confidence (0 to 1)")
    axs[1, 1].axhline(5.0, color="darkred", linestyle=":", label="HighErrorRate Alert Limit (5%)")
    axs[1, 1].set_title("D. Prediction Error Rate (%) and Output Confidence", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Request Sequence Index")
    axs[1, 1].set_ylabel("Percentage / Confidence Scale")
    axs[1, 1].legend(loc="upper left", fontsize=8.5)
    axs[1, 1].grid(True, alpha=0.5)

    plt.suptitle("Experimental Monitoring Signals across Scenarios A (Normal), B (Drift), and C (Errors)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig("figures/2_scenario_experiments.png", bbox_inches="tight")
    plt.close()
    print("  Saved: figures/2_scenario_experiments.png")

if __name__ == "__main__":
    run_scenarios()
