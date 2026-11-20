"""
create_dashboard.py
Generates the Grafana Dashboard JSON file for ML Model Monitoring.
"""

import json
import os

os.makedirs("grafana/dashboards", exist_ok=True)

dashboard = {
    "annotations": {"list": []},
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 1,
    "id": None,
    "links": [],
    "liveNow": True,
    "panels": [
        {
            "id": 1,
            "gridPos": {"h": 4, "w": 4, "x": 0, "y": 0},
            "type": "stat",
            "title": "Service Availability",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {"expr": 'up{job="ml_api"}', "legendFormat": "ML Service Status", "refId": "A"}
            ],
            "fieldConfig": {
                "defaults": {
                    "mappings": [
                        {
                            "type": "value",
                            "options": {
                                "0": {"text": "DOWN", "color": "red"},
                                "1": {"text": "UP", "color": "green"}
                            }
                        }
                    ],
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "green", "value": 1}
                        ]
                    }
                }
            }
        },
        {
            "id": 2,
            "gridPos": {"h": 4, "w": 5, "x": 4, "y": 0},
            "type": "stat",
            "title": "Total Prediction Requests",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {"expr": "sum(ml_prediction_requests_total)", "legendFormat": "Total Requests", "refId": "A"}
            ],
            "fieldConfig": {
                "defaults": {
                    "color": {"mode": "palette-classic"},
                    "unit": "short"
                }
            }
        },
        {
            "id": 3,
            "gridPos": {"h": 4, "w": 5, "x": 9, "y": 0},
            "type": "stat",
            "title": "Prediction Error Rate (%)",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": '100 * sum(rate(ml_prediction_errors_total[1m])) / clamp_min(sum(rate(ml_prediction_requests_total[1m])), 1e-9)',
                    "legendFormat": "Error Rate %",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "yellow", "value": 2.0},
                            {"color": "red", "value": 5.0}
                        ]
                    }
                }
            }
        },
        {
            "id": 4,
            "gridPos": {"h": 4, "w": 5, "x": 14, "y": 0},
            "type": "gauge",
            "title": "Average Prediction Confidence",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": "sum(rate(model_prediction_confidence_sum[5m])) / sum(rate(model_prediction_confidence_count[5m]))",
                    "legendFormat": "Confidence",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "unit": "percentunit",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.70},
                            {"color": "green", "value": 0.85}
                        ]
                    }
                }
            }
        },
        {
            "id": 5,
            "gridPos": {"h": 4, "w": 5, "x": 19, "y": 0},
            "type": "gauge",
            "title": "Rolling Accuracy (Feedback)",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": 'sum(increase(model_feedback_total{outcome="correct"}[5m])) / sum(increase(model_feedback_total[5m]))',
                    "legendFormat": "Accuracy",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "unit": "percentunit",
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "yellow", "value": 0.80},
                            {"color": "green", "value": 0.90}
                        ]
                    }
                }
            }
        },
        {
            "id": 6,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
            "type": "timeseries",
            "title": "Prediction Request Rate (req/s)",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {"expr": "sum(rate(ml_prediction_requests_total[1m]))", "legendFormat": "Request Rate", "refId": "A"}
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {"lineInterpolation": "smooth", "fillOpacity": 15},
                    "unit": "reqps"
                }
            }
        },
        {
            "id": 7,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
            "type": "timeseries",
            "title": "Prediction Latency: Average vs P95 Latency",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": "sum(rate(ml_prediction_latency_seconds_sum[1m])) / sum(rate(ml_prediction_latency_seconds_count[1m]))",
                    "legendFormat": "Average Latency",
                    "refId": "A"
                },
                {
                    "expr": 'histogram_quantile(0.95, sum by (le) (rate(ml_prediction_latency_seconds_bucket[1m])))',
                    "legendFormat": "P95 Latency",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {"lineInterpolation": "smooth"},
                    "unit": "s"
                }
            }
        },
        {
            "id": 8,
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
            "type": "timeseries",
            "title": "Predictions by Class (req/s)",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": "sum by (predicted_class) (rate(ml_predictions_total[1m]))",
                    "legendFormat": "{{predicted_class}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {"stacking": {"mode": "normal"}, "fillOpacity": 25},
                    "unit": "reqps"
                }
            }
        },
        {
            "id": 9,
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
            "type": "timeseries",
            "title": "Feature Data Drift Z-Scores (Threshold ±3.0)",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": "model_feature_drift_zscore",
                    "legendFormat": "{{feature}}",
                    "refId": "A"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "custom": {"thresholdsStyle": {"mode": "line"}},
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": -3.0},
                            {"color": "green", "value": -2.99},
                            {"color": "red", "value": 3.0}
                        ]
                    }
                }
            }
        },
        {
            "id": 10,
            "gridPos": {"h": 6, "w": 24, "x": 0, "y": 20},
            "type": "timeseries",
            "title": "Host Infrastructure: CPU & Memory Utilization",
            "datasource": {"type": "prometheus", "uid": "Prometheus"},
            "targets": [
                {
                    "expr": '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
                    "legendFormat": "Host CPU Usage (%)",
                    "refId": "A"
                },
                {
                    "expr": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
                    "legendFormat": "Memory Usage (%)",
                    "refId": "B"
                }
            ],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100
                }
            }
        }
    ],
    "refresh": "5s",
    "schemaVersion": 38,
    "style": "dark",
    "tags": ["mlops", "fastapi", "prometheus", "monitoring"],
    "time": {"from": "now-15m", "to": "now"},
    "timepicker": {"refresh_intervals": ["5s", "10s", "30s", "1m"]},
    "timezone": "browser",
    "title": "ML Model Monitoring - Iris Classifier",
    "uid": "ml-model-monitoring-iris",
    "version": 1
}

with open("grafana/dashboards/ml_model_monitoring.json", "w", encoding="utf-8") as f:
    json.dump(dashboard, f, indent=2)

print("Dashboard successfully saved at: grafana/dashboards/ml_model_monitoring.json")

