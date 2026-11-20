# Assignment 8: Monitor model using Prometheus and Grafana

**Course:** Machine Learning Operations (MLOps) / Machine Learning and AI Optimization  
**Academic Year:** Third Year B.Tech (AI & DS), Semester 5  
**Student Name:** Siddhant Udavant  
**Roll No:** 20 | **PRN:** 12414447 | **Div:** F  

---

## 1. Executive Summary

This project implements an end-to-end production monitoring pipeline for a Machine Learning application serving predictions via a FastAPI REST API. The system exposes real-time operational and statistical metrics scraped periodically by **Prometheus**, visualised dynamically through an interactive **Grafana** dashboard, and monitored for automated alerts (outages, error spikes, latency degradation, data drift, and accuracy drops) routed through **Alertmanager**.

```
+--------------------+        HTTP POST /predict        +----------------------------+
|  Client / Traffic  | -------------------------------> |  FastAPI ML Service        |
|  (simulate.py)     | <------------------------------- |  (Iris Classifier Model)   |
+--------------------+      Prediction & Latency        +--------------+-------------+
                                                                       | Exposes /metrics
                                                                       v
+--------------------+         PromQL Queries           +----------------------------+
|  Grafana           | <------------------------------- |  Prometheus TSDB           |
|  Dashboard         |                                  |  (Scrapes every 5 seconds) |
+--------------------+                                  +--------------+-------------+
                                                                       | Evaluates Rules
                                                                       v
                                                        +----------------------------+
                                                        |  Alertmanager              |
                                                        |  (Deduplication & Alerts)  |
                                                        +----------------------------+
```

---

## 2. Project Directory Structure

```
Assignment 8/
├── model/
│   ├── model.joblib                     # Trained Random Forest classifier
│   └── training_stats.json              # Baseline feature distributions for drift detection
├── prometheus/
│   ├── prometheus.yml                   # Prometheus scrape configuration & jobs
│   └── alert_rules.yml                  # Alerting rules (ServiceDown, Drift, ErrorRate, Latency)
├── alertmanager/
│   └── alertmanager.yml                 # Alertmanager notification routing configuration
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml   # Automatic Prometheus data source setup
│   │   └── dashboards/dashboards.yml    # Automatic dashboard provider setup
│   └── dashboards/
│       └── ml_model_monitoring.json     # 10-panel Grafana monitoring dashboard JSON
├── app.py                               # Instrumented FastAPI service
├── train.py                             # Model training and baseline extraction
├── simulate.py                          # Traffic simulation (normal, drift, errors, mixed)
├── run_experiments.py                   # Automated experiment harness and plot generator
├── create_dashboard.py                  # Dashboard JSON builder
├── Dockerfile                           # Container definition for ML service
├── docker-compose.yml                   # Multi-container orchestration stack
├── requirements.txt                     # Python dependencies
└── README.md                            # Documentation and operational manual
```

---

## 3. Metrics Instrumentation Design

The service exposes metrics via `http://localhost:8000/metrics` adhering to low-cardinality label standards:

| Metric Name | Type | Labels | Description / Purpose |
| :--- | :--- | :--- | :--- |
| `ml_prediction_requests_total` | Counter | `method`, `endpoint`, `status` | Total prediction requests received |
| `ml_prediction_errors_total` | Counter | `endpoint`, `error_type` | Total failed prediction requests (HTTP 4xx / 5xx) |
| `ml_prediction_latency_seconds` | Histogram | `endpoint` | Distribution of prediction request latency |
| `ml_predictions_total` | Counter | `predicted_class` | Total predictions categorized by class (setosa, versicolor, virginica) |
| `model_feature_drift_zscore` | Gauge | `feature` | Rolling feature z-score relative to training baseline: $z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$ |
| `model_prediction_confidence` | Histogram | None | Distribution of highest output class probability |
| `model_feedback_total` | Counter | `outcome` (`correct`, `incorrect`) | Ground-truth feedback counts for online accuracy |
| `up` | Gauge | `job="ml_api"` | Service availability (1 = UP, 0 = DOWN) |

---

## 4. PromQL Queries for Grafana Panels

### A. Prediction Request Rate
```promql
sum(rate(ml_prediction_requests_total[1m]))
```

### B. Average Prediction Latency
```promql
sum(rate(ml_prediction_latency_seconds_sum[1m])) / sum(rate(ml_prediction_latency_seconds_count[1m]))
```

### C. 95th Percentile (P95) Prediction Latency
```promql
histogram_quantile(0.95, sum by (le) (rate(ml_prediction_latency_seconds_bucket[1m])))
```

### D. Prediction Error Rate (%)
```promql
100 * sum(rate(ml_prediction_errors_total[1m])) / clamp_min(sum(rate(ml_prediction_requests_total[1m])), 1e-9)
```

### E. Service Availability
```promql
up{job="ml_api"}
```

### F. Feature Data Drift (Z-Score)
```promql
model_feature_drift_zscore
```

---

## 5. Quickstart & Execution Guide

### Option 1: Local Execution (Fastest for testing)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train model and save baseline statistics:**
   ```bash
   python train.py
   ```

3. **Start the FastAPI model serving service:**
   ```bash
   python -m uvicorn app:app --host 127.0.0.1 --port 8000
   ```

4. **Verify endpoints:**
   - Root info: `http://localhost:8000/`
   - Healthcheck: `http://localhost:8000/health`
   - Prometheus metrics: `http://localhost:8000/metrics`
   - Interactive Swagger API docs: `http://localhost:8000/docs`

5. **Generate simulation traffic:**
   ```bash
   # Normal traffic
   python simulate.py normal --count 120

   # Induce feature data drift
   python simulate.py drift --count 120

   # Inject API error payloads
   python simulate.py errors --count 180 --delay 0.5

   # Send mixed 100-request batch
   python simulate.py batch --count 100
   ```

6. **Run complete automated experiment suite:**
   ```bash
   python run_experiments.py
   ```

---

### Option 2: Docker Compose Orchestration

1. **Build and start all services:**
   ```bash
   docker compose up -d --build
   ```

2. **Access services:**
   - **FastAPI Service:** `http://localhost:8000`
   - **Prometheus UI:** `http://localhost:9090` (Navigate to `Status -> Targets` to verify `ml_api` is UP)
   - **Grafana Dashboard:** `http://localhost:3000` (User: `admin` | Password: `admin`)
   - **Alertmanager UI:** `http://localhost:9093`

3. **Tear down stack when finished:**
   ```bash
   docker compose down
   ```

---

## 6. Test Scenarios

Start generating traffic after Prometheus completes its first successful scrape. Check **Alerts** in Prometheus and the active alerts in Alertmanager.

| Scenario | Trigger | Expected signal |
| --- | --- | --- |
| Routine traffic | `python simulate.py normal --count 120` | Successful predictions increase request and class counters. |
| Shifted inputs | `python simulate.py drift --count 120` | Feature z-scores change. `FeatureDriftDetected` fires if an absolute z-score stays above 3 for one minute. |
| Invalid payloads | `python simulate.py errors --count 180 --delay 0.5` | Invalid inputs return HTTP 422. `HighErrorRate` fires if the rolling one-minute error ratio stays above 5% for one minute. |
| Service outage | `docker compose stop model-service` | `up{job="ml_api"}` drops to 0. `ModelServiceDown` fires after 30 seconds. |
| Recovery | `docker compose start model-service` | The scrape target returns to 1 and the outage alert clears. Process-local metrics reset on restart. |

Latency reflects successful inference. Feedback-based accuracy is unavailable during windows with no submitted labels. Simulator feedback is synthetic. The default Alertmanager receiver supports local inspection and has no external notification integration.

