"""
simulate.py
-----------
Traffic Generator and Monitoring Simulator.
Assignment 8: Monitor a Machine Learning Model using Prometheus and Grafana
Student: Siddhant Udavant (Roll No: 20, PRN: 12414447, Branch: AI&DS, Div: F)

Supported Modes:
1. normal  - Generates realistic Iris samples consistent with training distributions.
2. drift   - Introduces severe feature distribution drift (+2.0 petal, +1.0 sepal).
3. errors  - Injects malformed payloads to test error tracking and alert triggering.
4. mixed   - Interleaves normal traffic, periodic drift, and error bursts.
5. batch   - Sends a defined batch of requests for repeatable benchmark testing.
"""

import argparse
import random
import sys
import time
from typing import Dict, Any
import requests

DEFAULT_URL = "http://localhost:8000"
CLASSES = ["setosa", "versicolor", "virginica"]

def generate_sample(mode: str = "normal") -> Dict[str, float]:
    """Generates synthetic input feature payload based on operation mode."""
    # Base baseline parameters from Iris training set
    sepal_length = random.gauss(5.84, 0.84)
    sepal_width = random.gauss(3.05, 0.45)
    petal_length = random.gauss(3.77, 1.76)
    petal_width = random.gauss(1.20, 0.76)

    if mode == "drift":
        # Shift petal and sepal distributions significantly to trigger drift alert (|z| > 3.0)
        petal_length += 2.2
        sepal_length += 1.4
        petal_width += 0.8

    return {
        "sepal_length": round(max(0.5, sepal_length), 2),
        "sepal_width": round(max(0.5, sepal_width), 2),
        "petal_length": round(max(0.2, petal_length), 2),
        "petal_width": round(max(0.1, petal_width), 2)
    }

def generate_invalid_sample() -> Dict[str, Any]:
    """Generates various invalid payloads to test HTTP 422 validation handling."""
    error_type = random.choice(["missing_field", "bad_type", "negative_value", "extra_bad"])
    if error_type == "missing_field":
        return {"sepal_length": 5.1, "sepal_width": 3.5}
    elif error_type == "bad_type":
        return {"sepal_length": "invalid_string", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    elif error_type == "negative_value":
        return {"sepal_length": -3.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    else:
        return {"corrupt": True, "data": None}

def run_simulation(url: str, mode: str, count: int = 0, delay: float = 0.15):
    print("=" * 60)
    print(f"Starting Traffic Simulator: Mode = '{mode}' | Target = {url}")
    print(f"Count: {'Continuous (Press Ctrl+C to stop)' if count <= 0 else count} requests")
    print("=" * 60)

    total_requests = 0
    success_count = 0
    error_count = 0
    start_time = time.time()

    try:
        req_num = 0
        while True:
            req_num += 1
            if count > 0 and req_num > count:
                break

            current_mode = mode
            if mode == "mixed":
                rand = random.random()
                if rand < 0.15:
                    current_mode = "errors"
                elif rand < 0.40:
                    current_mode = "drift"
                else:
                    current_mode = "normal"

            if current_mode == "errors" or (mode == "errors" and random.random() < 0.8):
                payload = generate_invalid_sample()
            else:
                payload = generate_sample(mode=current_mode)

            # Send prediction request
            try:
                res = requests.post(f"{url}/predict", json=payload, timeout=5)
                total_requests += 1

                if res.status_code == 200:
                    success_count += 1
                    data = res.json()
                    pred = data["prediction"]
                    conf = data["confidence"]

                    # Simulate delayed ground-truth feedback
                    if random.random() < 0.35:
                        p_correct = 0.95 if current_mode == "normal" else 0.50
                        actual = pred if random.random() < p_correct else random.choice(CLASSES)
                        requests.post(
                            f"{url}/feedback",
                            json={"predicted": pred, "actual": actual},
                            timeout=2
                        )

                    print(f"[{req_num:04d}] Mode={current_mode:<6} -> 200 OK | Pred={pred:<10} | Conf={conf:.3f}")
                else:
                    error_count += 1
                    print(f"[{req_num:04d}] Mode={current_mode:<6} -> {res.status_code} Error: {res.text[:60]}")

            except requests.exceptions.RequestException as e:
                error_count += 1
                total_requests += 1
                print(f"[{req_num:04d}] Connection Error: {str(e)[:60]}")

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nSimulation halted by user.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Simulation Summary:")
    print(f"Total Requests: {total_requests}")
    print(f"Successful (200 OK): {success_count} ({(success_count/max(1, total_requests))*100:.1f}%)")
    print(f"Errors (4xx / 5xx): {error_count} ({(error_count/max(1, total_requests))*100:.1f}%)")
    print(f"Elapsed Time: {elapsed:.2f}s | Request Rate: {total_requests/max(0.1, elapsed):.2f} req/s")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate traffic for MLOps monitoring")
    parser.add_argument("mode", nargs="?", default="normal", choices=["normal", "drift", "errors", "mixed", "batch"])
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of prediction service")
    parser.add_argument("--count", type=int, default=0, help="Number of requests to send (0 = infinite)")
    parser.add_argument("--delay", type=float, default=0.10, help="Delay between requests in seconds")

    args = parser.parse_args()
    if args.mode == "batch":
        run_simulation(args.url, mode="mixed", count=args.count or 100, delay=args.delay)
    else:
        run_simulation(args.url, mode=args.mode, count=args.count, delay=args.delay)

