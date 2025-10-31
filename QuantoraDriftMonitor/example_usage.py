"""
Example usage script for Quantora API

Demonstrates how to:
1. Register a quantitative model
2. Upload performance metrics
3. Retrieve drift analysis
4. Get alerts
5. Generate PDF reports
"""

import requests
import time
import random
from datetime import datetime

BASE_URL = "http://localhost:5000"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def register_model():
    """Register a new quantitative model with baseline distribution."""
    print_section("1. Register Model")
    
    response = requests.post(f"{BASE_URL}/models", json={
        "name": f"momentum_strategy_{int(time.time())}",
        "description": "Long-short momentum equity strategy with mean reversion overlay",
        "model_type": "quantitative",
        "baseline_distribution": {
            "momentum_score": [0.1, 0.15, 0.12, 0.18, 0.14, 0.16, 0.11, 0.13],
            "volatility": [0.2, 0.22, 0.19, 0.21, 0.23, 0.20, 0.18, 0.21],
            "market_beta": [0.8, 0.85, 0.82, 0.87, 0.79, 0.84, 0.81, 0.83]
        }
    })
    
    if response.status_code == 200:
        model = response.json()
        print(f"✓ Model registered successfully!")
        print(f"  Model ID: {model['id']}")
        print(f"  Name: {model['name']}")
        print(f"  Type: {model['model_type']}")
        return model['id']
    else:
        print(f"✗ Error registering model: {response.status_code}")
        print(response.text)
        return None


def upload_metrics(model_id, num_days=30):
    """Upload performance metrics for multiple days."""
    print_section("2. Upload Performance Metrics")
    
    print(f"Uploading metrics for {num_days} days...\n")
    
    for day in range(num_days):
        sharpe = 1.5 - (day * 0.04) + random.uniform(-0.1, 0.1)
        returns = 0.02 - (day * 0.0008) + random.uniform(-0.005, 0.005)
        
        response = requests.post(
            f"{BASE_URL}/models/{model_id}/metrics",
            json={
                "returns": returns,
                "sharpe_ratio": sharpe,
                "volatility": 0.12 + random.uniform(-0.02, 0.02),
                "max_drawdown": -0.08 - (day * 0.001),
                "features": {
                    "momentum_score": [0.15 + day * 0.01 + random.uniform(-0.02, 0.02)],
                    "volatility": [0.20 + day * 0.005 + random.uniform(-0.01, 0.01)],
                    "market_beta": [0.82 + day * 0.003 + random.uniform(-0.03, 0.03)]
                },
                "extra_metadata": {
                    "trading_day": day + 1,
                    "market_regime": "volatile" if day > 15 else "stable"
                }
            }
        )
        
        if response.status_code == 200:
            if day % 5 == 0:
                print(f"  Day {day+1}: ✓ Uploaded (Sharpe: {sharpe:.2f}, Returns: {returns:.3f})")
        else:
            print(f"  Day {day+1}: ✗ Error {response.status_code}")
        
        time.sleep(0.1)
    
    print(f"\n✓ All {num_days} days of metrics uploaded successfully!")


def get_latest_drift_analysis(model_id):
    """Retrieve and display the latest drift analysis."""
    print_section("3. Get Drift Analysis")
    
    response = requests.get(f"{BASE_URL}/models/{model_id}/drift/latest")
    
    if response.status_code == 200:
        analysis = response.json()
        
        print("Drift Metrics:")
        print(f"  PSI Score: {analysis['psi_score']:.4f}")
        print(f"  KS Statistic: {analysis['ks_statistic']:.4f}")
        print(f"  KS p-value: {analysis['ks_p_value']:.4f}")
        print(f"  Drift Detected: {'YES' if analysis['drift_detected'] else 'NO'}")
        print(f"  Severity: {analysis['drift_severity'].upper()}")
        
        print("\nPerformance Metrics:")
        print(f"  Rolling Sharpe: {analysis['rolling_sharpe']:.4f}")
        print(f"  Sharpe Deterioration: {analysis['sharpe_deterioration']:.2f}%")
        
        print("\nFailure Forecasts:")
        if analysis['time_to_failure_bayesian'] and analysis['time_to_failure_bayesian'] > 0:
            print(f"  Bayesian: {analysis['time_to_failure_bayesian']:.1f} days")
        else:
            print(f"  Bayesian: No failure predicted")
        
        if analysis['time_to_failure_neural_sde'] and analysis['time_to_failure_neural_sde'] > 0:
            print(f"  Neural SDE: {analysis['time_to_failure_neural_sde']:.1f} days")
        else:
            print(f"  Neural SDE: No failure predicted")
        
        if analysis['failure_probability']:
            print(f"  Failure Probability: {analysis['failure_probability']:.2%}")
        
        return analysis
    else:
        print(f"✗ Error retrieving drift analysis: {response.status_code}")
        return None


def get_alerts(model_id):
    """Retrieve and display alerts."""
    print_section("4. Get Alerts")
    
    response = requests.get(f"{BASE_URL}/models/{model_id}/alerts")
    
    if response.status_code == 200:
        alerts = response.json()
        
        if alerts:
            print(f"Found {len(alerts)} alert(s):\n")
            for i, alert in enumerate(alerts[:5], 1):
                severity_icon = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'high' else "🟢"
                print(f"{i}. {severity_icon} [{alert['severity'].upper()}] {alert['alert_type']}")
                print(f"   {alert['message']}")
                print(f"   Time: {alert['alert_timestamp']}")
                print()
        else:
            print("No alerts found. Model is performing normally.")
        
        return alerts
    else:
        print(f"✗ Error retrieving alerts: {response.status_code}")
        return None


def generate_pdf_report(model_id):
    """Generate and save PDF report."""
    print_section("5. Generate PDF Report")
    
    response = requests.post(
        f"{BASE_URL}/reports/pdf",
        params={"model_id": model_id}
    )
    
    if response.status_code == 200:
        filename = f"quantora_report_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        with open(filename, "wb") as f:
            f.write(response.content)
        
        print(f"✓ PDF report generated successfully!")
        print(f"  Saved to: {filename}")
        print(f"  Size: {len(response.content)} bytes")
        return filename
    else:
        print(f"✗ Error generating PDF report: {response.status_code}")
        return None


def main():
    """Run complete example workflow."""
    print("\n" + "="*60)
    print("  QUANTORA API - Example Usage Workflow")
    print("="*60)
    
    model_id = register_model()
    
    if not model_id:
        print("\n✗ Failed to register model. Exiting.")
        return
    
    upload_metrics(model_id, num_days=30)
    
    time.sleep(2)
    
    get_latest_drift_analysis(model_id)
    
    get_alerts(model_id)
    
    generate_pdf_report(model_id)
    
    print("\n" + "="*60)
    print("  Workflow completed successfully!")
    print("="*60)
    print(f"\nYou can now:")
    print(f"  - View interactive API docs: {BASE_URL}/docs")
    print(f"  - Check model status: {BASE_URL}/models/{model_id}")
    print(f"  - View drift history: {BASE_URL}/models/{model_id}/drift")
    print()


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to Quantora API")
        print(f"  Make sure the server is running at {BASE_URL}")
        print(f"  Start it with: uvicorn app.main:app --host 0.0.0.0 --port 5000")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
