# Quantora - Quantitative Model Monitoring System

Quantora is a production-grade FastAPI-based backend service that analyzes quantitative and machine-learning models for performance decay, concept drift, and data instability in real time.

## 🎯 Key Features

- **Advanced Drift Detection**
  - Population Stability Index (PSI) for feature distribution monitoring
  - Kolmogorov-Smirnov tests for distribution drift detection
  - Multi-feature drift analysis with automated severity classification

- **Performance Monitoring**
  - Rolling Sharpe ratio deterioration tracking
  - Real-time performance metric ingestion
  - Historical trend analysis with configurable windows

- **Predictive Analytics**
  - Bayesian regression-based time-to-failure forecasting
  - Neural SDE (Stochastic Differential Equation) forecasting
  - Dual forecasting methods for robust predictions

- **Comprehensive Reporting**
  - JSON API responses for programmatic access
  - PDF report generation with visualizations
  - Statistical summaries and actionable recommendations

- **Intelligent Alerting**
  - Threshold-based alert system
  - Multi-severity classification (low, medium, high, critical)
  - Customizable drift thresholds per model

## 🔧 Technology Stack

- **Framework**: FastAPI with Uvicorn ASGI server
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Statistical Computing**: NumPy, SciPy, pandas, scikit-learn
- **Bayesian Inference**: PyMC with PyTensor backend
- **Visualization**: matplotlib, seaborn
- **PDF Generation**: ReportLab
- **Background Processing**: APScheduler

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quantora.git
cd quantora

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/quantora"

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

### Access the API

- **Interactive API Docs**: http://localhost:5000/docs
- **Alternative Docs**: http://localhost:5000/redoc
- **Health Check**: http://localhost:5000/health

## 📊 API Usage Examples

### 1. Register a Model

```python
import requests

response = requests.post("http://localhost:5000/models", json={
    "name": "momentum_strategy_v1",
    "description": "Long-short momentum equity strategy",
    "model_type": "quantitative",
    "baseline_distribution": {
        "feature_1": [0.1, 0.2, 0.15, 0.18, 0.22],
        "feature_2": [1.5, 1.6, 1.4, 1.7, 1.5]
    }
})

model = response.json()
model_id = model["id"]
```

### 2. Upload Performance Metrics

```python
response = requests.post(
    f"http://localhost:5000/models/{model_id}/metrics",
    json={
        "returns": 0.025,
        "sharpe_ratio": 1.5,
        "volatility": 0.12,
        "max_drawdown": -0.08,
        "features": {
            "feature_1": [0.25, 0.28, 0.22],
            "feature_2": [1.8, 1.9, 1.7]
        }
    }
)
```

### 3. Get Drift Analysis

```python
response = requests.get(f"http://localhost:5000/models/{model_id}/drift/latest")
analysis = response.json()

print(f"PSI Score: {analysis['psi_score']}")
print(f"KS Statistic: {analysis['ks_statistic']}")
print(f"Drift Detected: {analysis['drift_detected']}")
print(f"Time to Failure (Bayesian): {analysis['time_to_failure_bayesian']} days")
print(f"Failure Probability: {analysis['failure_probability']}")
```

### 4. Generate PDF Report

```python
response = requests.post(
    f"http://localhost:5000/reports/pdf?model_id={model_id}",
)

with open("quantora_report.pdf", "wb") as f:
    f.write(response.content)
```

### 5. Get Alerts

```python
response = requests.get(f"http://localhost:5000/models/{model_id}/alerts")
alerts = response.json()

for alert in alerts:
    print(f"{alert['severity'].upper()}: {alert['message']}")
```

## 📈 Statistical Methods

### Population Stability Index (PSI)

Measures the shift in distribution between baseline and current feature values:

- **PSI < 0.1**: No significant drift
- **0.1 ≤ PSI < 0.25**: Moderate drift - monitor
- **PSI ≥ 0.25**: Significant drift - investigate

### Kolmogorov-Smirnov Test

Two-sample KS test to detect distribution changes:

- **p-value < 0.05**: Statistically significant drift detected
- Combined with KS statistic for severity classification

### Rolling Sharpe Ratio

Monitors performance deterioration:

- Calculated over configurable rolling windows
- Detects sudden performance drops
- Triggers alerts on threshold breaches

### Bayesian Time-to-Failure Forecasting

Uses Bayesian linear regression to model drift progression:

- Probabilistic predictions with uncertainty quantification
- Estimates days until critical threshold breach
- Provides failure probability estimates

### Neural SDE Forecasting

Ornstein-Uhlenbeck process for stochastic drift modeling:

- Captures non-linear drift patterns
- Monte Carlo simulation for uncertainty
- Alternative to Bayesian approach for robustness

## 🎛️ Configuration

Default thresholds (configurable via environment or `app/config.py`):

```python
PSI_THRESHOLD = 0.25
KS_THRESHOLD = 0.2
SHARPE_WINDOW = 30
SHARPE_THRESHOLD = -0.5
MIN_SAMPLES_FOR_ANALYSIS = 20
```

## 📋 API Endpoints

### Models
- `POST /models` - Register a new model
- `GET /models` - List all models
- `GET /models/{model_id}` - Get model details

### Metrics
- `POST /models/{model_id}/metrics` - Upload performance metrics
- `GET /models/{model_id}/metrics` - Get metric history

### Drift Analysis
- `POST /models/{model_id}/analyze` - Trigger drift analysis
- `GET /models/{model_id}/drift` - Get drift analysis history
- `GET /models/{model_id}/drift/latest` - Get latest analysis

### Alerts
- `GET /models/{model_id}/alerts` - Get alerts for a model
- `POST /alerts/{alert_id}/acknowledge` - Acknowledge an alert

### Reports
- `POST /reports/pdf` - Generate PDF decay report

### System
- `GET /health` - Health check endpoint
- `GET /` - API information

## 🏗️ Architecture

```
quantora/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── config.py                  # Configuration management
│   ├── database.py                # Database connection
│   ├── models.py                  # SQLAlchemy ORM models
│   ├── schemas.py                 # Pydantic schemas
│   ├── drift_detectors.py         # PSI, KS, Sharpe calculations
│   ├── bayesian_forecaster.py     # Bayesian regression forecasting
│   ├── neural_sde_forecaster.py   # Neural SDE forecasting
│   ├── drift_service.py           # Drift analysis orchestration
│   └── report_generator.py        # PDF report generation
├── requirements.txt
├── .gitignore
└── README.md
```

## 🎯 Use Cases

### Hedge Fund Quants
- Monitor trading strategies for performance decay
- Detect regime changes in market conditions
- Early warning for alpha deterioration

### AI Risk Auditors
- Track ML model drift in production
- Compliance reporting for model governance
- Risk management for deployed AI systems

### Fintech Engineers
- Real-time monitoring of prediction models
- Automated retraining triggers
- Performance SLA enforcement

## 📊 Example Workflow

```python
import requests
import time

base_url = "http://localhost:5000"

model = requests.post(f"{base_url}/models", json={
    "name": "my_trading_model",
    "model_type": "quantitative",
    "baseline_distribution": {
        "momentum": [0.1, 0.15, 0.12, 0.18],
        "volatility": [0.2, 0.22, 0.19, 0.21]
    }
}).json()

model_id = model["id"]

for day in range(30):
    requests.post(f"{base_url}/models/{model_id}/metrics", json={
        "returns": 0.02 - day * 0.001,
        "sharpe_ratio": 1.5 - day * 0.05,
        "features": {
            "momentum": [0.15 + day * 0.01],
            "volatility": [0.20 + day * 0.005]
        }
    })
    time.sleep(1)

analysis = requests.get(f"{base_url}/models/{model_id}/drift/latest").json()
print(f"Drift Severity: {analysis['drift_severity']}")
print(f"Days to Failure: {analysis['time_to_failure_bayesian']}")

requests.post(f"{base_url}/reports/pdf?model_id={model_id}")
```

## 🔒 Security

- Never expose API keys or database credentials in code
- Use environment variables for sensitive configuration
- Implement authentication/authorization for production deployment
- Use HTTPS in production environments

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Built with ❤️ for quantitative analysts and ML engineers who demand production-grade model monitoring.

---

**Quantora** - Because your models deserve real-time vigilance.
