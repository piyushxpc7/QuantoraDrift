# Quantora - Project Documentation

## Overview

Quantora is a production-grade FastAPI service for real-time quantitative model monitoring. It provides advanced statistical drift detection, performance tracking, and predictive failure forecasting for hedge-fund quants, AI risk auditors, and fintech engineers.

## Project Structure

```
quantora/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application with all endpoints
│   ├── config.py                  # Settings and configuration
│   ├── database.py                # PostgreSQL connection and session management
│   ├── models.py                  # SQLAlchemy ORM models
│   ├── schemas.py                 # Pydantic request/response schemas
│   ├── drift_detectors.py         # PSI, KS, Sharpe ratio calculators
│   ├── bayesian_forecaster.py     # Bayesian regression forecasting
│   ├── neural_sde_forecaster.py   # Neural SDE forecasting
│   ├── drift_service.py           # Drift analysis orchestration service
│   └── report_generator.py        # PDF report generation with charts
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore patterns
└── README.md                     # Comprehensive documentation
```

## Recent Changes

**October 31, 2025**
- Initial complete implementation of Quantora
- Implemented all core statistical modules (PSI, KS test, Sharpe monitoring)
- Built Bayesian regression forecaster using PyMC
- Implemented Neural SDE forecaster with Ornstein-Uhlenbeck process
- Created comprehensive FastAPI REST API with 15+ endpoints
- Built PDF report generator with matplotlib charts and ReportLab
- Set up PostgreSQL database with full schema
- Configured workflow for uvicorn server
- Fixed SQLAlchemy reserved name conflict (metadata → extra_metadata)
- Fixed Pydantic protected namespace warnings

## User Preferences

- Production-ready code quality
- Comprehensive statistical implementations
- Full-featured API (not MVP)
- Professional documentation
- GitHub deployment ready

## Technical Architecture

### Database Schema

**Models Table**: Stores registered quantitative models
- Fields: id, name, description, model_type, baseline_distribution, timestamps

**PerformanceMetrics Table**: Stores uploaded metrics
- Fields: returns, sharpe_ratio, volatility, predictions, actuals, features

**DriftAnalyses Table**: Stores computed drift analyses
- Fields: PSI, KS stats, Sharpe metrics, forecasts, severity

**Alerts Table**: Stores threshold breach alerts
- Fields: alert_type, severity, message, metric values

### Key Statistical Modules

**Population Stability Index (PSI)**
- Measures distribution shift between baseline and current data
- Bins data into deciles and compares frequency distributions
- Thresholds: <0.1 (no drift), 0.1-0.25 (moderate), >0.25 (significant)

**Kolmogorov-Smirnov Test**
- Two-sample test for distribution drift
- Returns KS statistic and p-value
- Combines with PSI for comprehensive drift detection

**Rolling Sharpe Ratio Monitor**
- Calculates rolling Sharpe over configurable windows (default 30 periods)
- Detects performance deterioration
- Triggers alerts on threshold breaches

**Bayesian Time-to-Failure Forecaster**
- Uses PyMC for Bayesian linear regression
- Models drift progression as: drift = α + β * time
- Forecasts days until failure threshold
- Provides probability estimates

**Neural SDE Forecaster**
- Implements Ornstein-Uhlenbeck process
- Fits: dX_t = θ(μ - X_t)dt + σdW_t
- Monte Carlo simulation for uncertainty
- Alternative to Bayesian for robustness

### API Endpoints

**Core Operations**
- Model registration and management
- Metric upload with background drift analysis
- Manual drift analysis triggers
- Alert retrieval and acknowledgment
- PDF report generation

**Auto-generated Documentation**
- Swagger UI at /docs
- ReDoc at /redoc
- Full OpenAPI 3.0 schema

### Background Processing

- Metric uploads trigger background drift analysis
- Compute-intensive operations offloaded from request cycle
- APScheduler for task management

## Dependencies

**Core Framework**
- FastAPI 0.104.1 with Uvicorn
- SQLAlchemy 2.0.23 + psycopg2-binary
- Pydantic 2.5.0 for validation

**Statistical Computing**
- NumPy 1.26.2, SciPy 1.11.4
- pandas 2.1.3, scikit-learn 1.3.2

**Bayesian Inference**
- PyMC 5.10.1, PyTensor 2.18.1

**Visualization**
- matplotlib 3.8.2, seaborn 0.13.0
- ReportLab 4.0.7 for PDFs

## Configuration

Environment variables (set via Replit secrets):
- `DATABASE_URL`: PostgreSQL connection string
- Optional: `PSI_THRESHOLD`, `KS_THRESHOLD`, etc.

Default thresholds (app/config.py):
```python
psi_threshold = 0.25
ks_threshold = 0.2
sharpe_window = 30
sharpe_threshold = -0.5
min_samples_for_analysis = 20
```

## Workflow

**quantora-api**: 
- Command: `uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload`
- Type: webview on port 5000
- Status: Running

## Known Issues / Warnings

- Pydantic warnings about model_ namespace (resolved with ConfigDict)
- PyTensor BLAS warning (non-critical, uses NumPy backend)
- SQLAlchemy type hints in LSP (false positives, runtime works correctly)

## Future Enhancements

Potential additions:
- Webhook notifications for critical alerts
- Additional drift metrics (Wasserstein, Jensen-Shannon divergence)
- Interactive dashboard UI
- Batch model comparison
- MLflow integration
- Docker containerization
- Authentication/authorization
- Rate limiting
- Caching layer

## Deployment

Ready for GitHub deployment with:
- Complete codebase
- Requirements.txt
- Comprehensive README
- .gitignore configured
- Production-ready structure

## Usage Patterns

1. Register model with baseline distribution
2. Upload performance metrics regularly
3. System auto-analyzes drift in background
4. Retrieve drift analyses via API
5. Generate PDF reports as needed
6. Monitor alerts for threshold breaches
7. Acknowledge alerts when addressed

## Support

- API docs: http://localhost:5000/docs
- Health check: http://localhost:5000/health
- GitHub: (to be added after deployment)
