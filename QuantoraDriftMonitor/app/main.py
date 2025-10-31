from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import tempfile
import os
from datetime import datetime

from app import models, schemas
from app.database import get_db, init_db, engine
from app.drift_service import DriftAnalysisService
from app.report_generator import DriftReportGenerator
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Quantora - Quantitative Model Monitoring System
    
    Real-time monitoring and drift detection for quantitative and ML models.
    Provides Population Stability Index, Kolmogorov-Smirnov tests, 
    Sharpe ratio tracking, and Bayesian/Neural-SDE failure forecasting.
    
    Perfect for hedge-fund quants, AI risk auditors, and fintech engineers.
    """,
)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    init_db()


@app.get("/", tags=["Root"])
def root():
    """Root endpoint with API information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "register_model": "POST /models",
            "upload_metrics": "POST /models/{model_id}/metrics",
            "get_drift_analysis": "GET /models/{model_id}/drift",
            "generate_report": "POST /reports",
            "get_alerts": "GET /models/{model_id}/alerts"
        }
    }


@app.post("/models", response_model=schemas.ModelResponse, tags=["Models"])
def register_model(model: schemas.ModelCreate, db: Session = Depends(get_db)):
    """
    Register a new quantitative model for monitoring.
    
    Provide baseline feature distributions for drift detection.
    """
    existing = db.query(models.Model).filter(models.Model.name == model.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model with this name already exists")
    
    db_model = models.Model(
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        baseline_distribution=model.baseline_distribution
    )
    
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return db_model


@app.get("/models", response_model=List[schemas.ModelResponse], tags=["Models"])
def list_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all registered models."""
    models_list = db.query(models.Model).offset(skip).limit(limit).all()
    return models_list


@app.get("/models/{model_id}", response_model=schemas.ModelResponse, tags=["Models"])
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get details of a specific model."""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.post("/models/{model_id}/metrics", response_model=schemas.MetricResponse, tags=["Metrics"])
def upload_metric(
    model_id: int, 
    metric: schemas.MetricUpload, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Upload performance metrics for a model.
    
    Triggers drift analysis in the background.
    Metrics include returns, Sharpe ratio, predictions, actuals, and features.
    """
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db_metric = models.PerformanceMetric(
        model_id=model_id,
        returns=metric.returns,
        sharpe_ratio=metric.sharpe_ratio,
        volatility=metric.volatility,
        max_drawdown=metric.max_drawdown,
        predictions=metric.predictions,
        actuals=metric.actuals,
        features=metric.features,
        extra_metadata=metric.extra_metadata
    )
    
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    
    background_tasks.add_task(run_drift_analysis, model_id, db)
    
    return db_metric


def run_drift_analysis(model_id: int, db: Session):
    """Background task to run drift analysis."""
    service = DriftAnalysisService(db)
    service.analyze_model_drift(model_id)


@app.get("/models/{model_id}/metrics", response_model=List[schemas.MetricResponse], tags=["Metrics"])
def get_metrics(
    model_id: int, 
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get performance metrics for a model."""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metrics = db.query(models.PerformanceMetric).filter(
        models.PerformanceMetric.model_id == model_id
    ).order_by(
        models.PerformanceMetric.timestamp.desc()
    ).offset(skip).limit(limit).all()
    
    return metrics


@app.post("/models/{model_id}/analyze", response_model=schemas.DriftAnalysisResponse, tags=["Drift Analysis"])
def trigger_drift_analysis(model_id: int, db: Session = Depends(get_db)):
    """
    Manually trigger drift analysis for a model.
    
    Computes PSI, KS test, Sharpe deterioration, and failure forecasts.
    """
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    service = DriftAnalysisService(db)
    analysis = service.analyze_model_drift(model_id)
    
    if not analysis:
        raise HTTPException(
            status_code=400, 
            detail="Insufficient data for analysis. Need at least 20 metrics."
        )
    
    return analysis


@app.get("/models/{model_id}/drift", response_model=List[schemas.DriftAnalysisResponse], tags=["Drift Analysis"])
def get_drift_analyses(
    model_id: int, 
    skip: int = 0, 
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get drift analysis history for a model."""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    analyses = db.query(models.DriftAnalysis).filter(
        models.DriftAnalysis.model_id == model_id
    ).order_by(
        models.DriftAnalysis.analysis_timestamp.desc()
    ).offset(skip).limit(limit).all()
    
    return analyses


@app.get("/models/{model_id}/drift/latest", response_model=schemas.DriftAnalysisResponse, tags=["Drift Analysis"])
def get_latest_drift_analysis(model_id: int, db: Session = Depends(get_db)):
    """Get the most recent drift analysis for a model."""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    analysis = db.query(models.DriftAnalysis).filter(
        models.DriftAnalysis.model_id == model_id
    ).order_by(
        models.DriftAnalysis.analysis_timestamp.desc()
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="No drift analysis found")
    
    return analysis


@app.get("/models/{model_id}/alerts", response_model=List[schemas.AlertResponse], tags=["Alerts"])
def get_alerts(
    model_id: int, 
    acknowledged: Optional[bool] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get alerts for a model."""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    query = db.query(models.Alert).filter(models.Alert.model_id == model_id)
    
    if acknowledged is not None:
        query = query.filter(models.Alert.acknowledged == acknowledged)
    
    alerts = query.order_by(
        models.Alert.alert_timestamp.desc()
    ).offset(skip).limit(limit).all()
    
    return alerts


@app.post("/alerts/{alert_id}/acknowledge", tags=["Alerts"])
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    """Acknowledge an alert."""
    alert = db.query(models.Alert).filter(models.Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    db.commit()
    
    return {"status": "acknowledged", "alert_id": alert_id}


@app.post("/reports/pdf", tags=["Reports"])
def generate_pdf_report(
    model_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Generate PDF decay report for a model.
    
    Returns downloadable PDF with drift metrics, forecasts, and visualizations.
    """
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    query = db.query(models.DriftAnalysis).filter(
        models.DriftAnalysis.model_id == model_id
    )
    
    if start_date:
        query = query.filter(models.DriftAnalysis.analysis_timestamp >= start_date)
    if end_date:
        query = query.filter(models.DriftAnalysis.analysis_timestamp <= end_date)
    
    analyses = query.order_by(models.DriftAnalysis.analysis_timestamp).all()
    
    if not analyses:
        raise HTTPException(status_code=404, detail="No drift analyses found for this period")
    
    analyses_dict = [
        {
            "analysis_timestamp": a.analysis_timestamp,
            "psi_score": a.psi_score,
            "ks_statistic": a.ks_statistic,
            "ks_p_value": a.ks_p_value,
            "rolling_sharpe": a.rolling_sharpe,
            "time_to_failure_bayesian": a.time_to_failure_bayesian,
            "time_to_failure_neural_sde": a.time_to_failure_neural_sde,
            "failure_probability": a.failure_probability,
            "drift_severity": a.drift_severity
        }
        for a in analyses
    ]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_file.close()
    
    generator = DriftReportGenerator()
    pdf_path = generator.generate_pdf_report(
        model_name=model.name,
        drift_analyses=analyses_dict,
        output_path=temp_file.name
    )
    
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"quantora_report_{model.name}_{datetime.now().strftime('%Y%m%d')}.pdf"
    )


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    try:
        from app.database import get_engine
        engine = get_engine()
        engine.connect()
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "operational",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
