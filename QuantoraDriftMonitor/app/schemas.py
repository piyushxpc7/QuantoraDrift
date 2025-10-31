from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


class ModelCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Unique model identifier")
    description: Optional[str] = None
    model_type: Optional[str] = Field("quantitative", description="Type of model (e.g., quantitative, ML)")
    baseline_distribution: Optional[Dict[str, List[float]]] = Field(None, description="Baseline feature distributions")


class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: int
    name: str
    description: Optional[str]
    model_type: Optional[str]
    created_at: datetime
    is_active: bool


class MetricUpload(BaseModel):
    returns: Optional[float] = Field(None, description="Period returns")
    sharpe_ratio: Optional[float] = None
    volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    predictions: Optional[List[float]] = None
    actuals: Optional[List[float]] = None
    features: Optional[Dict[str, List[float]]] = Field(None, description="Feature values for drift detection")
    extra_metadata: Optional[Dict[str, Any]] = None


class MetricResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: int
    model_id: int
    timestamp: datetime
    returns: Optional[float]
    sharpe_ratio: Optional[float]


class DriftAnalysisResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: int
    model_id: int
    analysis_timestamp: datetime
    psi_score: Optional[float]
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    rolling_sharpe: Optional[float]
    sharpe_deterioration: Optional[float]
    time_to_failure_bayesian: Optional[float]
    time_to_failure_neural_sde: Optional[float]
    failure_probability: Optional[float]
    drift_detected: bool
    drift_severity: Optional[str]
    detailed_metrics: Optional[Dict[str, Any]]


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    
    id: int
    model_id: int
    alert_timestamp: datetime
    alert_type: str
    severity: str
    message: str
    metric_name: Optional[str]
    metric_value: Optional[float]
    acknowledged: bool


class DriftReportRequest(BaseModel):
    model_id: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: str = Field("json", description="Output format: 'json' or 'pdf'")
