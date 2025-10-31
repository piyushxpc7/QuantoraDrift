from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text)
    model_type = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    baseline_distribution = Column(JSON)
    
    metrics = relationship("PerformanceMetric", back_populates="model", cascade="all, delete-orphan")
    drift_analyses = relationship("DriftAnalysis", back_populates="model", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="model", cascade="all, delete-orphan")


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    returns = Column(Float)
    sharpe_ratio = Column(Float)
    volatility = Column(Float)
    max_drawdown = Column(Float)
    
    predictions = Column(JSON)
    actuals = Column(JSON)
    features = Column(JSON)
    
    extra_metadata = Column(JSON)
    
    model = relationship("Model", back_populates="metrics")


class DriftAnalysis(Base):
    __tablename__ = "drift_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    psi_score = Column(Float)
    ks_statistic = Column(Float)
    ks_p_value = Column(Float)
    rolling_sharpe = Column(Float)
    sharpe_deterioration = Column(Float)
    
    time_to_failure_bayesian = Column(Float)
    time_to_failure_neural_sde = Column(Float)
    failure_probability = Column(Float)
    
    drift_detected = Column(Boolean, default=False)
    drift_severity = Column(String)
    
    detailed_metrics = Column(JSON)
    
    model = relationship("Model", back_populates="drift_analyses")


class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    alert_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    alert_type = Column(String)
    severity = Column(String)
    message = Column(Text)
    
    metric_name = Column(String)
    metric_value = Column(Float)
    threshold = Column(Float)
    
    acknowledged = Column(Boolean, default=False)
    extra_metadata = Column(JSON)
    
    model = relationship("Model", back_populates="alerts")
