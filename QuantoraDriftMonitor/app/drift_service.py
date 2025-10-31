from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
from app import models, schemas
from app.drift_detectors import (
    PopulationStabilityIndex, 
    KolmogorovSmirnovTest,
    SharpeRatioMonitor,
    DriftSeverityClassifier
)
from app.bayesian_forecaster import BayesianTimeToFailureForecaster
from app.neural_sde_forecaster import NeuralSDEForecaster
from app.config import get_settings


class DriftAnalysisService:
    """
    Orchestrates drift detection, forecasting, and alert generation.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
    
    def analyze_model_drift(self, model_id: int) -> Optional[models.DriftAnalysis]:
        """
        Perform comprehensive drift analysis on a model.
        
        Returns:
            DriftAnalysis object with all metrics
        """
        model = self.db.query(models.Model).filter(models.Model.id == model_id).first()
        if not model:
            return None
        
        metrics = self.db.query(models.PerformanceMetric).filter(
            models.PerformanceMetric.model_id == model_id
        ).order_by(models.PerformanceMetric.timestamp).all()
        
        if len(metrics) < self.settings.min_samples_for_analysis:
            return None
        
        drift_analysis = models.DriftAnalysis(model_id=model_id)
        
        psi_score, ks_stat, ks_p = self._compute_feature_drift(model, metrics)
        drift_analysis.psi_score = psi_score
        drift_analysis.ks_statistic = ks_stat
        drift_analysis.ks_p_value = ks_p
        
        rolling_sharpe, deterioration = self._compute_sharpe_metrics(metrics)
        drift_analysis.rolling_sharpe = rolling_sharpe
        drift_analysis.sharpe_deterioration = deterioration
        
        ttf_bayesian, ttf_neural, failure_prob = self._forecast_failure(metrics, psi_score)
        drift_analysis.time_to_failure_bayesian = ttf_bayesian
        drift_analysis.time_to_failure_neural_sde = ttf_neural
        drift_analysis.failure_probability = failure_prob
        
        drift_analysis.drift_detected = self._detect_drift(psi_score, ks_stat, ks_p)
        drift_analysis.drift_severity = self._classify_severity(psi_score, ks_stat, ks_p)
        
        drift_analysis.detailed_metrics = {
            "n_samples": len(metrics),
            "analysis_window_days": (metrics[-1].timestamp - metrics[0].timestamp).days,
            "latest_timestamp": metrics[-1].timestamp.isoformat()
        }
        
        self.db.add(drift_analysis)
        self.db.commit()
        self.db.refresh(drift_analysis)
        
        self._generate_alerts(model_id, drift_analysis)
        
        return drift_analysis
    
    def _compute_feature_drift(self, 
                              model: models.Model, 
                              metrics: List[models.PerformanceMetric]) -> Tuple[float, float, float]:
        """Compute PSI and KS test for feature drift."""
        if not model.baseline_distribution:
            recent_metrics = metrics[-20:]
            current_features = self._aggregate_features(recent_metrics)
            return 0.0, 0.0, 1.0
        
        baseline = model.baseline_distribution
        
        recent_metrics = metrics[-20:]
        current_features = self._aggregate_features(recent_metrics)
        
        psi_scores = PopulationStabilityIndex.calculate_multi_feature_psi(
            baseline, current_features
        )
        avg_psi = np.mean(list(psi_scores.values())) if psi_scores else 0.0
        
        ks_results = KolmogorovSmirnovTest.calculate_multi_feature_ks(
            baseline, current_features
        )
        
        if ks_results:
            avg_ks_stat = np.mean([r['statistic'] for r in ks_results.values()])
            avg_ks_p = np.mean([r['p_value'] for r in ks_results.values()])
        else:
            avg_ks_stat, avg_ks_p = 0.0, 1.0
        
        return float(avg_psi), float(avg_ks_stat), float(avg_ks_p)
    
    def _aggregate_features(self, 
                           metrics: List[models.PerformanceMetric]) -> Dict[str, List[float]]:
        """Aggregate features from metrics."""
        aggregated = {}
        
        for metric in metrics:
            if metric.features:
                for feature_name, values in metric.features.items():
                    if feature_name not in aggregated:
                        aggregated[feature_name] = []
                    if isinstance(values, list):
                        aggregated[feature_name].extend(values)
                    else:
                        aggregated[feature_name].append(values)
        
        return aggregated
    
    def _compute_sharpe_metrics(self, 
                               metrics: List[models.PerformanceMetric]) -> Tuple[float, float]:
        """Compute rolling Sharpe ratio and deterioration."""
        returns = [m.returns for m in metrics if m.returns is not None]
        
        if len(returns) < self.settings.sharpe_window:
            return 0.0, 0.0
        
        rolling_sharpe = SharpeRatioMonitor.calculate_rolling_sharpe(
            returns, window=self.settings.sharpe_window
        )
        
        if not rolling_sharpe:
            return 0.0, 0.0
        
        current_sharpe = rolling_sharpe[-1]
        baseline_sharpe = rolling_sharpe[0] if rolling_sharpe[0] != 0 else 1.0
        
        deterioration = SharpeRatioMonitor.calculate_sharpe_deterioration(
            current_sharpe, baseline_sharpe
        )
        
        return float(current_sharpe), float(deterioration)
    
    def _forecast_failure(self, 
                         metrics: List[models.PerformanceMetric],
                         psi_score: float) -> Tuple[float, float, float]:
        """Forecast time to failure using both methods."""
        if len(metrics) < 10:
            return -1.0, -1.0, 0.0
        
        timestamps = [m.timestamp for m in metrics]
        
        recent_analyses = self.db.query(models.DriftAnalysis).filter(
            models.DriftAnalysis.model_id == metrics[0].model_id
        ).order_by(models.DriftAnalysis.analysis_timestamp).limit(30).all()
        
        if len(recent_analyses) >= 5:
            analysis_times = [a.analysis_timestamp for a in recent_analyses]
            psi_scores = [a.psi_score for a in recent_analyses if a.psi_score is not None]
            
            if len(psi_scores) >= 5:
                bayesian_forecaster = BayesianTimeToFailureForecaster(
                    failure_threshold=self.settings.psi_threshold * 2
                )
                ttf_bayes, prob_bayes = bayesian_forecaster.forecast_from_metrics(
                    analysis_times, psi_scores
                )
                
                neural_forecaster = NeuralSDEForecaster(
                    drift_threshold=self.settings.psi_threshold * 2
                )
                ttf_neural, prob_neural = neural_forecaster.forecast_from_metrics(
                    analysis_times, psi_scores
                )
                
                avg_prob = (prob_bayes + prob_neural) / 2
                
                return ttf_bayes, ttf_neural, avg_prob
        
        return -1.0, -1.0, 0.0
    
    def _detect_drift(self, psi_score: float, ks_stat: float, ks_p: float) -> bool:
        """Determine if drift is detected."""
        psi_drift = psi_score > self.settings.psi_threshold
        ks_drift = ks_stat > self.settings.ks_threshold and ks_p < 0.05
        
        return psi_drift or ks_drift
    
    def _classify_severity(self, psi_score: float, ks_stat: float, ks_p: float) -> str:
        """Classify drift severity."""
        psi_severity = DriftSeverityClassifier.classify_psi_severity(psi_score)
        ks_severity = DriftSeverityClassifier.classify_ks_severity(ks_stat, ks_p)
        
        return DriftSeverityClassifier.get_overall_severity(psi_severity, ks_severity)
    
    def _generate_alerts(self, model_id: int, analysis: models.DriftAnalysis):
        """Generate alerts based on drift analysis."""
        if analysis.psi_score and analysis.psi_score > self.settings.psi_threshold:
            alert = models.Alert(
                model_id=model_id,
                alert_type="drift_detection",
                severity="high" if analysis.psi_score > 0.5 else "medium",
                message=f"PSI score ({analysis.psi_score:.4f}) exceeds threshold ({self.settings.psi_threshold})",
                metric_name="psi_score",
                metric_value=analysis.psi_score,
                threshold=self.settings.psi_threshold
            )
            self.db.add(alert)
        
        if analysis.ks_statistic and analysis.ks_statistic > self.settings.ks_threshold:
            if analysis.ks_p_value and analysis.ks_p_value < 0.05:
                alert = models.Alert(
                    model_id=model_id,
                    alert_type="drift_detection",
                    severity="high" if analysis.ks_statistic > 0.3 else "medium",
                    message=f"KS statistic ({analysis.ks_statistic:.4f}) indicates distribution shift",
                    metric_name="ks_statistic",
                    metric_value=analysis.ks_statistic,
                    threshold=self.settings.ks_threshold
                )
                self.db.add(alert)
        
        if analysis.time_to_failure_bayesian and 0 < analysis.time_to_failure_bayesian < 14:
            alert = models.Alert(
                model_id=model_id,
                alert_type="failure_warning",
                severity="critical",
                message=f"Model failure predicted in {analysis.time_to_failure_bayesian:.1f} days",
                metric_name="time_to_failure",
                metric_value=analysis.time_to_failure_bayesian,
                threshold=14.0
            )
            self.db.add(alert)
        
        self.db.commit()
