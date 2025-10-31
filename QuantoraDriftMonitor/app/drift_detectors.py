import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) calculator for feature drift detection.
    PSI measures the shift in distribution between baseline and current data.
    """
    
    @staticmethod
    def calculate_psi(baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Calculate PSI between baseline and current distributions.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            bins: Number of bins for discretization
            
        Returns:
            PSI score (0 = no drift, >0.25 = significant drift)
        """
        if len(baseline) == 0 or len(current) == 0:
            return 0.0
        
        breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        if len(breakpoints) < 2:
            return 0.0
        
        baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
        current_counts = np.histogram(current, bins=breakpoints)[0]
        
        baseline_percents = baseline_counts / len(baseline)
        current_percents = current_counts / len(current)
        
        epsilon = 1e-10
        baseline_percents = np.where(baseline_percents == 0, epsilon, baseline_percents)
        current_percents = np.where(current_percents == 0, epsilon, current_percents)
        
        psi_values = (current_percents - baseline_percents) * np.log(current_percents / baseline_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    
    @staticmethod
    def calculate_multi_feature_psi(baseline: Dict[str, List[float]], 
                                   current: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate PSI for multiple features.
        
        Returns:
            Dictionary of feature_name: PSI score
        """
        psi_scores = {}
        
        for feature_name in baseline.keys():
            if feature_name in current:
                baseline_vals = np.array(baseline[feature_name])
                current_vals = np.array(current[feature_name])
                
                psi_scores[feature_name] = PopulationStabilityIndex.calculate_psi(
                    baseline_vals, current_vals
                )
        
        return psi_scores


class KolmogorovSmirnovTest:
    """
    Kolmogorov-Smirnov test for distribution drift detection.
    Tests whether two samples come from the same distribution.
    """
    
    @staticmethod
    def calculate_ks(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform two-sample KS test.
        
        Args:
            baseline: Baseline feature values
            current: Current feature values
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        if len(baseline) == 0 or len(current) == 0:
            return 0.0, 1.0
        
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        
        return float(ks_stat), float(p_value)
    
    @staticmethod
    def calculate_multi_feature_ks(baseline: Dict[str, List[float]], 
                                   current: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate KS test for multiple features.
        
        Returns:
            Dictionary of feature_name: {statistic, p_value}
        """
        ks_results = {}
        
        for feature_name in baseline.keys():
            if feature_name in current:
                baseline_vals = np.array(baseline[feature_name])
                current_vals = np.array(current[feature_name])
                
                ks_stat, p_value = KolmogorovSmirnovTest.calculate_ks(
                    baseline_vals, current_vals
                )
                
                ks_results[feature_name] = {
                    "statistic": ks_stat,
                    "p_value": p_value
                }
        
        return ks_results


class SharpeRatioMonitor:
    """
    Rolling Sharpe ratio calculator for performance deterioration detection.
    """
    
    @staticmethod
    def calculate_rolling_sharpe(returns: List[float], window: int = 30, 
                                 risk_free_rate: float = 0.0) -> List[float]:
        """
        Calculate rolling Sharpe ratio.
        
        Args:
            returns: List of returns
            window: Rolling window size
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            List of rolling Sharpe ratios
        """
        if len(returns) < window:
            return []
        
        returns_series = pd.Series(returns)
        
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return rolling_sharpe.tolist()
    
    @staticmethod
    def calculate_sharpe_deterioration(current_sharpe: float, 
                                      baseline_sharpe: float) -> float:
        """
        Calculate deterioration in Sharpe ratio.
        
        Returns:
            Deterioration percentage (negative means worse performance)
        """
        if baseline_sharpe == 0:
            return 0.0
        
        deterioration = ((current_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100
        
        return float(deterioration)
    
    @staticmethod
    def detect_sharpe_threshold_breach(rolling_sharpe: List[float], 
                                       threshold: float = -0.5) -> bool:
        """
        Detect if Sharpe ratio has breached threshold.
        
        Returns:
            True if threshold breached
        """
        if not rolling_sharpe:
            return False
        
        recent_sharpe = rolling_sharpe[-10:] if len(rolling_sharpe) >= 10 else rolling_sharpe
        
        return any(s < threshold for s in recent_sharpe)


class DriftSeverityClassifier:
    """
    Classifies drift severity based on PSI and KS test results.
    """
    
    PSI_THRESHOLDS = {
        "low": 0.1,
        "medium": 0.25,
        "high": 0.5
    }
    
    KS_THRESHOLDS = {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.3
    }
    
    @staticmethod
    def classify_psi_severity(psi_score: float) -> str:
        """Classify PSI score severity."""
        if psi_score < DriftSeverityClassifier.PSI_THRESHOLDS["low"]:
            return "none"
        elif psi_score < DriftSeverityClassifier.PSI_THRESHOLDS["medium"]:
            return "low"
        elif psi_score < DriftSeverityClassifier.PSI_THRESHOLDS["high"]:
            return "medium"
        else:
            return "high"
    
    @staticmethod
    def classify_ks_severity(ks_stat: float, p_value: float) -> str:
        """Classify KS statistic severity."""
        if p_value > 0.05:
            return "none"
        elif ks_stat < DriftSeverityClassifier.KS_THRESHOLDS["medium"]:
            return "low"
        elif ks_stat < DriftSeverityClassifier.KS_THRESHOLDS["high"]:
            return "medium"
        else:
            return "high"
    
    @staticmethod
    def get_overall_severity(psi_severity: str, ks_severity: str) -> str:
        """Get overall drift severity."""
        severity_levels = {"none": 0, "low": 1, "medium": 2, "high": 3}
        
        max_severity = max(severity_levels.get(psi_severity, 0), 
                          severity_levels.get(ks_severity, 0))
        
        for severity, level in severity_levels.items():
            if level == max_severity:
                return severity
        
        return "none"
