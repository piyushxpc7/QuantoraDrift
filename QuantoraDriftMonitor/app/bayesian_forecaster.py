import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class BayesianTimeToFailureForecaster:
    """
    Bayesian regression-based time-to-failure forecaster.
    Uses probabilistic modeling to predict when model performance will degrade beyond threshold.
    """
    
    def __init__(self, failure_threshold: float = 0.3):
        """
        Args:
            failure_threshold: PSI/KS threshold above which model is considered failed
        """
        self.failure_threshold = failure_threshold
        self.model = None
        self.trace = None
    
    def fit_and_forecast(self, 
                        timestamps: np.ndarray, 
                        drift_scores: np.ndarray,
                        forecast_horizon: int = 30) -> Tuple[float, float]:
        """
        Fit Bayesian regression model and forecast time to failure.
        
        Args:
            timestamps: Array of timestamps (as days since start)
            drift_scores: Array of drift scores (PSI or KS statistics)
            forecast_horizon: Number of days to forecast ahead
            
        Returns:
            Tuple of (predicted_days_to_failure, failure_probability)
        """
        if len(timestamps) < 5 or len(drift_scores) < 5:
            return -1.0, 0.0
        
        timestamps = np.array(timestamps).astype(float)
        drift_scores = np.array(drift_scores).astype(float)
        
        timestamps_unique, unique_indices = np.unique(timestamps, return_index=True)
        if len(timestamps_unique) < len(timestamps):
            timestamps = timestamps_unique
            drift_scores = drift_scores[unique_indices]
        
        if len(timestamps) < 5:
            return -1.0, 0.0
        
        timestamps = (timestamps - timestamps[0])
        
        try:
            with pm.Model() as self.model:
                alpha = pm.Normal('alpha', mu=0, sigma=1)
                beta = pm.Normal('beta', mu=0, sigma=1)
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                mu = alpha + beta * timestamps
                
                y = pm.Normal('y', mu=mu, sigma=sigma, observed=drift_scores)
                
                self.trace = pm.sample(500, tune=250, return_inferencedata=True, 
                                      progressbar=False, random_seed=42, 
                                      cores=1, chains=2)
            
            posterior_alpha = self.trace.posterior['alpha'].values.flatten()
            posterior_beta = self.trace.posterior['beta'].values.flatten()
            
            mean_alpha = np.mean(posterior_alpha)
            mean_beta = np.mean(posterior_beta)
            
            if mean_beta <= 0:
                return -1.0, 0.0
            
            current_time = timestamps[-1]
            days_to_failure = (self.failure_threshold - mean_alpha) / mean_beta - current_time
            
            if days_to_failure < 0:
                days_to_failure = 0.0
            
            forecast_times = np.linspace(current_time, current_time + forecast_horizon, 100)
            forecast_scores = mean_alpha + mean_beta * forecast_times
            
            failure_probability = np.mean(forecast_scores >= self.failure_threshold)
            
            return float(days_to_failure), float(failure_probability)
            
        except Exception as e:
            print(f"Bayesian forecasting error: {e}")
            return -1.0, 0.0
    
    def forecast_from_metrics(self, 
                            metric_times: list, 
                            psi_scores: list) -> Tuple[float, float]:
        """
        Convenience method to forecast from raw metrics.
        
        Args:
            metric_times: List of datetime objects
            psi_scores: List of PSI scores
            
        Returns:
            Tuple of (days_to_failure, failure_probability)
        """
        if not metric_times or not psi_scores:
            return -1.0, 0.0
        
        timestamps_days = np.array([(t - metric_times[0]).days 
                                    for t in metric_times])
        
        return self.fit_and_forecast(timestamps_days, np.array(psi_scores))


class BayesianPerformanceForecaster:
    """
    Bayesian forecaster for Sharpe ratio deterioration.
    """
    
    def __init__(self, critical_sharpe: float = 0.0):
        """
        Args:
            critical_sharpe: Sharpe ratio below which performance is critical
        """
        self.critical_sharpe = critical_sharpe
    
    def forecast_sharpe_deterioration(self, 
                                     timestamps: np.ndarray, 
                                     sharpe_ratios: np.ndarray) -> Tuple[float, float]:
        """
        Forecast when Sharpe ratio will drop below critical level.
        
        Returns:
            Tuple of (days_to_critical, probability_below_critical)
        """
        if len(timestamps) < 5 or len(sharpe_ratios) < 5:
            return -1.0, 0.0
        
        timestamps = np.array(timestamps).astype(float)
        sharpe_ratios = np.array(sharpe_ratios).astype(float)
        
        timestamps = timestamps - timestamps[0]
        
        try:
            timestamps_unique, unique_indices = np.unique(timestamps, return_index=True)
            if len(timestamps_unique) < len(timestamps):
                timestamps = timestamps_unique
                sharpe_ratios = sharpe_ratios[unique_indices]
            
            if len(timestamps) < 5:
                return -1.0, 0.0
            
            with pm.Model() as model:
                alpha = pm.Normal('alpha', mu=sharpe_ratios.mean(), sigma=1)
                beta = pm.Normal('beta', mu=0, sigma=0.1)
                sigma = pm.HalfNormal('sigma', sigma=0.5)
                
                mu = alpha + beta * timestamps
                
                y = pm.Normal('y', mu=mu, sigma=sigma, observed=sharpe_ratios)
                
                trace = pm.sample(500, tune=250, return_inferencedata=True, 
                                progressbar=False, random_seed=42,
                                cores=1, chains=2)
            
            mean_alpha = float(trace.posterior['alpha'].values.mean())
            mean_beta = float(trace.posterior['beta'].values.mean())
            
            if mean_beta >= 0:
                return -1.0, 0.0
            
            current_time = timestamps[-1]
            days_to_critical = (self.critical_sharpe - mean_alpha) / mean_beta - current_time
            
            if days_to_critical < 0:
                days_to_critical = 0.0
            
            current_sharpe = sharpe_ratios[-1]
            probability_below = 1.0 if current_sharpe < self.critical_sharpe else 0.5
            
            return float(days_to_critical), float(probability_below)
            
        except Exception as e:
            print(f"Sharpe forecasting error: {e}")
            return -1.0, 0.0
