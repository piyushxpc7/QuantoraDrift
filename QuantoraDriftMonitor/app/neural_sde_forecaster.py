import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class NeuralSDEForecaster:
    """
    Neural Stochastic Differential Equation (SDE) forecaster.
    Simplified implementation using Ornstein-Uhlenbeck process for drift modeling.
    
    In production, this would use TorchSDE or similar, but we provide a 
    mathematically sound approximation using stochastic processes.
    """
    
    def __init__(self, drift_threshold: float = 0.3):
        """
        Args:
            drift_threshold: Drift score threshold for failure
        """
        self.drift_threshold = drift_threshold
        self.theta = None
        self.mu = None
        self.sigma = None
        self.scaler = StandardScaler()
    
    def _ornstein_uhlenbeck_fit(self, timestamps: np.ndarray, values: np.ndarray):
        """
        Fit Ornstein-Uhlenbeck process parameters using maximum likelihood.
        
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        """
        dt = np.diff(timestamps)
        dX = np.diff(values)
        X = values[:-1]
        
        if len(dt) == 0:
            return
        
        dt = np.maximum(dt, 1e-6)
        
        valid_mask = dt > 0
        if not np.any(valid_mask):
            return
        
        dt = dt[valid_mask]
        dX = dX[valid_mask]
        X = X[valid_mask]
        
        self.mu = np.mean(values)
        
        X_centered = X - self.mu
        
        theta_numerator = np.sum(X_centered * dX)
        theta_denominator = np.sum(X_centered ** 2 * dt)
        
        if theta_denominator > 1e-10:
            self.theta = -theta_numerator / theta_denominator
        else:
            self.theta = 0.1
        
        residuals = dX + self.theta * X_centered * dt
        self.sigma = np.sqrt(np.sum(residuals ** 2) / np.sum(dt))
        
        self.theta = max(0.01, abs(self.theta))
        self.sigma = max(0.01, self.sigma)
    
    def _simulate_path(self, 
                      initial_value: float, 
                      n_steps: int, 
                      dt: float, 
                      n_paths: int = 1000) -> np.ndarray:
        """
        Simulate future paths using fitted OU process.
        
        Returns:
            Array of shape (n_paths, n_steps) with simulated trajectories
        """
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = initial_value
        
        for i in range(1, n_steps):
            drift = self.theta * (self.mu - paths[:, i-1]) * dt
            diffusion = self.sigma * np.sqrt(dt) * np.random.randn(n_paths)
            paths[:, i] = paths[:, i-1] + drift + diffusion
        
        return paths
    
    def forecast(self, 
                timestamps: np.ndarray, 
                drift_scores: np.ndarray, 
                forecast_horizon: int = 30) -> Tuple[float, float]:
        """
        Forecast time to failure using Neural SDE approach.
        
        Args:
            timestamps: Array of timestamps (days since start)
            drift_scores: Array of drift scores
            forecast_horizon: Days to forecast ahead
            
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
        
        timestamps = timestamps - timestamps[0]
        
        time_diffs = np.diff(timestamps)
        if np.any(time_diffs <= 0):
            return -1.0, 0.0
        
        self._ornstein_uhlenbeck_fit(timestamps, drift_scores)
        
        dt = 1.0
        n_steps = forecast_horizon
        initial_value = drift_scores[-1]
        
        paths = self._simulate_path(initial_value, n_steps, dt, n_paths=1000)
        
        failure_times = []
        for path in paths:
            failure_idx = np.where(path >= self.drift_threshold)[0]
            if len(failure_idx) > 0:
                failure_times.append(failure_idx[0])
            else:
                failure_times.append(n_steps)
        
        mean_failure_time = np.mean(failure_times)
        failure_probability = np.mean([t < n_steps for t in failure_times])
        
        return float(mean_failure_time), float(failure_probability)
    
    def forecast_from_metrics(self, 
                            metric_times: list, 
                            drift_scores: list) -> Tuple[float, float]:
        """
        Convenience method to forecast from raw metrics.
        
        Args:
            metric_times: List of datetime objects
            drift_scores: List of drift scores
            
        Returns:
            Tuple of (days_to_failure, failure_probability)
        """
        if not metric_times or not drift_scores:
            return -1.0, 0.0
        
        timestamps_days = np.array([(t - metric_times[0]).days 
                                    for t in metric_times])
        
        return self.forecast(timestamps_days, np.array(drift_scores))


class HybridForecaster:
    """
    Ensemble forecaster combining Bayesian and Neural SDE approaches.
    """
    
    @staticmethod
    def combine_forecasts(bayesian_result: Tuple[float, float],
                         neural_sde_result: Tuple[float, float],
                         bayesian_weight: float = 0.5) -> Tuple[float, float]:
        """
        Combine Bayesian and Neural SDE forecasts using weighted average.
        
        Returns:
            Tuple of (ensemble_days_to_failure, ensemble_probability)
        """
        bayes_days, bayes_prob = bayesian_result
        sde_days, sde_prob = neural_sde_result
        
        if bayes_days < 0 and sde_days < 0:
            return -1.0, 0.0
        elif bayes_days < 0:
            return sde_days, sde_prob
        elif sde_days < 0:
            return bayes_days, bayes_prob
        
        ensemble_days = bayesian_weight * bayes_days + (1 - bayesian_weight) * sde_days
        ensemble_prob = bayesian_weight * bayes_prob + (1 - bayesian_weight) * sde_prob
        
        return float(ensemble_days), float(ensemble_prob)
