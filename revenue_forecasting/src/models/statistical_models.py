"""
Statistical forecasting models (ARIMA, SARIMA, SARIMAX, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from .base_model import BaseForecastingModel

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ARIMAModel(BaseForecastingModel):
    """ARIMA (AutoRegressive Integrated Moving Average) model."""
    
    def __init__(self, 
                 order: tuple = (1, 1, 1),
                 seasonal_order: tuple = None,
                 **kwargs):
        """
        Initialize ARIMA model.
        
        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order (for SARIMA)
            **kwargs: Additional parameters
        """
        super().__init__("ARIMA", **kwargs)
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA models")
            
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.train_data = None
        self.target_column = None
        self.date_column = None
        
    def fit(self, 
            train_data: pd.DataFrame,
            target_variable: str,
            date_column: str = 'date',
            feature_list: Optional[List[str]] = None,
            **kwargs) -> 'ARIMAModel':
        """Fit the ARIMA model."""
        start_time = datetime.now()
        
        self.validate_input(train_data, target_variable, date_column, feature_list)
        
        # Store data info
        self.train_data = train_data.copy()
        self.target_variable = target_variable
        self.date_column = date_column
        self.feature_list = feature_list or []
        
        # For backward compatibility
        self.target_column = target_variable
        
        # Prepare time series
        df = train_data.set_index(pd.to_datetime(train_data[date_column]))
        ts = df[target_variable].dropna()
        
        try:
            # Create and fit model
            if self.seasonal_order:
                self.model = SARIMAX(ts, 
                                   order=self.order,
                                   seasonal_order=self.seasonal_order,
                                   **kwargs)
            else:
                self.model = ARIMA(ts, order=self.order, **kwargs)
                
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Store metadata
            self.fit_time = (datetime.now() - start_time).total_seconds()
            self.add_metadata('aic', self.fitted_model.aic)
            self.add_metadata('bic', self.fitted_model.bic)
            self.add_metadata('training_samples', len(ts))
            
            logger.info(f"ARIMA model fitted successfully in {self.fit_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
            
        return self
    
    def predict(self, 
                periods: int,
                confidence_level: float = 0.95,
                include_history: bool = False,
                **kwargs) -> pd.DataFrame:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        start_time = datetime.now()
        
        try:
            # Generate predictions
            forecast = self.fitted_model.forecast(steps=periods, alpha=1-confidence_level)
            conf_int = self.fitted_model.get_forecast(steps=periods, alpha=1-confidence_level).conf_int()
            
            # Create date range for predictions
            last_date = pd.to_datetime(self.train_data[self.date_column]).max()
            freq = pd.infer_freq(pd.to_datetime(self.train_data[self.date_column]))
            future_dates = pd.date_range(start=last_date + pd.Timedelta(freq or 'D'), 
                                       periods=periods, freq=freq or 'D')
            
            # Create prediction DataFrame
            predictions = pd.DataFrame({
                self.date_column: future_dates,
                'forecast': forecast.values,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values
            })
            
            # Include history if requested
            if include_history:
                historical = self.train_data[[self.date_column, self.target_column]].copy()
                historical['forecast'] = historical[self.target_column]
                historical['lower_bound'] = historical[self.target_column]
                historical['upper_bound'] = historical[self.target_column]
                predictions = pd.concat([historical, predictions], ignore_index=True)
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Predictions generated in {self.prediction_time:.2f} seconds")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         **kwargs) -> pd.DataFrame:
        """Generate predictions with exogenous variables."""
        # For basic ARIMA, this is the same as regular predict
        # SARIMAX models can override this to handle exogenous variables
        return self.predict(periods, confidence_level, **kwargs)
    
    def get_model_summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            return "Model not fitted"
        return str(self.fitted_model.summary())


class SARIMAXModel(ARIMAModel):
    """SARIMAX (Seasonal ARIMA with eXogenous variables) model."""
    
    def __init__(self, 
                 order: tuple = (1, 1, 1),
                 seasonal_order: tuple = (1, 1, 1, 12),
                 exog_columns: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize SARIMAX model.
        
        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, s) seasonal order
            exog_columns: List of exogenous variable column names
            **kwargs: Additional parameters
        """
        super().__init__(order, seasonal_order, **kwargs)
        self.name = "SARIMAX"
        self.exog_columns = exog_columns or []
        
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str,
            **kwargs) -> 'SARIMAXModel':
        """Fit the SARIMAX model."""
        start_time = datetime.now()
        
        self.validate_input(train_data, target_column, date_column)
        
        # Store data info
        self.train_data = train_data.copy()
        self.target_column = target_column
        self.date_column = date_column
        
        # Prepare time series and exogenous variables
        df = train_data.set_index(pd.to_datetime(train_data[date_column]))
        ts = df[target_column].dropna()
        
        exog = None
        if self.exog_columns:
            # Check if exogenous columns exist
            missing_cols = [col for col in self.exog_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing exogenous columns: {missing_cols}")
                self.exog_columns = [col for col in self.exog_columns if col in df.columns]
            
            if self.exog_columns:
                exog = df[self.exog_columns].loc[ts.index]
        
        try:
            # Create and fit model
            self.model = SARIMAX(ts, 
                               exog=exog,
                               order=self.order,
                               seasonal_order=self.seasonal_order,
                               **kwargs)
                
            self.fitted_model = self.model.fit()
            self.is_fitted = True
            
            # Store metadata
            self.fit_time = (datetime.now() - start_time).total_seconds()
            self.add_metadata('aic', self.fitted_model.aic)
            self.add_metadata('bic', self.fitted_model.bic)
            self.add_metadata('training_samples', len(ts))
            self.add_metadata('exog_columns', self.exog_columns)
            
            logger.info(f"SARIMAX model fitted successfully in {self.fit_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {str(e)}")
            raise
            
        return self
    
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         **kwargs) -> pd.DataFrame:
        """Generate predictions with exogenous variables."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        start_time = datetime.now()
        
        try:
            # Prepare exogenous variables for prediction
            exog_pred = None
            if self.exog_columns and exog_future is not None:
                # Validate exogenous data
                missing_cols = [col for col in self.exog_columns if col not in exog_future.columns]
                if missing_cols:
                    raise ValueError(f"Missing exogenous columns in future data: {missing_cols}")
                
                if len(exog_future) < periods:
                    raise ValueError(f"Exogenous data has {len(exog_future)} rows, but {periods} periods requested")
                
                exog_pred = exog_future[self.exog_columns].iloc[:periods]
            
            # Generate predictions
            forecast_result = self.fitted_model.get_forecast(steps=periods, exog=exog_pred, alpha=1-confidence_level)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
            
            # Create date range for predictions
            last_date = pd.to_datetime(self.train_data[self.date_column]).max()
            freq = pd.infer_freq(pd.to_datetime(self.train_data[self.date_column]))
            future_dates = pd.date_range(start=last_date + pd.Timedelta(freq or 'D'), 
                                       periods=periods, freq=freq or 'D')
            
            # Create prediction DataFrame
            predictions = pd.DataFrame({
                self.date_column: future_dates,
                'forecast': forecast.values,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values
            })
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"SARIMAX predictions generated in {self.prediction_time:.2f} seconds")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating SARIMAX predictions: {str(e)}")
            raise


def check_stationarity(ts: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Check if a time series is stationary using Augmented Dickey-Fuller test.
    
    Args:
        ts: Time series to test
        significance_level: Significance level for the test
        
    Returns:
        Dictionary with test results
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for stationarity testing")
    
    result = adfuller(ts.dropna())
    
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level,
        'significance_level': significance_level
    }


def auto_arima_order(ts: pd.Series, 
                    max_p: int = 5, 
                    max_d: int = 2, 
                    max_q: int = 5,
                    seasonal: bool = False,
                    m: int = 12) -> tuple:
    """
    Automatically determine ARIMA order using AIC criterion.
    
    Args:
        ts: Time series data
        max_p: Maximum p order to test
        max_d: Maximum d order to test  
        max_q: Maximum q order to test
        seasonal: Whether to include seasonal terms
        m: Seasonal period
        
    Returns:
        Best (p, d, q) order and seasonal order if applicable
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for auto ARIMA")
    
    best_aic = float('inf')
    best_order = (0, 0, 0)
    best_seasonal_order = None
    
    # Test different combinations
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    if seasonal:
                        # Test seasonal orders
                        for P in range(3):
                            for D in range(2):
                                for Q in range(3):
                                    try:
                                        model = SARIMAX(ts, 
                                                      order=(p, d, q),
                                                      seasonal_order=(P, D, Q, m))
                                        fitted = model.fit(disp=False)
                                        
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                            best_seasonal_order = (P, D, Q, m)
                                    except:
                                        continue
                    else:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                            
                except:
                    continue
    
    if seasonal:
        return best_order, best_seasonal_order
    else:
        return best_order