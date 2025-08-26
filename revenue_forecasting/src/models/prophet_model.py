"""
Prophet model implementation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .base_model import BaseForecastingModel

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProphetModel(BaseForecastingModel):
    """Facebook Prophet model for time series forecasting."""
    
    def __init__(self, 
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 growth: str = 'linear',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 country_holidays: Optional[str] = None,
                 **kwargs):
        """
        Initialize Prophet model.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            growth: 'linear' or 'logistic'
            changepoint_prior_scale: Flexibility of automatic changepoint selection
            seasonality_prior_scale: Strength of seasonality model
            holidays_prior_scale: Strength of holiday components
            country_holidays: Country code for built-in holidays (e.g., 'US', 'UK')
            **kwargs: Additional Prophet parameters
        """
        super().__init__("Prophet", **kwargs)
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet models")
            
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.country_holidays = country_holidays
        
        self.model = None
        self.train_data = None
        self.target_column = None
        self.date_column = None
        self.regressors = []
        
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str,
            regressors: Optional[List[str]] = None,
            **kwargs) -> 'ProphetModel':
        """Fit the Prophet model."""
        start_time = datetime.now()
        
        self.validate_input(train_data, target_column, date_column)
        
        # Store data info
        self.train_data = train_data.copy()
        self.target_column = target_column
        self.date_column = date_column
        self.regressors = regressors or []
        
        try:
            # Prepare data in Prophet format
            df_prophet = train_data[[date_column, target_column]].copy()
            df_prophet.columns = ['ds', 'y']
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            
            # Add regressors if specified
            for regressor in self.regressors:
                if regressor in train_data.columns:
                    df_prophet[regressor] = train_data[regressor]
                else:
                    logger.warning(f"Regressor '{regressor}' not found in data")
            
            # Initialize Prophet model
            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                growth=self.growth,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                **kwargs
            )
            
            # Add country holidays if specified
            if self.country_holidays:
                try:
                    self.model.add_country_holidays(country_name=self.country_holidays)
                    logger.info(f"Added holidays for {self.country_holidays}")
                except Exception as e:
                    logger.warning(f"Could not add holidays for {self.country_holidays}: {str(e)}")
            
            # Add regressors
            for regressor in self.regressors:
                if regressor in df_prophet.columns:
                    self.model.add_regressor(regressor)
                    logger.info(f"Added regressor: {regressor}")
            
            # Fit the model
            self.model.fit(df_prophet)
            self.is_fitted = True
            
            # Store metadata
            self.fit_time = (datetime.now() - start_time).total_seconds()
            self.add_metadata('training_samples', len(df_prophet))
            self.add_metadata('regressors', self.regressors)
            self.add_metadata('country_holidays', self.country_holidays)
            self.add_metadata('seasonality_mode', self.seasonality_mode)
            self.add_metadata('growth', self.growth)
            
            logger.info(f"Prophet model fitted successfully in {self.fit_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
            
        return self
    
    def predict(self, 
                periods: int,
                confidence_level: float = 0.95,
                include_history: bool = False,
                freq: str = 'D',
                **kwargs) -> pd.DataFrame:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        start_time = datetime.now()
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=include_history)
            
            # Generate predictions
            forecast = self.model.predict(future)
            
            # Extract relevant columns
            result_columns = ['ds', 'yhat']
            
            # Add confidence intervals based on confidence level
            if confidence_level == 0.95:
                result_columns.extend(['yhat_lower', 'yhat_upper'])
            elif confidence_level == 0.80:
                # Prophet doesn't have 80% intervals by default, use approximation
                uncertainty = (forecast['yhat_upper'] - forecast['yhat_lower']) * 0.67 / 1.96
                forecast['yhat_lower_80'] = forecast['yhat'] - uncertainty
                forecast['yhat_upper_80'] = forecast['yhat'] + uncertainty
                result_columns.extend(['yhat_lower_80', 'yhat_upper_80'])
            else:
                # Use default 95% intervals
                result_columns.extend(['yhat_lower', 'yhat_upper'])
            
            # Create result dataframe
            result_df = forecast[result_columns].copy()
            
            # Rename columns to match interface
            column_mapping = {
                'ds': self.date_column,
                'yhat': 'forecast',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'upper_bound',
                'yhat_lower_80': 'lower_bound',
                'yhat_upper_80': 'upper_bound'
            }
            
            result_df = result_df.rename(columns=column_mapping)
            
            # If not including history, take only future predictions
            if not include_history:
                result_df = result_df.tail(periods).reset_index(drop=True)
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Prophet predictions generated in {self.prediction_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating Prophet predictions: {str(e)}")
            raise
    
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         freq: str = 'D',
                         **kwargs) -> pd.DataFrame:
        """Generate predictions with exogenous variables (regressors)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        start_time = datetime.now()
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
            
            # Add regressor values for future periods
            if exog_future is not None and self.regressors:
                # Validate exogenous data
                missing_regressors = [reg for reg in self.regressors if reg not in exog_future.columns]
                if missing_regressors:
                    raise ValueError(f"Missing regressors in future data: {missing_regressors}")
                
                if len(exog_future) < periods:
                    raise ValueError(f"Exogenous data has {len(exog_future)} rows, but {periods} periods requested")
                
                # Add regressor values to future dataframe
                for regressor in self.regressors:
                    future[regressor] = exog_future[regressor].iloc[:periods].values
            elif self.regressors:
                logger.warning("Regressors were used in training but no future values provided")
                # Use last known values or zeros
                last_values = self.train_data[self.regressors].iloc[-1]
                for regressor in self.regressors:
                    future[regressor] = last_values[regressor]
            
            # Generate predictions
            forecast = self.model.predict(future)
            
            # Extract relevant columns
            result_columns = ['ds', 'yhat']
            
            # Add confidence intervals
            if confidence_level == 0.95:
                result_columns.extend(['yhat_lower', 'yhat_upper'])
            elif confidence_level == 0.80:
                uncertainty = (forecast['yhat_upper'] - forecast['yhat_lower']) * 0.67 / 1.96
                forecast['yhat_lower_80'] = forecast['yhat'] - uncertainty
                forecast['yhat_upper_80'] = forecast['yhat'] + uncertainty
                result_columns.extend(['yhat_lower_80', 'yhat_upper_80'])
            else:
                result_columns.extend(['yhat_lower', 'yhat_upper'])
            
            # Create result dataframe
            result_df = forecast[result_columns].copy()
            
            # Rename columns
            column_mapping = {
                'ds': self.date_column,
                'yhat': 'forecast',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'upper_bound',
                'yhat_lower_80': 'lower_bound',
                'yhat_upper_80': 'upper_bound'
            }
            
            result_df = result_df.rename(columns=column_mapping)
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Prophet predictions with regressors generated in {self.prediction_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating Prophet predictions with regressors: {str(e)}")
            raise
    
    def add_custom_seasonality(self, 
                              name: str, 
                              period: float, 
                              fourier_order: int,
                              prior_scale: float = 10.0) -> None:
        """
        Add custom seasonality to the model.
        
        Args:
            name: Name of the seasonality component
            period: Period of the seasonality in days
            fourier_order: Number of Fourier terms to use
            prior_scale: Strength of the seasonality
        """
        if self.model is None:
            raise ValueError("Model must be initialized before adding seasonality")
            
        if self.is_fitted:
            logger.warning("Adding seasonality to an already fitted model. You need to refit.")
            
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale
        )
        
        logger.info(f"Added custom seasonality: {name} (period={period}, fourier_order={fourier_order})")
    
    def add_holiday(self, holidays_df: pd.DataFrame) -> None:
        """
        Add custom holidays to the model.
        
        Args:
            holidays_df: DataFrame with 'holiday' and 'ds' columns
        """
        if self.model is None:
            raise ValueError("Model must be initialized before adding holidays")
            
        if self.is_fitted:
            logger.warning("Adding holidays to an already fitted model. You need to refit.")
        
        # Validate holidays dataframe
        required_columns = ['holiday', 'ds']
        if not all(col in holidays_df.columns for col in required_columns):
            raise ValueError(f"Holidays dataframe must have columns: {required_columns}")
        
        self.model.holidays = holidays_df
        logger.info(f"Added {len(holidays_df)} custom holidays")
    
    def cross_validate_model(self, 
                           initial: str = '730 days',
                           period: str = '180 days', 
                           horizon: str = '365 days',
                           parallel: str = 'processes') -> pd.DataFrame:
        """
        Perform cross-validation on the model.
        
        Args:
            initial: Size of the initial training period
            period: Spacing between cutoff dates
            horizon: Size of the forecast horizon
            parallel: Parallelization method
            
        Returns:
            DataFrame with cross-validation results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
            
        try:
            df_cv = cross_validation(
                self.model, 
                initial=initial, 
                period=period, 
                horizon=horizon,
                parallel=parallel
            )
            
            # Calculate performance metrics
            df_metrics = performance_metrics(df_cv)
            
            self.add_metadata('cv_metrics', df_metrics.to_dict())
            logger.info("Cross-validation completed successfully")
            
            return df_cv
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def get_component_plots_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for plotting model components.
        
        Returns:
            Dictionary with component data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting components")
            
        try:
            # Create future dataframe for components
            future = self.model.make_future_dataframe(periods=0, include_history=True)
            forecast = self.model.predict(future)
            
            components = {}
            
            # Trend
            components['trend'] = forecast[['ds', 'trend']].copy()
            
            # Seasonality components
            if self.yearly_seasonality:
                components['yearly'] = forecast[['ds', 'yearly']].copy()
            
            if self.weekly_seasonality:
                components['weekly'] = forecast[['ds', 'weekly']].copy()
            
            if self.daily_seasonality:
                components['daily'] = forecast[['ds', 'daily']].copy()
            
            # Holidays if present
            if 'holidays' in forecast.columns:
                components['holidays'] = forecast[['ds', 'holidays']].copy()
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting component data: {str(e)}")
            raise