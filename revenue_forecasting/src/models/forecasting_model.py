"""
Main ForecastingModel class with factory pattern for different model types.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type, Union
from datetime import datetime
import logging

from .base_model import BaseForecastingModel
from .statistical_models import ARIMAModel, SARIMAXModel
from .ml_models import XGBoostModel, LightGBMModel
from .deep_learning_models import NBEATSModel, TCNModel
from .prophet_model import ProphetModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating forecasting models."""
    
    _models = {
        'arima': ARIMAModel,
        'sarimax': SARIMAXModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'nbeats': NBEATSModel,
        'tcn': TCNModel,
        'prophet': ProphetModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseForecastingModel:
        """
        Create a forecasting model of the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Initialized forecasting model
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available models: {available_models}")
        
        model_class = cls._models[model_type]
        
        try:
            return model_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {str(e)}")
            raise
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseForecastingModel]) -> None:
        """
        Register a new model type.
        
        Args:
            name: Name for the model type
            model_class: Model class that inherits from BaseForecastingModel
        """
        if not issubclass(model_class, BaseForecastingModel):
            raise ValueError("Model class must inherit from BaseForecastingModel")
        
        cls._models[name.lower()] = model_class
        logger.info(f"Registered new model type: {name}")


class ForecastingModel:
    """
    Main forecasting model class that provides a unified interface for different
    forecasting algorithms.
    """
    
    def __init__(self, 
                 model_type: str,
                 **model_params):
        """
        Initialize the forecasting model.
        
        Args:
            model_type: Type of model to use ('arima', 'xgboost', 'prophet', etc.)
            **model_params: Parameters specific to the chosen model type
        """
        self.model_type = model_type.lower()
        self.model_params = model_params
        self.model = None
        self.is_fitted = False
        self.metadata = {}
        
        # Create the specific model instance
        try:
            self.model = ModelFactory.create_model(self.model_type, **model_params)
            logger.info(f"Initialized {self.model_type} forecasting model")
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} model: {str(e)}")
            raise
    
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str = 'date',
            **fit_params) -> 'ForecastingModel':
        """
        Fit the forecasting model to training data.
        
        Args:
            train_data: Training dataset
            target_column: Name of the target variable column
            date_column: Name of the date column
            **fit_params: Additional parameters for fitting
            
        Returns:
            Fitted forecasting model
        """
        try:
            self.model.fit(train_data, target_column, date_column, **fit_params)
            self.is_fitted = True
            
            # Store fitting metadata
            self.metadata.update(self.model.get_metadata())
            
            logger.info(f"{self.model_type} model fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting {self.model_type} model: {str(e)}")
            raise
    
    def predict(self, 
                periods: int,
                confidence_level: float = 0.95,
                include_history: bool = False,
                **predict_params) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            include_history: Whether to include historical data in output
            **predict_params: Additional parameters for prediction
            
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            predictions = self.model.predict(
                periods=periods,
                confidence_level=confidence_level,
                include_history=include_history,
                **predict_params
            )
            
            # Update metadata with prediction info
            self.metadata.update(self.model.get_metadata())
            
            logger.info(f"Generated {periods} forecasts using {self.model_type} model")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions with {self.model_type} model: {str(e)}")
            raise
    
    def predict_with_scenarios(self,
                              periods: int,
                              scenarios: Dict[str, pd.DataFrame],
                              confidence_level: float = 0.95,
                              **predict_params) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for multiple what-if scenarios.
        
        Args:
            periods: Number of periods to forecast
            scenarios: Dictionary of scenario names and their exogenous data
            confidence_level: Confidence level for prediction intervals
            **predict_params: Additional prediction parameters
            
        Returns:
            Dictionary with scenario names as keys and forecast DataFrames as values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        results = {}
        
        for scenario_name, exog_data in scenarios.items():
            try:
                if hasattr(self.model, 'predict_with_exog'):
                    prediction = self.model.predict_with_exog(
                        periods=periods,
                        exog_future=exog_data,
                        confidence_level=confidence_level,
                        **predict_params
                    )
                else:
                    # Fallback to regular prediction if model doesn't support exogenous variables
                    logger.warning(f"{self.model_type} model doesn't support exogenous variables")
                    prediction = self.model.predict(
                        periods=periods,
                        confidence_level=confidence_level,
                        **predict_params
                    )
                
                results[scenario_name] = prediction
                logger.info(f"Generated forecast for scenario: {scenario_name}")
                
            except Exception as e:
                logger.error(f"Error generating prediction for scenario {scenario_name}: {str(e)}")
                results[scenario_name] = None
        
        return results
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if supported by the model.
        
        Returns:
            Dictionary of feature names and importance scores, or None
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        else:
            logger.info(f"{self.model_type} model doesn't support feature importance")
            return None
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns:
            String representation of model summary
        """
        if not self.is_fitted:
            return f"Unfitted {self.model_type} model"
        
        if hasattr(self.model, 'get_model_summary'):
            return self.model.get_model_summary()
        else:
            return str(self.model)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata including training and prediction times.
        
        Returns:
            Dictionary with model metadata
        """
        base_metadata = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        
        if self.model:
            base_metadata.update(self.model.get_metadata())
        
        base_metadata.update(self.metadata)
        return base_metadata
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        try:
            import pickle
            
            model_data = {
                'model_type': self.model_type,
                'model_params': self.model_params,
                'model': self.model,
                'metadata': self.metadata
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ForecastingModel':
        """
        Load a saved model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded forecasting model
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            forecasting_model = cls(
                model_type=model_data['model_type'],
                **model_data['model_params']
            )
            
            # Restore the fitted model and metadata
            forecasting_model.model = model_data['model']
            forecasting_model.is_fitted = True
            forecasting_model.metadata = model_data['metadata']
            
            logger.info(f"Model loaded from {filepath}")
            return forecasting_model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """String representation of the forecasting model."""
        status = "fitted" if self.is_fitted else "unfitted"
        return f"ForecastingModel(type={self.model_type}, status={status})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ForecastingModel(model_type='{self.model_type}', params={self.model_params}, fitted={self.is_fitted})"


# Convenience functions for quick model creation
def create_arima_model(**params) -> ForecastingModel:
    """Create an ARIMA forecasting model."""
    return ForecastingModel('arima', **params)


def create_sarimax_model(**params) -> ForecastingModel:
    """Create a SARIMAX forecasting model."""
    return ForecastingModel('sarimax', **params)


def create_xgboost_model(**params) -> ForecastingModel:
    """Create an XGBoost forecasting model."""
    return ForecastingModel('xgboost', **params)


def create_lightgbm_model(**params) -> ForecastingModel:
    """Create a LightGBM forecasting model."""
    return ForecastingModel('lightgbm', **params)


def create_prophet_model(**params) -> ForecastingModel:
    """Create a Prophet forecasting model."""
    return ForecastingModel('prophet', **params)


def create_nbeats_model(**params) -> ForecastingModel:
    """Create an N-BEATS forecasting model."""
    return ForecastingModel('nbeats', **params)


def create_tcn_model(**params) -> ForecastingModel:
    """Create a TCN forecasting model."""
    return ForecastingModel('tcn', **params)


def get_available_models() -> List[str]:
    """Get list of all available model types."""
    return ModelFactory.get_available_models()