"""
Base class for all forecasting models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseForecastingModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the interface that all forecasting models must implement.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the base forecasting model.
        
        Args:
            name: Name of the model
            **kwargs: Additional model parameters
        """
        self.name = name
        self.params = kwargs
        self.is_fitted = False
        self.fit_time = None
        self.prediction_time = None
        self.metadata = {}
        
    @abstractmethod
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str,
            **kwargs) -> 'BaseForecastingModel':
        """
        Fit the model to training data.
        
        Args:
            train_data: Training dataset
            target_column: Name of the target variable
            date_column: Name of the date column
            **kwargs: Additional fitting parameters
            
        Returns:
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, 
                periods: int,
                confidence_level: float = 0.95,
                include_history: bool = False,
                **kwargs) -> pd.DataFrame:
        """
        Generate predictions.
        
        Args:
            periods: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            include_history: Whether to include historical data in output
            **kwargs: Additional prediction parameters
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        pass
    
    @abstractmethod
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         **kwargs) -> pd.DataFrame:
        """
        Generate predictions with exogenous variables.
        
        Args:
            periods: Number of periods to forecast
            exog_future: Future values of exogenous variables
            confidence_level: Confidence level for prediction intervals
            **kwargs: Additional prediction parameters
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        return None
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.params.copy()
    
    def set_params(self, **params) -> 'BaseForecastingModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Model instance
        """
        self.params.update(params)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata including training and prediction times.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'fit_time': self.fit_time,
            'prediction_time': self.prediction_time,
            'params': self.params,
            **self.metadata
        }
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the model.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def validate_input(self, 
                      data: pd.DataFrame,
                      target_column: str,
                      date_column: str) -> None:
        """
        Validate input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target variable
            date_column: Name of the date column
            
        Raises:
            ValueError: If validation fails
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
            
        if data[target_column].isnull().all():
            raise ValueError("Target column contains only null values")
            
        try:
            pd.to_datetime(data[date_column])
        except Exception as e:
            raise ValueError(f"Date column cannot be converted to datetime: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"