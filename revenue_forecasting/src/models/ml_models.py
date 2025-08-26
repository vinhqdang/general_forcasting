"""
Machine Learning forecasting models (XGBoost, LightGBM, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from .base_model import BaseForecastingModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class XGBoostModel(BaseForecastingModel):
    """XGBoost model for time series forecasting."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 lags: List[int] = None,
                 **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            lags: List of lag periods to use as features
            **kwargs: Additional XGBoost parameters
        """
        super().__init__("XGBoost", **kwargs)
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoost models")
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags or [1, 2, 3, 7, 14, 30]
        
        self.model = None
        self.feature_columns = None
        self.train_data = None
        self.target_column = None
        self.date_column = None
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag and time features."""
        df = data.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        # Create lag features
        for lag in self.lags:
            df[f'lag_{lag}'] = df[self.target_column].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            if window <= len(df):
                df[f'rolling_mean_{window}'] = df[self.target_column].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df[self.target_column].rolling(window=window).std()
        
        # Create time features
        df['day_of_week'] = df[self.date_column].dt.dayofweek
        df['month'] = df[self.date_column].dt.month
        df['quarter'] = df[self.date_column].dt.quarter
        df['year'] = df[self.date_column].dt.year
        df['day_of_year'] = df[self.date_column].dt.dayofyear
        df['is_weekend'] = (df[self.date_column].dt.dayofweek >= 5).astype(int)
        
        return df
    
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str,
            validation_split: float = 0.2,
            **kwargs) -> 'XGBoostModel':
        """Fit the XGBoost model."""
        start_time = datetime.now()
        
        self.validate_input(train_data, target_column, date_column)
        
        # Store data info
        self.train_data = train_data.copy()
        self.target_column = target_column
        self.date_column = date_column
        
        try:
            # Create features
            df_features = self._create_features(train_data)
            
            # Remove rows with NaN values (due to lags)
            df_clean = df_features.dropna()
            
            if len(df_clean) == 0:
                raise ValueError("No data left after creating lag features")
            
            # Prepare features and target
            feature_cols = [col for col in df_clean.columns 
                          if col not in [self.target_column, self.date_column]]
            self.feature_columns = feature_cols
            
            X = df_clean[feature_cols].values
            y = df_clean[self.target_column].values
            
            # Split for validation if requested
            if validation_split > 0:
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                eval_set = [(X_train, y_train), (X_val, y_val)]
            else:
                X_train, y_train = X, y
                eval_set = [(X_train, y_train)]
            
            # Initialize and fit model
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                **kwargs
            )
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            self.is_fitted = True
            
            # Store metadata
            self.fit_time = (datetime.now() - start_time).total_seconds()
            self.add_metadata('training_samples', len(X_train))
            self.add_metadata('feature_count', len(feature_cols))
            self.add_metadata('lags_used', self.lags)
            
            if validation_split > 0:
                val_pred = self.model.predict(X_val)
                self.add_metadata('validation_mae', mean_absolute_error(y_val, val_pred))
                self.add_metadata('validation_rmse', np.sqrt(mean_squared_error(y_val, val_pred)))
            
            logger.info(f"XGBoost model fitted successfully in {self.fit_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting XGBoost model: {str(e)}")
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
            # Get the last part of training data for creating initial features
            df_full = self._create_features(self.train_data)
            last_data = df_full.tail(max(self.lags) + periods).copy()
            
            # Generate predictions step by step
            predictions = []
            last_date = pd.to_datetime(self.train_data[self.date_column]).max()
            freq = pd.infer_freq(pd.to_datetime(self.train_data[self.date_column]))
            
            for i in range(periods):
                # Current prediction date
                pred_date = last_date + pd.Timedelta((i + 1) * (freq or 'D'))
                
                # Create features for current prediction
                current_data = last_data.iloc[-(max(self.lags) + periods - i):].copy()
                
                # Update time features for prediction date
                pred_row = pd.DataFrame({
                    self.date_column: [pred_date],
                    self.target_column: [np.nan]  # Will be filled with prediction
                })
                
                for col in current_data.columns:
                    if col not in [self.date_column, self.target_column]:
                        pred_row[col] = np.nan
                
                # Add time features
                pred_row['day_of_week'] = pred_date.dayofweek
                pred_row['month'] = pred_date.month
                pred_row['quarter'] = pred_date.quarter
                pred_row['year'] = pred_date.year
                pred_row['day_of_year'] = pred_date.dayofyear
                pred_row['is_weekend'] = int(pred_date.dayofweek >= 5)
                
                # Create lag features
                for lag in self.lags:
                    if len(current_data) >= lag:
                        pred_row[f'lag_{lag}'] = current_data[self.target_column].iloc[-lag]
                
                # Create rolling features
                for window in [7, 14, 30]:
                    if len(current_data) >= window:
                        recent_values = current_data[self.target_column].tail(window)
                        pred_row[f'rolling_mean_{window}'] = recent_values.mean()
                        pred_row[f'rolling_std_{window}'] = recent_values.std()
                
                # Make prediction
                X_pred = pred_row[self.feature_columns].values
                if not np.isnan(X_pred).any():
                    pred_value = self.model.predict(X_pred)[0]
                else:
                    # Handle missing features by using simple forecasting
                    pred_value = current_data[self.target_column].tail(7).mean()
                
                predictions.append(pred_value)
                
                # Update data with prediction for next iteration
                pred_row[self.target_column] = pred_value
                last_data = pd.concat([last_data, pred_row], ignore_index=True)
            
            # Create prediction DataFrame
            future_dates = pd.date_range(start=last_date + pd.Timedelta(freq or 'D'), 
                                       periods=periods, freq=freq or 'D')
            
            # For ML models, we don't have true confidence intervals
            # We can estimate them using training error or bootstrapping
            pred_std = np.std(predictions) if len(predictions) > 1 else np.abs(np.mean(predictions)) * 0.1
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # for 99%
            
            result_df = pd.DataFrame({
                self.date_column: future_dates,
                'forecast': predictions,
                'lower_bound': np.array(predictions) - z_score * pred_std,
                'upper_bound': np.array(predictions) + z_score * pred_std
            })
            
            # Include history if requested
            if include_history:
                historical = self.train_data[[self.date_column, self.target_column]].copy()
                historical['forecast'] = historical[self.target_column]
                historical['lower_bound'] = historical[self.target_column]
                historical['upper_bound'] = historical[self.target_column]
                result_df = pd.concat([historical, result_df], ignore_index=True)
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"XGBoost predictions generated in {self.prediction_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating XGBoost predictions: {str(e)}")
            raise
    
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         **kwargs) -> pd.DataFrame:
        """Generate predictions with exogenous variables."""
        # For now, use regular predict method
        # Can be extended to include exogenous variables as features
        return self.predict(periods, confidence_level, **kwargs)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted or self.model is None:
            return None
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))


class LightGBMModel(BaseForecastingModel):
    """LightGBM model for time series forecasting."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 learning_rate: float = 0.1,
                 lags: List[int] = None,
                 **kwargs):
        """
        Initialize LightGBM model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            lags: List of lag periods to use as features
            **kwargs: Additional LightGBM parameters
        """
        super().__init__("LightGBM", **kwargs)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required for LightGBM models")
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags or [1, 2, 3, 7, 14, 30]
        
        self.model = None
        self.feature_columns = None
        self.train_data = None
        self.target_column = None
        self.date_column = None
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lag and time features (same as XGBoost)."""
        df = data.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        # Create lag features
        for lag in self.lags:
            df[f'lag_{lag}'] = df[self.target_column].shift(lag)
        
        # Create rolling statistics
        for window in [7, 14, 30]:
            if window <= len(df):
                df[f'rolling_mean_{window}'] = df[self.target_column].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df[self.target_column].rolling(window=window).std()
        
        # Create time features
        df['day_of_week'] = df[self.date_column].dt.dayofweek
        df['month'] = df[self.date_column].dt.month
        df['quarter'] = df[self.date_column].dt.quarter
        df['year'] = df[self.date_column].dt.year
        df['day_of_year'] = df[self.date_column].dt.dayofyear
        df['is_weekend'] = (df[self.date_column].dt.dayofweek >= 5).astype(int)
        
        return df
    
    def fit(self, 
            train_data: pd.DataFrame,
            target_column: str,
            date_column: str,
            validation_split: float = 0.2,
            **kwargs) -> 'LightGBMModel':
        """Fit the LightGBM model."""
        start_time = datetime.now()
        
        self.validate_input(train_data, target_column, date_column)
        
        # Store data info
        self.train_data = train_data.copy()
        self.target_column = target_column
        self.date_column = date_column
        
        try:
            # Create features
            df_features = self._create_features(train_data)
            
            # Remove rows with NaN values (due to lags)
            df_clean = df_features.dropna()
            
            if len(df_clean) == 0:
                raise ValueError("No data left after creating lag features")
            
            # Prepare features and target
            feature_cols = [col for col in df_clean.columns 
                          if col not in [self.target_column, self.date_column]]
            self.feature_columns = feature_cols
            
            X = df_clean[feature_cols].values
            y = df_clean[self.target_column].values
            
            # Split for validation if requested
            if validation_split > 0:
                split_idx = int(len(X) * (1 - validation_split))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                eval_set = [(X_val, y_val)]
            else:
                X_train, y_train = X, y
                eval_set = None
            
            # Initialize and fit model
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbose=-1,
                **kwargs
            )
            
            if eval_set:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
                )
            else:
                self.model.fit(X_train, y_train)
            
            self.is_fitted = True
            
            # Store metadata
            self.fit_time = (datetime.now() - start_time).total_seconds()
            self.add_metadata('training_samples', len(X_train))
            self.add_metadata('feature_count', len(feature_cols))
            self.add_metadata('lags_used', self.lags)
            
            if validation_split > 0:
                val_pred = self.model.predict(X_val)
                self.add_metadata('validation_mae', mean_absolute_error(y_val, val_pred))
                self.add_metadata('validation_rmse', np.sqrt(mean_squared_error(y_val, val_pred)))
            
            logger.info(f"LightGBM model fitted successfully in {self.fit_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fitting LightGBM model: {str(e)}")
            raise
            
        return self
    
    def predict(self, 
                periods: int,
                confidence_level: float = 0.95,
                include_history: bool = False,
                **kwargs) -> pd.DataFrame:
        """Generate predictions (same logic as XGBoost)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        start_time = datetime.now()
        
        try:
            # Get the last part of training data for creating initial features
            df_full = self._create_features(self.train_data)
            last_data = df_full.tail(max(self.lags) + periods).copy()
            
            # Generate predictions step by step
            predictions = []
            last_date = pd.to_datetime(self.train_data[self.date_column]).max()
            freq = pd.infer_freq(pd.to_datetime(self.train_data[self.date_column]))
            
            for i in range(periods):
                # Current prediction date
                pred_date = last_date + pd.Timedelta((i + 1) * (freq or 'D'))
                
                # Create features for current prediction
                current_data = last_data.iloc[-(max(self.lags) + periods - i):].copy()
                
                # Update time features for prediction date
                pred_row = pd.DataFrame({
                    self.date_column: [pred_date],
                    self.target_column: [np.nan]  # Will be filled with prediction
                })
                
                for col in current_data.columns:
                    if col not in [self.date_column, self.target_column]:
                        pred_row[col] = np.nan
                
                # Add time features
                pred_row['day_of_week'] = pred_date.dayofweek
                pred_row['month'] = pred_date.month
                pred_row['quarter'] = pred_date.quarter
                pred_row['year'] = pred_date.year
                pred_row['day_of_year'] = pred_date.dayofyear
                pred_row['is_weekend'] = int(pred_date.dayofweek >= 5)
                
                # Create lag features
                for lag in self.lags:
                    if len(current_data) >= lag:
                        pred_row[f'lag_{lag}'] = current_data[self.target_column].iloc[-lag]
                
                # Create rolling features
                for window in [7, 14, 30]:
                    if len(current_data) >= window:
                        recent_values = current_data[self.target_column].tail(window)
                        pred_row[f'rolling_mean_{window}'] = recent_values.mean()
                        pred_row[f'rolling_std_{window}'] = recent_values.std()
                
                # Make prediction
                X_pred = pred_row[self.feature_columns].values
                if not np.isnan(X_pred).any():
                    pred_value = self.model.predict(X_pred)[0]
                else:
                    # Handle missing features by using simple forecasting
                    pred_value = current_data[self.target_column].tail(7).mean()
                
                predictions.append(pred_value)
                
                # Update data with prediction for next iteration
                pred_row[self.target_column] = pred_value
                last_data = pd.concat([last_data, pred_row], ignore_index=True)
            
            # Create prediction DataFrame
            future_dates = pd.date_range(start=last_date + pd.Timedelta(freq or 'D'), 
                                       periods=periods, freq=freq or 'D')
            
            # Estimate confidence intervals
            pred_std = np.std(predictions) if len(predictions) > 1 else np.abs(np.mean(predictions)) * 0.1
            z_score = 1.96 if confidence_level == 0.95 else 2.576
            
            result_df = pd.DataFrame({
                self.date_column: future_dates,
                'forecast': predictions,
                'lower_bound': np.array(predictions) - z_score * pred_std,
                'upper_bound': np.array(predictions) + z_score * pred_std
            })
            
            # Include history if requested
            if include_history:
                historical = self.train_data[[self.date_column, self.target_column]].copy()
                historical['forecast'] = historical[self.target_column]
                historical['lower_bound'] = historical[self.target_column]
                historical['upper_bound'] = historical[self.target_column]
                result_df = pd.concat([historical, result_df], ignore_index=True)
            
            self.prediction_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"LightGBM predictions generated in {self.prediction_time:.2f} seconds")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating LightGBM predictions: {str(e)}")
            raise
    
    def predict_with_exog(self,
                         periods: int,
                         exog_future: Optional[pd.DataFrame] = None,
                         confidence_level: float = 0.95,
                         **kwargs) -> pd.DataFrame:
        """Generate predictions with exogenous variables."""
        return self.predict(periods, confidence_level, **kwargs)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_fitted or self.model is None:
            return None
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))