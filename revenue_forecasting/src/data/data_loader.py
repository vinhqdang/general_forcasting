"""
DataLoader class for loading and preprocessing dataframes for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader class for loading and preprocessing time series data.
    
    Supports multiple file formats and provides data validation and preprocessing
    capabilities for time series forecasting.
    """
    
    def __init__(self, 
                 date_column: str = 'date',
                 target_column: str = 'revenue',
                 frequency: str = 'D'):
        """
        Initialize DataLoader.
        
        Args:
            date_column: Name of the datetime column
            target_column: Name of the target variable column
            frequency: Data frequency (D=daily, H=hourly, W=weekly, M=monthly)
        """
        self.date_column = date_column
        self.target_column = target_column
        self.frequency = frequency
        self.data = None
        self.metadata = {}
        
    def load_from_file(self, 
                      file_path: Union[str, Path],
                      **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
            self.data = data
            self.metadata['file_path'] = str(file_path)
            self.metadata['load_time'] = datetime.now()
            self.metadata['original_shape'] = data.shape
            
            logger.info(f"Successfully loaded data with shape {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Load data from existing DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Loaded DataFrame
        """
        self.data = df.copy()
        self.metadata['load_time'] = datetime.now()
        self.metadata['original_shape'] = df.shape
        
        logger.info(f"Loaded data from DataFrame with shape {df.shape}")
        return self.data
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data for time series forecasting.
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        validation_results = {
            'has_date_column': self.date_column in self.data.columns,
            'has_target_column': self.target_column in self.data.columns,
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_dates': 0,
            'date_range': None,
            'data_points': len(self.data),
            'columns': list(self.data.columns)
        }
        
        if validation_results['has_date_column']:
            try:
                date_series = pd.to_datetime(self.data[self.date_column])
                validation_results['date_range'] = {
                    'start': date_series.min(),
                    'end': date_series.max()
                }
                validation_results['duplicate_dates'] = date_series.duplicated().sum()
            except Exception as e:
                logger.warning(f"Error processing date column: {str(e)}")
                validation_results['date_conversion_error'] = str(e)
        
        logger.info(f"Data validation completed: {validation_results}")
        return validation_results
    
    def preprocess(self, 
                  sort_by_date: bool = True,
                  remove_duplicates: bool = True,
                  fill_missing: Optional[str] = 'interpolate',
                  date_format: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess the data for time series forecasting.
        
        Args:
            sort_by_date: Whether to sort data by date
            remove_duplicates: Whether to remove duplicate dates
            fill_missing: Method to fill missing values ('interpolate', 'forward', 'backward', None)
            date_format: Format string for date parsing
            
        Returns:
            Preprocessed DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        df = self.data.copy()
        logger.info("Starting data preprocessing")
        
        # Convert date column to datetime
        if self.date_column in df.columns:
            if date_format:
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=date_format)
            else:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        if sort_by_date and self.date_column in df.columns:
            df = df.sort_values(self.date_column).reset_index(drop=True)
            logger.info("Data sorted by date")
        
        # Remove duplicates
        if remove_duplicates and self.date_column in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=[self.date_column]).reset_index(drop=True)
            removed = initial_len - len(df)
            if removed > 0:
                logger.info(f"Removed {removed} duplicate records")
        
        # Fill missing values
        if fill_missing and self.target_column in df.columns:
            missing_before = df[self.target_column].isnull().sum()
            
            if fill_missing == 'interpolate':
                df[self.target_column] = df[self.target_column].interpolate()
            elif fill_missing == 'forward':
                df[self.target_column] = df[self.target_column].fillna(method='ffill')
            elif fill_missing == 'backward':
                df[self.target_column] = df[self.target_column].fillna(method='bfill')
            
            missing_after = df[self.target_column].isnull().sum()
            if missing_before > missing_after:
                logger.info(f"Filled {missing_before - missing_after} missing values")
        
        self.data = df
        self.metadata['preprocessing_time'] = datetime.now()
        self.metadata['preprocessed_shape'] = df.shape
        
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        return df
    
    def create_time_features(self) -> pd.DataFrame:
        """
        Create time-based features from the date column.
        
        Returns:
            DataFrame with additional time features
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        if self.date_column not in self.data.columns:
            raise ValueError(f"Date column '{self.date_column}' not found")
            
        df = self.data.copy()
        date_series = pd.to_datetime(df[self.date_column])
        
        # Create time features
        df['year'] = date_series.dt.year
        df['month'] = date_series.dt.month
        df['day'] = date_series.dt.day
        df['day_of_week'] = date_series.dt.dayofweek
        df['day_of_year'] = date_series.dt.dayofyear
        df['week_of_year'] = date_series.dt.isocalendar().week
        df['quarter'] = date_series.dt.quarter
        df['is_weekend'] = date_series.dt.dayofweek.isin([5, 6]).astype(int)
        
        # Frequency-specific features
        if self.frequency in ['H', 'h']:
            df['hour'] = date_series.dt.hour
            df['is_business_hour'] = date_series.dt.hour.between(9, 17).astype(int)
        
        if self.frequency in ['T', 'min']:
            df['hour'] = date_series.dt.hour
            df['minute'] = date_series.dt.minute
        
        self.data = df
        logger.info("Time features created successfully")
        return df
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'metadata': self.metadata
        }
        
        if self.target_column in self.data.columns:
            target_stats = self.data[self.target_column].describe()
            info['target_statistics'] = target_stats.to_dict()
        
        return info
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the data loader.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        logger.info(f"Added metadata: {key}")
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the current data.
        
        Returns:
            Current DataFrame or None if no data loaded
        """
        return self.data
    
    def export_data(self, 
                   file_path: Union[str, Path],
                   format_type: str = 'csv',
                   **kwargs) -> None:
        """
        Export the current data to file.
        
        Args:
            file_path: Output file path
            format_type: Export format ('csv', 'excel', 'parquet', 'json')
            **kwargs: Additional arguments for export functions
        """
        if self.data is None:
            raise ValueError("No data to export")
            
        file_path = Path(file_path)
        
        try:
            if format_type.lower() == 'csv':
                self.data.to_csv(file_path, index=False, **kwargs)
            elif format_type.lower() == 'excel':
                self.data.to_excel(file_path, index=False, **kwargs)
            elif format_type.lower() == 'parquet':
                self.data.to_parquet(file_path, **kwargs)
            elif format_type.lower() == 'json':
                self.data.to_json(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
            logger.info(f"Data exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise