"""
Unit tests for DataLoader class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path

import sys
sys.path.append('src')

from data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000, 100, len(dates)),
            'extra_col': np.random.randn(len(dates))
        })
        return data
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance."""
        return DataLoader(
            date_column='date',
            target_variable='revenue',  # Updated parameter name
            frequency='D'
        )
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader(
            date_column='timestamp',
            target_variable='sales',  # Updated parameter name
            frequency='H'
        )
        
        assert loader.date_column == 'timestamp'
        assert loader.target_variable == 'sales'
        assert loader.target_column == 'sales'  # Backward compatibility
        assert loader.frequency == 'H'
        assert loader.data is None
        assert isinstance(loader.metadata, dict)
        assert isinstance(loader.feature_list, list)
    
    def test_load_from_dataframe(self, data_loader, sample_data):
        """Test loading data from DataFrame."""
        result = data_loader.load_from_dataframe(sample_data)
        
        assert result is not None
        assert len(result) == len(sample_data)
        assert list(result.columns) == list(sample_data.columns)
        assert data_loader.data is not None
        assert 'load_time' in data_loader.metadata
        assert 'original_shape' in data_loader.metadata
    
    def test_load_from_csv_file(self, data_loader, sample_data):
        """Test loading data from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            result = data_loader.load_from_file(temp_file)
            
            assert result is not None
            assert len(result) == len(sample_data)
            assert 'date' in result.columns
            assert 'revenue' in result.columns
            assert 'file_path' in data_loader.metadata
            
        finally:
            os.unlink(temp_file)
    
    def test_load_from_nonexistent_file(self, data_loader):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_from_file('nonexistent_file.csv')
    
    def test_validate_data_success(self, data_loader, sample_data):
        """Test successful data validation."""
        data_loader.load_from_dataframe(sample_data)
        validation = data_loader.validate_data()
        
        assert validation['has_date_column'] is True
        assert validation['has_target_column'] is True
        assert validation['data_points'] == len(sample_data)
        assert isinstance(validation['missing_values'], dict)
        assert 'date_range' in validation
    
    def test_validate_data_missing_columns(self, data_loader):
        """Test validation with missing columns."""
        bad_data = pd.DataFrame({'wrong_col': [1, 2, 3]})
        data_loader.load_from_dataframe(bad_data)
        
        validation = data_loader.validate_data()
        assert validation['has_date_column'] is False
        assert validation['has_target_column'] is False
    
    def test_validate_data_no_data(self, data_loader):
        """Test validation without loaded data."""
        with pytest.raises(ValueError, match="No data loaded"):
            data_loader.validate_data()
    
    def test_preprocess_basic(self, data_loader, sample_data):
        """Test basic data preprocessing."""
        data_loader.load_from_dataframe(sample_data)
        result = data_loader.preprocess()
        
        assert result is not None
        assert len(result) <= len(sample_data)  # May remove duplicates
        assert 'preprocessing_time' in data_loader.metadata
        assert 'preprocessed_shape' in data_loader.metadata
    
    def test_preprocess_with_duplicates(self, data_loader):
        """Test preprocessing with duplicate dates."""
        dates = ['2023-01-01', '2023-01-01', '2023-01-02']
        data = pd.DataFrame({
            'date': dates,
            'revenue': [100, 200, 300]
        })
        
        data_loader.load_from_dataframe(data)
        result = data_loader.preprocess(remove_duplicates=True)
        
        assert len(result) == 2  # One duplicate removed
    
    def test_preprocess_with_missing_values(self, data_loader):
        """Test preprocessing with missing values."""
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'revenue': [100, np.nan, 300, np.nan, 500]
        })
        
        data_loader.load_from_dataframe(data)
        result = data_loader.preprocess(fill_missing='interpolate')
        
        assert result['revenue'].isnull().sum() == 0
    
    def test_create_time_features(self, data_loader, sample_data):
        """Test time feature creation."""
        data_loader.load_from_dataframe(sample_data)
        result = data_loader.create_time_features()
        
        expected_features = [
            'year', 'month', 'day', 'day_of_week', 
            'day_of_year', 'week_of_year', 'quarter', 'is_weekend'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_create_time_features_no_data(self, data_loader):
        """Test time feature creation without data."""
        with pytest.raises(ValueError, match="No data loaded"):
            data_loader.create_time_features()
    
    def test_create_time_features_no_date_column(self, data_loader):
        """Test time feature creation without date column."""
        data = pd.DataFrame({'revenue': [100, 200, 300]})
        data_loader.load_from_dataframe(data)
        
        with pytest.raises(ValueError, match="Date column .* not found"):
            data_loader.create_time_features()
    
    def test_get_data_info(self, data_loader, sample_data):
        """Test getting data information."""
        data_loader.load_from_dataframe(sample_data)
        info = data_loader.get_data_info()
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'memory_usage' in info
        assert 'metadata' in info
        assert 'target_statistics' in info
    
    def test_get_data_info_no_data(self, data_loader):
        """Test getting info without data."""
        info = data_loader.get_data_info()
        assert 'error' in info
    
    def test_add_metadata(self, data_loader):
        """Test adding metadata."""
        data_loader.add_metadata('test_key', 'test_value')
        assert data_loader.metadata['test_key'] == 'test_value'
    
    def test_get_data(self, data_loader, sample_data):
        """Test getting current data."""
        assert data_loader.get_data() is None
        
        data_loader.load_from_dataframe(sample_data)
        result = data_loader.get_data()
        
        assert result is not None
        assert len(result) == len(sample_data)
    
    def test_export_data_csv(self, data_loader, sample_data):
        """Test exporting data to CSV."""
        data_loader.load_from_dataframe(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            data_loader.export_data(temp_file, 'csv')
            
            # Verify file was created and has correct content
            exported = pd.read_csv(temp_file)
            assert len(exported) == len(sample_data)
            assert list(exported.columns) == list(sample_data.columns)
            
        finally:
            os.unlink(temp_file)
    
    def test_export_data_no_data(self, data_loader):
        """Test exporting without data."""
        with pytest.raises(ValueError, match="No data to export"):
            data_loader.export_data('test.csv')
    
    def test_different_frequencies(self):
        """Test different frequency settings."""
        frequencies = ['D', 'H', 'W', 'M']
        
        for freq in frequencies:
            loader = DataLoader(frequency=freq, target_variable='test_target')
            assert loader.frequency == freq
            assert loader.target_variable == 'test_target'
    
    def test_chain_operations(self, sample_data):
        """Test chaining multiple operations."""
        loader = DataLoader(target_variable='revenue')  # Specify target
        
        # Load data first
        loader.load_from_dataframe(sample_data)
        
        # Process data
        processed = loader.preprocess()
        assert processed is not None
        
        # Add time features
        with_features = loader.create_time_features()
        
        assert with_features is not None
        assert len(with_features.columns) > len(sample_data.columns)  # Added time features


class TestDataLoaderIntegration:
    """Integration tests for DataLoader."""
    
    def test_complete_workflow(self):
        """Test complete data loading and preprocessing workflow."""
        # Create test data
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000, 100, len(dates)) + 
                      50 * np.sin(2 * np.pi * np.arange(len(dates)) / 7),  # Weekly pattern
            'promotion': np.random.choice([0, 1], len(dates), p=[0.8, 0.2])
        })
        
        # Add some missing values and duplicates
        data.iloc[10, data.columns.get_loc('revenue')] = np.nan
        data.iloc[20, data.columns.get_loc('revenue')] = np.nan
        
        # Duplicate a row
        data = pd.concat([data, data.iloc[[5]]], ignore_index=True)
        
        # Create loader and process
        loader = DataLoader(
            date_column='date',
            target_variable='revenue',
            frequency='D'
        )
        
        # Load and validate
        loader.load_from_dataframe(data)
        validation = loader.validate_data()
        
        assert validation['has_date_column']
        assert validation['has_target_column']
        assert validation['duplicate_dates'] > 0
        assert validation['missing_values']['revenue'] > 0
        
        # Preprocess
        processed = loader.preprocess(
            sort_by_date=True,
            remove_duplicates=True,
            fill_missing='interpolate'
        )
        
        assert processed['revenue'].isnull().sum() == 0
        assert len(processed) == len(dates)  # Duplicates removed
        
        # Add time features
        with_features = loader.create_time_features()
        
        expected_features = [
            'year', 'month', 'day', 'day_of_week',
            'day_of_year', 'week_of_year', 'quarter', 'is_weekend'
        ]
        
        for feature in expected_features:
            assert feature in with_features.columns
        
        # Get final info
        info = loader.get_data_info()
        assert info['shape'][0] == len(dates)
        assert 'target_statistics' in info
        
        # Verify metadata was collected
        metadata = loader.metadata
        assert 'load_time' in metadata
        assert 'preprocessing_time' in metadata
        assert 'original_shape' in metadata
        assert 'preprocessed_shape' in metadata


if __name__ == '__main__':
    pytest.main([__file__])