"""
Unit tests for ForecastingModel and ModelFactory.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
sys.path.append('src')

from models.forecasting_model import (
    ForecastingModel, 
    ModelFactory,
    create_arima_model,
    create_xgboost_model,
    get_available_models
)
from models.base_model import BaseForecastingModel


class MockModel(BaseForecastingModel):
    """Mock model for testing."""
    
    def __init__(self, test_param=None, **kwargs):
        super().__init__("MockModel", **kwargs)
        self.test_param = test_param
        
    def fit(self, train_data, target_variable, date_column='date', feature_list=None, **kwargs):
        self.is_fitted = True
        self.fit_time = 0.1
        return self
        
    def predict(self, periods, confidence_level=0.95, include_history=False, **kwargs):
        dates = pd.date_range('2023-01-01', periods=periods, freq='D')
        return pd.DataFrame({
            'date': dates,
            'forecast': np.random.randn(periods) * 100 + 1000,
            'lower_bound': np.random.randn(periods) * 100 + 900,
            'upper_bound': np.random.randn(periods) * 100 + 1100
        })
        
    def predict_with_exog(self, periods, exog_future=None, confidence_level=0.95, **kwargs):
        return self.predict(periods, confidence_level, **kwargs)


class TestModelFactory:
    """Test cases for ModelFactory class."""
    
    def test_register_custom_model(self):
        """Test registering a custom model."""
        ModelFactory.register_model('mock', MockModel)
        
        assert 'mock' in ModelFactory.get_available_models()
        
        # Test creating the custom model
        model = ModelFactory.create_model('mock', test_param='test_value')
        assert isinstance(model, MockModel)
        assert model.test_param == 'test_value'
    
    def test_create_model_case_insensitive(self):
        """Test model creation is case insensitive."""
        ModelFactory.register_model('mock', MockModel)
        
        model_lower = ModelFactory.create_model('mock')
        model_upper = ModelFactory.create_model('MOCK')
        
        assert type(model_lower) == type(model_upper)
    
    def test_create_unknown_model(self):
        """Test creating unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model('nonexistent_model')
    
    def test_get_available_models(self):
        """Test getting available models."""
        available = ModelFactory.get_available_models()
        
        assert isinstance(available, list)
        assert len(available) > 0
        
        # Should include built-in models
        expected_models = ['arima', 'sarimax', 'xgboost', 'lightgbm', 'prophet']
        for model_type in expected_models:
            assert model_type in available
    
    def test_register_invalid_model_class(self):
        """Test registering invalid model class."""
        class NotAForecastingModel:
            pass
        
        with pytest.raises(ValueError, match="must inherit from BaseForecastingModel"):
            ModelFactory.register_model('invalid', NotAForecastingModel)


class TestForecastingModel:
    """Test cases for ForecastingModel class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000, 100, len(dates)),
            'feature1': np.random.randn(len(dates))
        })
    
    def test_init_with_mock_model(self):
        """Test initialization with mock model."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock', test_param='test_value')
        
        assert model.model_type == 'mock'
        assert model.model_params['test_param'] == 'test_value'
        assert not model.is_fitted
        assert isinstance(model.model, MockModel)
    
    def test_init_case_insensitive(self):
        """Test initialization is case insensitive."""
        ModelFactory.register_model('mock', MockModel)
        
        model_lower = ForecastingModel('mock')
        model_upper = ForecastingModel('MOCK')
        
        assert model_lower.model_type == model_upper.model_type
    
    def test_fit(self, sample_data):
        """Test model fitting."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        result = model.fit(sample_data, 'revenue', 'date')
        
        assert result is model  # Should return self
        assert model.is_fitted
        assert 'fit_time' in model.metadata
    
    def test_predict_unfitted_model(self):
        """Test prediction with unfitted model raises error."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(10)
    
    def test_predict_fitted_model(self, sample_data):
        """Test prediction with fitted model."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        model.fit(sample_data, 'revenue', 'date')
        
        predictions = model.predict(10)
        
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 10
        assert 'date' in predictions.columns
        assert 'forecast' in predictions.columns
        assert 'lower_bound' in predictions.columns
        assert 'upper_bound' in predictions.columns
    
    def test_predict_with_scenarios(self, sample_data):
        """Test prediction with multiple scenarios."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        model.fit(sample_data, 'revenue', 'date')
        
        scenarios = {
            'baseline': pd.DataFrame({'feature1': [0, 0, 0]}),
            'high_promo': pd.DataFrame({'feature1': [1, 1, 1]})
        }
        
        results = model.predict_with_scenarios(3, scenarios)
        
        assert isinstance(results, dict)
        assert 'baseline' in results
        assert 'high_promo' in results
        assert isinstance(results['baseline'], pd.DataFrame)
        assert isinstance(results['high_promo'], pd.DataFrame)
    
    def test_get_feature_importance(self, sample_data):
        """Test getting feature importance."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        model.fit(sample_data, 'revenue', 'date')
        
        # Mock model doesn't implement feature importance
        importance = model.get_feature_importance()
        assert importance is None
    
    def test_get_model_summary(self, sample_data):
        """Test getting model summary."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        
        # Unfitted model
        summary = model.get_model_summary()
        assert 'Unfitted' in summary
        
        # Fitted model
        model.fit(sample_data, 'revenue', 'date')
        summary = model.get_model_summary()
        assert isinstance(summary, str)
    
    def test_get_metadata(self, sample_data):
        """Test getting model metadata."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock', test_param='test')
        metadata = model.get_metadata()
        
        assert 'model_type' in metadata
        assert 'model_params' in metadata
        assert 'is_fitted' in metadata
        assert metadata['model_type'] == 'mock'
        assert metadata['model_params']['test_param'] == 'test'
        assert metadata['is_fitted'] is False
        
        # After fitting
        model.fit(sample_data, 'revenue', 'date')
        metadata = model.get_metadata()
        assert metadata['is_fitted'] is True
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        ModelFactory.register_model('mock', MockModel)
        
        # Create and fit model
        original_model = ForecastingModel('mock', test_param='test')
        original_model.fit(sample_data, 'revenue', 'date')
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save model
            original_model.save_model(temp_file)
            
            # Load model
            loaded_model = ForecastingModel.load_model(temp_file)
            
            assert loaded_model.model_type == original_model.model_type
            assert loaded_model.model_params == original_model.model_params
            assert loaded_model.is_fitted == original_model.is_fitted
            
            # Test that loaded model can make predictions
            predictions = loaded_model.predict(5)
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == 5
            
        finally:
            import os
            os.unlink(temp_file)
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model raises error."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock')
        
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save_model('test.pkl')
    
    def test_string_representations(self):
        """Test string representations of the model."""
        ModelFactory.register_model('mock', MockModel)
        
        model = ForecastingModel('mock', test_param='test')
        
        str_repr = str(model)
        assert 'mock' in str_repr
        assert 'unfitted' in str_repr.lower()
        
        detailed_repr = repr(model)
        assert 'mock' in detailed_repr
        assert 'test_param' in detailed_repr


class TestConvenienceFunctions:
    """Test convenience functions for model creation."""
    
    def test_create_arima_model(self):
        """Test creating ARIMA model via convenience function."""
        # Test successful creation of ARIMA model
        model = create_arima_model(order=(1, 1, 1))
        assert model.model_type == 'arima'
        assert model.model_params['order'] == (1, 1, 1)
    
    def test_create_xgboost_model(self):
        """Test creating XGBoost model via convenience function."""
        # Since xgboost is not installed, this should raise ImportError
        with pytest.raises(ImportError, match="xgboost is required"):
            create_xgboost_model(n_estimators=100)
    
    def test_get_available_models_function(self):
        """Test get_available_models function."""
        available = get_available_models()
        
        assert isinstance(available, list)
        assert len(available) > 0


class TestForecastingModelIntegration:
    """Integration tests for ForecastingModel."""
    
    @pytest.fixture
    def realistic_data(self):
        """Create realistic time series data."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Create realistic revenue with trend and seasonality
        t = np.arange(len(dates))
        trend = 1000 + 2 * t
        seasonal = 200 * np.sin(2 * np.pi * t / 365.25)  # Yearly
        weekly = 100 * np.sin(2 * np.pi * t / 7)  # Weekly
        noise = np.random.normal(0, 50, len(dates))
        
        revenue = trend + seasonal + weekly + noise
        
        return pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
    
    def test_complete_workflow_mock(self, realistic_data):
        """Test complete workflow with mock model."""
        ModelFactory.register_model('mock', MockModel)
        
        # Split data
        train_data = realistic_data[:-30]  # All but last 30 days
        
        # Create and fit model
        model = ForecastingModel('mock')
        model.fit(train_data, 'revenue', 'date')
        
        assert model.is_fitted
        
        # Make predictions
        predictions = model.predict(30)
        
        assert len(predictions) == 30
        assert predictions['forecast'].notna().all()
        
        # Test scenarios
        scenarios = {
            'base': pd.DataFrame({'promo': [0] * 30}),
            'promo': pd.DataFrame({'promo': [1] * 30})
        }
        
        scenario_results = model.predict_with_scenarios(30, scenarios)
        
        assert len(scenario_results) == 2
        assert 'base' in scenario_results
        assert 'promo' in scenario_results
        
        # Get metadata
        metadata = model.get_metadata()
        assert metadata['is_fitted']
        assert 'fit_time' in metadata
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with non-existent model
        with pytest.raises(ValueError):
            ForecastingModel('nonexistent_model')
        
        # Test fitting with empty data
        ModelFactory.register_model('mock', MockModel)
        model = ForecastingModel('mock')
        
        # Mock model should still fit successfully even with empty data
        # since it's a simple mock implementation
        empty_df = pd.DataFrame()
        model.fit(empty_df, 'revenue', 'date')
        assert model.is_fitted


if __name__ == '__main__':
    pytest.main([__file__])