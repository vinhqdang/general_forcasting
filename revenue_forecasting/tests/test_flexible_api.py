"""
Unit tests for the flexible API with feature_list and target_variable parameters.
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.append('src')

from data.data_loader import DataLoader


class TestFlexibleAPI:
    """Test cases for the flexible API features."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple features."""
        dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000, 100, len(dates)),
            'price': np.random.normal(25, 5, len(dates)),
            'marketing_spend': np.random.normal(500, 100, len(dates)),
            'weather_score': np.random.uniform(0, 100, len(dates)),
            'day_of_week': dates.dayofweek,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
            'category': np.random.choice(['A', 'B', 'C'], len(dates)),
            'high_cardinality': np.arange(len(dates)) * 100 + np.random.randint(0, 50, len(dates))  # High cardinality
        })
    
    def test_target_variable_parameter(self, sample_data):
        """Test target_variable parameter functionality."""
        loader = DataLoader(target_variable='revenue')
        loader.load_from_dataframe(sample_data)
        
        assert loader.target_variable == 'revenue'
        assert loader.target_column == 'revenue'  # Backward compatibility
        
        validation = loader.validate_data()
        assert validation['target_variable'] == 'revenue'
        assert validation['has_target_variable'] is True
    
    def test_auto_feature_detection(self, sample_data):
        """Test automatic feature detection."""
        loader = DataLoader(target_variable='revenue', feature_list=None)
        loader.load_from_dataframe(sample_data)
        
        # Should auto-detect numeric and low-cardinality categorical features
        expected_features = ['price', 'marketing_spend', 'weather_score', 
                           'day_of_week', 'is_weekend', 'category']
        
        # Check that main features are detected
        for feature in expected_features:
            assert feature in loader.feature_list, f"Expected feature '{feature}' not found in {loader.feature_list}"
        
        # Date and target should be excluded
        assert 'date' not in loader.feature_list
        assert 'revenue' not in loader.feature_list
        
        # Should have detected some features
        assert len(loader.feature_list) >= len(expected_features)
    
    def test_manual_feature_specification(self, sample_data):
        """Test manual feature specification."""
        manual_features = ['price', 'marketing_spend', 'is_weekend']
        
        loader = DataLoader(
            target_variable='revenue',
            feature_list=manual_features
        )
        loader.load_from_dataframe(sample_data)
        
        assert loader.feature_list == manual_features
        
        validation = loader.validate_data()
        assert validation['feature_list'] == manual_features
        assert validation['missing_features'] == []
    
    def test_set_feature_list(self, sample_data):
        """Test setting feature list after initialization."""
        loader = DataLoader(target_variable='revenue')
        loader.load_from_dataframe(sample_data)
        
        new_features = ['price', 'weather_score']
        loader.set_feature_list(new_features)
        
        assert loader.feature_list == new_features
    
    def test_add_features(self, sample_data):
        """Test adding features to existing list."""
        loader = DataLoader(
            target_variable='revenue',
            feature_list=['price', 'marketing_spend']
        )
        loader.load_from_dataframe(sample_data)
        
        loader.add_features(['weather_score', 'is_weekend'])
        
        expected = ['price', 'marketing_spend', 'weather_score', 'is_weekend']
        assert loader.feature_list == expected
    
    def test_remove_features(self, sample_data):
        """Test removing features from list."""
        loader = DataLoader(
            target_variable='revenue',
            feature_list=['price', 'marketing_spend', 'weather_score', 'is_weekend']
        )
        loader.load_from_dataframe(sample_data)
        
        loader.remove_features(['marketing_spend', 'weather_score'])
        
        expected = ['price', 'is_weekend']
        assert loader.feature_list == expected
    
    def test_feature_validation(self, sample_data):
        """Test feature validation."""
        # Test with missing features
        loader = DataLoader(
            target_variable='revenue',
            feature_list=['price', 'nonexistent_feature']
        )
        loader.load_from_dataframe(sample_data)
        
        validation = loader.validate_data()
        assert 'nonexistent_feature' in validation['missing_features']
    
    def test_set_invalid_features(self, sample_data):
        """Test setting invalid feature list."""
        loader = DataLoader(target_variable='revenue')
        loader.load_from_dataframe(sample_data)
        
        with pytest.raises(ValueError, match="Feature columns not found"):
            loader.set_feature_list(['invalid_feature'])
    
    def test_add_invalid_features(self, sample_data):
        """Test adding invalid features."""
        loader = DataLoader(
            target_variable='revenue',
            feature_list=['price']
        )
        loader.load_from_dataframe(sample_data)
        
        with pytest.raises(ValueError, match="Feature columns not found"):
            loader.add_features(['invalid_feature'])
    
    def test_data_info_with_features(self, sample_data):
        """Test data info includes feature information."""
        loader = DataLoader(
            target_variable='revenue',
            feature_list=['price', 'marketing_spend', 'category']
        )
        loader.load_from_dataframe(sample_data)
        
        info = loader.get_data_info()
        
        assert info['target_variable'] == 'revenue'
        assert info['feature_list'] == ['price', 'marketing_spend', 'category']
        assert info['feature_count'] == 3
        assert 'feature_statistics' in info
        assert 'target_statistics' in info
        
        # Check feature statistics
        feature_stats = info['feature_statistics']
        assert 'price' in feature_stats
        assert 'marketing_spend' in feature_stats
        assert 'category' in feature_stats
        
        # Numeric features should have descriptive stats
        assert 'mean' in feature_stats['price']
        assert 'std' in feature_stats['price']
        
        # Categorical features should have unique counts
        assert 'unique_values' in feature_stats['category']
        assert 'most_common' in feature_stats['category']
    
    def test_different_target_variables(self):
        """Test with different target variable names."""
        # Sales data
        sales_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'sales': np.random.randn(10),
            'units_sold': np.random.randint(1, 100, 10),
            'profit': np.random.randn(10)
        })
        
        loader = DataLoader(target_variable='sales')
        loader.load_from_dataframe(sales_data)
        
        assert loader.target_variable == 'sales'
        assert 'units_sold' in loader.feature_list
        assert 'profit' in loader.feature_list
        assert 'sales' not in loader.feature_list
        
        # User engagement data
        engagement_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'active_users': np.random.randint(100, 1000, 10),
            'page_views': np.random.randint(500, 5000, 10),
            'bounce_rate': np.random.uniform(0, 1, 10)
        })
        
        loader2 = DataLoader(
            date_column='timestamp',
            target_variable='active_users'
        )
        loader2.load_from_dataframe(engagement_data)
        
        assert loader2.target_variable == 'active_users'
        assert 'page_views' in loader2.feature_list
        assert 'bounce_rate' in loader2.feature_list
        assert 'active_users' not in loader2.feature_list
    
    def test_backward_compatibility(self, sample_data):
        """Test that backward compatibility is maintained."""
        loader = DataLoader(target_variable='revenue')
        loader.load_from_dataframe(sample_data)
        
        # target_column should still work for backward compatibility
        assert hasattr(loader, 'target_column')
        assert loader.target_column == loader.target_variable
        
        # Validation should include both old and new keys
        validation = loader.validate_data()
        assert 'has_target_column' in validation  # Old key
        assert 'has_target_variable' in validation  # New key
        assert validation['has_target_column'] == validation['has_target_variable']


class TestFlexibleAPIIntegration:
    """Integration tests for flexible API."""
    
    def test_complete_workflow(self):
        """Test complete workflow with flexible API."""
        # Create complex dataset
        dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000, 100, len(dates)) + 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 7),
            'marketing_budget': np.random.normal(200, 50, len(dates)),
            'price': np.random.normal(25, 3, len(dates)),
            'competitor_price': np.random.normal(27, 3, len(dates)),
            'weather_index': np.random.uniform(0, 100, len(dates)),
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
            'season': pd.Categorical(['winter', 'winter', 'spring'] * (len(dates) // 3 + 1))[:len(dates)]
        })
        
        # Initialize with specific target and features
        loader = DataLoader(
            date_column='date',
            target_variable='revenue',
            feature_list=[
                'marketing_budget', 'price', 'competitor_price', 
                'weather_index', 'day_of_week', 'is_weekend'
            ]
        )
        
        # Load and process
        loader.load_from_dataframe(data)
        
        # Validate everything is correct
        validation = loader.validate_data()
        assert validation['has_target_variable']
        assert validation['target_variable'] == 'revenue'
        assert len(validation['feature_list']) == 6
        assert validation['missing_features'] == []
        
        # Process data
        processed = loader.preprocess()
        assert processed is not None
        
        # Add time features
        enriched = loader.create_time_features()
        assert enriched is not None
        
        # Dynamically add seasonal feature
        loader.add_features(['season'])
        assert 'season' in loader.feature_list
        
        # Get comprehensive info
        info = loader.get_data_info()
        assert info['target_variable'] == 'revenue'
        assert info['feature_count'] == 7  # 6 original + 1 added
        assert 'target_statistics' in info
        assert 'feature_statistics' in info
        
        # Verify feature statistics
        feature_stats = info['feature_statistics']
        assert 'marketing_budget' in feature_stats
        assert 'season' in feature_stats
        
        # Numeric features should have descriptive stats
        assert 'mean' in feature_stats['marketing_budget']
        
        # Categorical features should have category info
        assert 'unique_values' in feature_stats['season']


if __name__ == '__main__':
    pytest.main([__file__])