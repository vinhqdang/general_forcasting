"""
Unit tests for sample data generator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('src')

from data.sample_data_generator import RevenueDataGenerator, create_sample_datasets


class TestRevenueDataGenerator:
    """Test cases for RevenueDataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a RevenueDataGenerator instance."""
        return RevenueDataGenerator(seed=42)
    
    def test_init(self):
        """Test generator initialization."""
        gen = RevenueDataGenerator(seed=123)
        assert gen.seed == 123
    
    def test_generate_daily_revenue_basic(self, generator):
        """Test basic daily revenue generation."""
        df = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31',
            base_revenue=1000
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 31  # January has 31 days
        assert 'date' in df.columns
        assert 'revenue' in df.columns
        assert df['revenue'].min() > 0  # No negative revenue
        assert df['revenue'].mean() > 0
    
    def test_generate_daily_revenue_columns(self, generator):
        """Test daily revenue generation has expected columns."""
        df = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        expected_columns = [
            'date', 'revenue', 'day_of_week', 'day_of_year',
            'month', 'year', 'is_weekend', 'is_holiday'
        ]
        
        for col in expected_columns:
            assert col in df.columns
    
    def test_generate_daily_revenue_date_range(self, generator):
        """Test daily revenue generation with different date ranges."""
        start_date = '2022-06-01'
        end_date = '2022-06-30'
        
        df = generator.generate_daily_revenue(
            start_date=start_date,
            end_date=end_date
        )
        
        assert df['date'].min() == pd.to_datetime(start_date)
        assert df['date'].max() == pd.to_datetime(end_date)
        assert len(df) == 30  # June has 30 days
    
    def test_generate_daily_revenue_parameters(self, generator):
        """Test daily revenue generation with different parameters."""
        # High base revenue
        df_high = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31',
            base_revenue=10000
        )
        
        # Low base revenue
        df_low = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31',
            base_revenue=1000
        )
        
        assert df_high['revenue'].mean() > df_low['revenue'].mean()
    
    def test_generate_hourly_revenue_basic(self, generator):
        """Test basic hourly revenue generation."""
        df = generator.generate_hourly_revenue(
            start_date='2023-01-01',
            days=7,
            base_revenue=100
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7 * 24  # 7 days * 24 hours
        assert 'date' in df.columns
        assert 'revenue' in df.columns
        assert 'hour' in df.columns
        assert df['revenue'].min() > 0
    
    def test_generate_hourly_revenue_columns(self, generator):
        """Test hourly revenue generation has expected columns."""
        df = generator.generate_hourly_revenue(
            start_date='2023-01-01',
            days=1
        )
        
        expected_columns = [
            'date', 'revenue', 'hour', 'day_of_week',
            'is_weekend', 'is_business_hour'
        ]
        
        for col in expected_columns:
            assert col in df.columns
    
    def test_generate_hourly_revenue_business_hours(self, generator):
        """Test hourly revenue has business hours pattern."""
        df = generator.generate_hourly_revenue(
            start_date='2023-01-02',  # Monday
            days=1,
            daily_pattern=True
        )
        
        business_hours = df[df['is_business_hour'] == 1]['revenue']
        non_business_hours = df[df['is_business_hour'] == 0]['revenue']
        
        # Business hours should generally have higher revenue
        assert business_hours.mean() > non_business_hours.mean()
    
    def test_add_promotional_campaigns(self, generator):
        """Test adding promotional campaigns."""
        # Generate base data
        df = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        base_revenue_mean = df['revenue'].mean()
        
        # Add campaigns
        campaigns = [('2023-01-10', '2023-01-12')]
        effects = [2.0]  # Double revenue
        
        df_with_campaigns = generator.add_promotional_campaigns(
            df, campaigns, effects
        )
        
        assert 'promotion_active' in df_with_campaigns.columns
        assert 'promotion_effect' in df_with_campaigns.columns
        
        # Check campaign period has higher revenue
        campaign_period = df_with_campaigns[
            (df_with_campaigns['date'] >= '2023-01-10') & 
            (df_with_campaigns['date'] <= '2023-01-12')
        ]
        
        assert campaign_period['promotion_active'].all()
        assert (campaign_period['promotion_effect'] == 2.0).all()
    
    def test_add_external_factors(self, generator):
        """Test adding external factors."""
        df = generator.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        df_extended = generator.add_external_factors(
            df,
            weather_effect=True,
            economic_effect=True,
            competitor_effect=True
        )
        
        assert 'weather_score' in df_extended.columns
        assert 'economic_index' in df_extended.columns
        assert 'competitor_activity' in df_extended.columns
        
        # Check value ranges
        assert df_extended['weather_score'].min() >= 0
        assert df_extended['weather_score'].max() <= 100
        assert df_extended['economic_index'].min() >= 30
        assert df_extended['economic_index'].max() <= 100
        assert df_extended['competitor_activity'].min() >= 0
    
    def test_weekend_effect(self, generator):
        """Test weekend effect in daily data."""
        df = generator.generate_daily_revenue(
            start_date='2023-01-01',  # Sunday
            end_date='2023-01-14',   # Two weeks
            weekend_effect=-0.5  # 50% reduction on weekends
        )
        
        weekday_revenue = df[df['is_weekend'] == 0]['revenue']
        weekend_revenue = df[df['is_weekend'] == 1]['revenue']
        
        # Weekend revenue should be lower
        assert weekend_revenue.mean() < weekday_revenue.mean()
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        gen1 = RevenueDataGenerator(seed=42)
        gen2 = RevenueDataGenerator(seed=42)
        
        df1 = gen1.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        df2 = gen2.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        pd.testing.assert_frame_equal(df1, df2)


class TestCreateSampleDatasets:
    """Test cases for create_sample_datasets function."""
    
    def test_create_sample_datasets(self):
        """Test creating sample datasets."""
        datasets = create_sample_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        
        # Check expected datasets exist
        expected_datasets = [
            'daily_ecommerce_revenue',
            'hourly_retail_revenue',
            'simple_daily_revenue',
            'weekly_revenue',
            'monthly_revenue'
        ]
        
        for dataset_name in expected_datasets:
            assert dataset_name in datasets
            assert isinstance(datasets[dataset_name], pd.DataFrame)
            assert len(datasets[dataset_name]) > 0
    
    def test_daily_ecommerce_dataset(self):
        """Test daily ecommerce dataset specifics."""
        datasets = create_sample_datasets()
        df = datasets['daily_ecommerce_revenue']
        
        assert 'date' in df.columns
        assert 'revenue' in df.columns
        assert 'promotion_active' in df.columns
        
        # Should have 4 years of data
        date_range = df['date'].max() - df['date'].min()
        assert date_range.days >= 1460  # ~4 years
    
    def test_hourly_retail_dataset(self):
        """Test hourly retail dataset specifics."""
        datasets = create_sample_datasets()
        df = datasets['hourly_retail_revenue']
        
        assert 'date' in df.columns
        assert 'revenue' in df.columns
        assert 'hour' in df.columns
        
        # Should have ~3 months of hourly data
        assert len(df) >= 2000  # ~90 days * 24 hours
    
    def test_simple_daily_dataset(self):
        """Test simple daily dataset."""
        datasets = create_sample_datasets()
        df = datasets['simple_daily_revenue']
        
        assert 'date' in df.columns
        assert 'revenue' in df.columns
        assert len(df.columns) == 2  # Only date and revenue
        
        # Should have ~6 months of data
        assert len(df) >= 180
    
    def test_aggregated_datasets(self):
        """Test weekly and monthly aggregated datasets."""
        datasets = create_sample_datasets()
        
        weekly_df = datasets['weekly_revenue']
        monthly_df = datasets['monthly_revenue']
        
        # Weekly dataset
        assert 'date' in weekly_df.columns
        assert 'revenue' in weekly_df.columns
        assert len(weekly_df) > 0
        
        # Monthly dataset
        assert 'date' in monthly_df.columns
        assert 'revenue' in monthly_df.columns
        assert len(monthly_df) > 0
        
        # Monthly should have fewer rows than weekly
        assert len(monthly_df) < len(weekly_df)


class TestDataQuality:
    """Test data quality aspects."""
    
    def test_no_negative_revenue(self):
        """Test that generated revenue is never negative."""
        gen = RevenueDataGenerator(seed=42)
        
        # Test daily data
        daily_df = gen.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        assert (daily_df['revenue'] > 0).all()
        
        # Test hourly data
        hourly_df = gen.generate_hourly_revenue(
            start_date='2023-01-01',
            days=30
        )
        assert (hourly_df['revenue'] > 0).all()
    
    def test_date_continuity(self):
        """Test that dates are continuous."""
        gen = RevenueDataGenerator(seed=42)
        
        df = gen.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        # Check dates are continuous
        date_diffs = df['date'].diff().dropna()
        expected_diff = pd.Timedelta(days=1)
        
        assert (date_diffs == expected_diff).all()
    
    def test_reasonable_revenue_values(self):
        """Test that revenue values are reasonable."""
        gen = RevenueDataGenerator(seed=42)
        
        df = gen.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-01-31',
            base_revenue=1000
        )
        
        # Revenue should be within reasonable bounds of base revenue
        assert df['revenue'].min() > 100  # At least 10% of base
        assert df['revenue'].max() < 10000  # Less than 10x base
        
        # Mean should be close to base revenue
        assert abs(df['revenue'].mean() - 1000) < 500
    
    def test_seasonal_patterns(self):
        """Test that seasonal patterns are present."""
        gen = RevenueDataGenerator(seed=42)
        
        df = gen.generate_daily_revenue(
            start_date='2023-01-01',
            end_date='2023-12-31',
            seasonality_amplitude=0.5
        )
        
        # Group by month and check for seasonal variation
        monthly_revenue = df.groupby(df['date'].dt.month)['revenue'].mean()
        
        # Should have some variation across months
        cv = monthly_revenue.std() / monthly_revenue.mean()
        assert cv > 0.05  # At least 5% coefficient of variation


if __name__ == '__main__':
    pytest.main([__file__])