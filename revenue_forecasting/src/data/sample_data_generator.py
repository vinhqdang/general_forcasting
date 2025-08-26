"""
Sample data generator for creating realistic revenue forecasting datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class RevenueDataGenerator:
    """Generator for synthetic revenue time series data."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def generate_daily_revenue(self,
                              start_date: str = '2020-01-01',
                              end_date: str = '2023-12-31',
                              base_revenue: float = 10000,
                              trend_slope: float = 0.1,
                              seasonality_amplitude: float = 0.3,
                              noise_level: float = 0.1,
                              weekend_effect: float = -0.2,
                              holiday_effect: float = 0.5) -> pd.DataFrame:
        """
        Generate daily revenue data with trend, seasonality, and noise.
        
        Args:
            start_date: Start date for the time series
            end_date: End date for the time series
            base_revenue: Base daily revenue amount
            trend_slope: Slope of the linear trend (per day)
            seasonality_amplitude: Amplitude of yearly seasonality
            noise_level: Level of random noise
            weekend_effect: Effect of weekends on revenue (negative = lower)
            holiday_effect: Effect of holidays on revenue
            
        Returns:
            DataFrame with date and revenue columns
        """
        # Reset random seed for reproducibility
        np.random.seed(self.seed)
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        
        # Base trend
        trend = base_revenue + trend_slope * np.arange(n_days)
        
        # Yearly seasonality (peak in winter/holiday season)
        day_of_year = dates.dayofyear
        yearly_season = seasonality_amplitude * base_revenue * np.sin(2 * np.pi * (day_of_year - 60) / 365)
        
        # Weekly seasonality (lower on weekends)
        day_of_week = dates.dayofweek
        weekly_season = weekend_effect * base_revenue * np.isin(day_of_week, [5, 6]).astype(float)
        
        # Holiday effects (simplified - major holidays)
        holiday_boost = np.zeros(n_days)
        for i, date in enumerate(dates):
            # Christmas season
            if date.month == 12 and date.day >= 20:
                holiday_boost[i] = holiday_effect * base_revenue
            # Black Friday (4th Thursday in November + following days)
            elif date.month == 11 and date.day >= 22 and date.day <= 28 and date.dayofweek >= 3:
                holiday_boost[i] = holiday_effect * base_revenue * 1.5
            # Valentine's Day
            elif date.month == 2 and date.day == 14:
                holiday_boost[i] = holiday_effect * base_revenue * 0.5
        
        # Random noise
        noise = np.random.normal(0, noise_level * base_revenue, n_days)
        
        # Combine all components
        revenue = trend + yearly_season + weekly_season + holiday_boost + noise
        
        # Ensure no negative revenue
        revenue = np.maximum(revenue, base_revenue * 0.1)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'day_of_week': day_of_week,
            'day_of_year': day_of_year,
            'month': dates.month,
            'year': dates.year,
            'is_weekend': np.isin(day_of_week, [5, 6]).astype(int),
            'is_holiday': (holiday_boost > 0).astype(int)
        })
        
        return df
    
    def generate_hourly_revenue(self,
                               start_date: str = '2023-01-01',
                               days: int = 90,
                               base_revenue: float = 500,
                               daily_pattern: bool = True,
                               weekend_effect: float = -0.3) -> pd.DataFrame:
        """
        Generate hourly revenue data.
        
        Args:
            start_date: Start date for the time series
            days: Number of days to generate
            base_revenue: Base hourly revenue
            daily_pattern: Whether to include daily business hours pattern
            weekend_effect: Effect of weekends on revenue
            
        Returns:
            DataFrame with hourly revenue data
        """
        # Create hourly date range
        start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=days*24, freq='h')
        n_hours = len(dates)
        
        # Base revenue
        revenue = np.full(n_hours, base_revenue, dtype=np.float64)
        
        # Daily pattern (business hours effect)
        if daily_pattern:
            hour_of_day = dates.hour.values  # Convert to numpy array
            # Business hours multiplier (peak at 2 PM)
            business_hours_multiplier = 0.3 + 0.7 * np.exp(-0.5 * ((hour_of_day - 14) / 4) ** 2)
            # Night hours (very low activity)
            night_mask = np.isin(hour_of_day, [0, 1, 2, 3, 4, 5])
            business_hours_multiplier[night_mask] *= 0.1
            revenue *= business_hours_multiplier
        
        # Weekend effect
        day_of_week = dates.dayofweek.values  # Convert to numpy array
        weekend_multiplier = np.ones(n_hours)
        weekend_mask = np.isin(day_of_week, [5, 6])
        weekend_multiplier[weekend_mask] *= (1 + weekend_effect)
        revenue *= weekend_multiplier
        
        # Add noise
        noise = np.random.normal(0, base_revenue * 0.15, n_hours)
        revenue += noise
        
        # Ensure positive revenue
        revenue = np.maximum(revenue, base_revenue * 0.05)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'revenue': revenue,
            'hour': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': np.isin(day_of_week, [5, 6]).astype(int),
            'is_business_hour': np.isin(hour_of_day, range(9, 18)).astype(int)
        })
        
        return df
    
    def add_promotional_campaigns(self,
                                 df: pd.DataFrame,
                                 campaign_dates: List[tuple],
                                 campaign_effects: List[float]) -> pd.DataFrame:
        """
        Add promotional campaign effects to existing revenue data.
        
        Args:
            df: Existing revenue DataFrame
            campaign_dates: List of (start_date, end_date) tuples
            campaign_effects: List of multiplicative effects for each campaign
            
        Returns:
            DataFrame with added campaign effects
        """
        df_with_campaigns = df.copy()
        df_with_campaigns['promotion_active'] = 0
        df_with_campaigns['promotion_effect'] = 1.0
        
        for (start_date, end_date), effect in zip(campaign_dates, campaign_effects):
            mask = (df_with_campaigns['date'] >= start_date) & (df_with_campaigns['date'] <= end_date)
            df_with_campaigns.loc[mask, 'promotion_active'] = 1
            df_with_campaigns.loc[mask, 'promotion_effect'] = effect
            df_with_campaigns.loc[mask, 'revenue'] *= effect
        
        return df_with_campaigns
    
    def add_external_factors(self,
                           df: pd.DataFrame,
                           weather_effect: bool = True,
                           economic_effect: bool = True,
                           competitor_effect: bool = True) -> pd.DataFrame:
        """
        Add external factors that might affect revenue.
        
        Args:
            df: Existing revenue DataFrame
            weather_effect: Whether to add weather effects
            economic_effect: Whether to add economic indicators
            competitor_effect: Whether to add competitor effects
            
        Returns:
            DataFrame with added external factors
        """
        df_extended = df.copy()
        n_rows = len(df_extended)
        
        if weather_effect:
            # Simplified weather score (0-100, higher is better for business)
            df_extended['weather_score'] = np.random.normal(70, 15, n_rows)
            df_extended['weather_score'] = np.clip(df_extended['weather_score'], 0, 100)
            
            # Weather affects revenue
            weather_multiplier = 0.8 + 0.4 * (df_extended['weather_score'] / 100)
            df_extended['revenue'] *= weather_multiplier
        
        if economic_effect:
            # Economic confidence index (trending upward with noise)
            trend = np.linspace(0, 20, n_rows)
            noise = np.random.normal(0, 5, n_rows)
            df_extended['economic_index'] = 50 + trend + noise
            df_extended['economic_index'] = np.clip(df_extended['economic_index'], 30, 100)
            
            # Economic effect on revenue
            econ_multiplier = 0.7 + 0.6 * (df_extended['economic_index'] / 100)
            df_extended['revenue'] *= econ_multiplier
        
        if competitor_effect:
            # Competitor activity (random spikes representing competitor campaigns)
            df_extended['competitor_activity'] = np.random.exponential(0.2, n_rows)
            df_extended['competitor_activity'] = np.clip(df_extended['competitor_activity'], 0, 2)
            
            # High competitor activity reduces revenue
            comp_multiplier = 1.2 - 0.4 * df_extended['competitor_activity']
            df_extended['revenue'] *= comp_multiplier
        
        return df_extended


def create_sample_datasets() -> Dict[str, pd.DataFrame]:
    """
    Create various sample datasets for testing and examples.
    
    Returns:
        Dictionary of sample datasets
    """
    generator = RevenueDataGenerator(seed=42)
    datasets = {}
    
    # Daily revenue data (4 years)
    logger.info("Generating daily revenue dataset...")
    daily_data = generator.generate_daily_revenue(
        start_date='2020-01-01',
        end_date='2023-12-31',
        base_revenue=15000,
        trend_slope=2.0,
        seasonality_amplitude=0.4,
        noise_level=0.12
    )
    
    # Add promotional campaigns
    campaigns = [
        ('2020-11-25', '2020-11-29'),  # Black Friday
        ('2021-07-04', '2021-07-10'),  # Summer sale
        ('2021-11-25', '2021-11-29'),  # Black Friday
        ('2022-05-15', '2022-05-22'),  # Spring promotion
        ('2022-11-25', '2022-11-29'),  # Black Friday
        ('2023-06-01', '2023-06-07'),  # Summer kickoff
        ('2023-11-25', '2023-11-29'),  # Black Friday
    ]
    campaign_effects = [1.8, 1.3, 1.9, 1.2, 2.0, 1.4, 2.1]
    
    daily_data = generator.add_promotional_campaigns(daily_data, campaigns, campaign_effects)
    daily_data = generator.add_external_factors(daily_data)
    
    datasets['daily_ecommerce_revenue'] = daily_data
    
    # Hourly revenue data (3 months)
    logger.info("Generating hourly revenue dataset...")
    hourly_data = generator.generate_hourly_revenue(
        start_date='2023-10-01',
        days=90,
        base_revenue=800,
        daily_pattern=True,
        weekend_effect=-0.2
    )
    datasets['hourly_retail_revenue'] = hourly_data
    
    # Simple daily data for quick testing
    logger.info("Generating simple test dataset...")
    simple_data = generator.generate_daily_revenue(
        start_date='2023-01-01',
        end_date='2023-06-30',
        base_revenue=5000,
        trend_slope=1.0,
        seasonality_amplitude=0.2,
        noise_level=0.08
    )
    datasets['simple_daily_revenue'] = simple_data[['date', 'revenue']]
    
    # Weekly aggregated data
    logger.info("Generating weekly revenue dataset...")
    weekly_data = daily_data.copy()
    weekly_data['week'] = weekly_data['date'].dt.to_period('W')
    weekly_agg = weekly_data.groupby('week').agg({
        'revenue': 'sum',
        'promotion_active': 'max',
        'weather_score': 'mean',
        'economic_index': 'mean'
    }).reset_index()
    weekly_agg['date'] = weekly_agg['week'].dt.start_time
    weekly_agg = weekly_agg.drop('week', axis=1)
    datasets['weekly_revenue'] = weekly_agg
    
    # Monthly data
    logger.info("Generating monthly revenue dataset...")
    monthly_data = daily_data.copy()
    monthly_data['month'] = monthly_data['date'].dt.to_period('M')
    monthly_agg = monthly_data.groupby('month').agg({
        'revenue': 'sum',
        'promotion_active': 'max',
        'weather_score': 'mean',
        'economic_index': 'mean'
    }).reset_index()
    monthly_agg['date'] = monthly_agg['month'].dt.start_time
    monthly_agg = monthly_agg.drop('month', axis=1)
    datasets['monthly_revenue'] = monthly_agg
    
    logger.info(f"Generated {len(datasets)} sample datasets")
    return datasets


def save_sample_datasets(datasets: Dict[str, pd.DataFrame], 
                        output_dir: str = "data/samples") -> None:
    """
    Save sample datasets to CSV files.
    
    Args:
        datasets: Dictionary of datasets to save
        output_dir: Directory to save the files
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in datasets.items():
        filepath = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {name} dataset to {filepath} ({len(df)} rows)")


if __name__ == "__main__":
    # Generate and save sample datasets
    datasets = create_sample_datasets()
    save_sample_datasets(datasets)
    
    # Print summary
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {list(df.columns)}")
        if 'revenue' in df.columns:
            print(f"  Revenue stats: min={df['revenue'].min():.2f}, "
                  f"max={df['revenue'].max():.2f}, mean={df['revenue'].mean():.2f}")