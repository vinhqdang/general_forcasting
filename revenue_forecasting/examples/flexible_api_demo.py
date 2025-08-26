"""
Demonstration of the flexible API with feature_list and target_variable parameters.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.data_loader import DataLoader
from data.sample_data_generator import create_sample_datasets

def demo_flexible_api():
    """Demonstrate the flexible API usage."""
    
    print("=== Revenue Forecasting Framework - Flexible API Demo ===\n")
    
    # 1. Load sample data
    print("1. Loading sample datasets...")
    datasets = create_sample_datasets()
    daily_data = datasets['daily_ecommerce_revenue']
    print(f"   Loaded daily e-commerce data: {daily_data.shape}")
    print(f"   Columns: {list(daily_data.columns)}")
    
    # 2. Initialize DataLoader with clear target and features
    print("\n2. Initializing DataLoader with flexible parameters...")
    
    # Option A: Let DataLoader auto-detect features
    loader_auto = DataLoader(
        date_column='date',
        target_variable='revenue',  # Clear target variable
        feature_list=None,          # Auto-detect features
        frequency='D'
    )
    
    loader_auto.load_from_dataframe(daily_data)
    print(f"   Auto-detected target: {loader_auto.target_variable}")
    print(f"   Auto-detected features: {loader_auto.feature_list}")
    
    # Option B: Manually specify features
    print("\n3. Manual feature specification...")
    manual_features = [
        'day_of_week', 'month', 'is_weekend', 'is_holiday',
        'promotion_active', 'weather_score', 'economic_index'
    ]
    
    loader_manual = DataLoader(
        date_column='date',
        target_variable='revenue',
        feature_list=manual_features,
        frequency='D'
    )
    
    loader_manual.load_from_dataframe(daily_data)
    print(f"   Manual target: {loader_manual.target_variable}")
    print(f"   Manual features: {loader_manual.feature_list}")
    
    # 3. Dynamic feature management
    print("\n4. Dynamic feature management...")
    
    # Add new features
    loader_manual.add_features(['year', 'competitor_activity'])
    print(f"   After adding features: {loader_manual.feature_list}")
    
    # Remove features
    loader_manual.remove_features(['year'])
    print(f"   After removing features: {loader_manual.feature_list}")
    
    # 4. Data validation with features
    print("\n5. Data validation...")
    validation = loader_manual.validate_data()
    print(f"   Has target variable: {validation['has_target_variable']}")
    print(f"   Target variable: {validation['target_variable']}")
    print(f"   Feature count: {len(validation['feature_list'])}")
    print(f"   Missing features: {validation['missing_features']}")
    
    # 5. Data info with feature statistics
    print("\n6. Comprehensive data information...")
    info = loader_manual.get_data_info()
    print(f"   Dataset shape: {info['shape']}")
    print(f"   Target variable: {info['target_variable']}")
    print(f"   Feature count: {info['feature_count']}")
    print(f"   Target statistics available: {'target_statistics' in info}")
    print(f"   Feature statistics available: {'feature_statistics' in info}")
    
    # 6. Different data scenarios
    print("\n7. Different data scenarios...")
    
    # Scenario A: Sales data
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'sales': np.random.normal(5000, 500, 100),
        'price': np.random.normal(25, 5, 100),
        'marketing_spend': np.random.normal(1000, 200, 100),
        'season': np.random.choice(['spring', 'summer', 'fall', 'winter'], 100)
    })
    
    sales_loader = DataLoader(
        target_variable='sales',     # Different target
        feature_list=['price', 'marketing_spend', 'season'],  # Specific features
        date_column='date'
    )
    sales_loader.load_from_dataframe(sales_data)
    print(f"   Sales data - Target: {sales_loader.target_variable}")
    print(f"   Sales data - Features: {sales_loader.feature_list}")
    
    # Scenario B: User engagement data
    engagement_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=200, freq='H'),
        'active_users': np.random.poisson(1000, 200),
        'page_views': np.random.poisson(5000, 200),
        'bounce_rate': np.random.beta(2, 8, 200),
        'device_mobile': np.random.choice([0, 1], 200, p=[0.3, 0.7])
    })
    
    engagement_loader = DataLoader(
        date_column='timestamp',
        target_variable='active_users',
        frequency='H'  # Hourly frequency
    )
    engagement_loader.load_from_dataframe(engagement_data)
    print(f"   Engagement data - Target: {engagement_loader.target_variable}")
    print(f"   Engagement data - Auto features: {engagement_loader.feature_list}")
    
    print("\n=== Demo completed successfully! ===")
    print("\nKey Benefits:")
    print("✅ Clear separation of target_variable and feature_list")
    print("✅ Automatic feature detection when needed")
    print("✅ Manual feature specification for control")
    print("✅ Dynamic feature management (add/remove)")
    print("✅ Comprehensive validation and statistics")
    print("✅ Backward compatibility maintained")
    print("✅ Works with any domain (sales, engagement, revenue, etc.)")

if __name__ == "__main__":
    demo_flexible_api()