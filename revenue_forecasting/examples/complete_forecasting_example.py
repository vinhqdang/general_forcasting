"""
Complete forecasting example using the flexible API with feature_list and target_variable.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from data.data_loader import DataLoader
from data.sample_data_generator import create_sample_datasets
from models.forecasting_model import ForecastingModel

def complete_forecasting_workflow():
    """Demonstrate complete forecasting workflow with flexible API."""
    
    print("=== Complete Revenue Forecasting Workflow ===\n")
    
    # 1. Data Preparation
    print("1. Loading and preparing data...")
    datasets = create_sample_datasets()
    raw_data = datasets['daily_ecommerce_revenue']
    print(f"   Raw data shape: {raw_data.shape}")
    
    # 2. Initialize DataLoader with specific requirements
    print("\n2. Setting up flexible DataLoader...")
    loader = DataLoader(
        date_column='date',
        target_variable='revenue',  # Clear target specification
        feature_list=[              # Explicit feature selection
            'day_of_week',
            'month', 
            'is_weekend',
            'promotion_active',
            'weather_score',
            'economic_index'
        ],
        frequency='D'
    )
    
    # Load and validate data
    loader.load_from_dataframe(raw_data)
    validation = loader.validate_data()
    
    print(f"   Target variable: {validation['target_variable']}")
    print(f"   Features selected: {len(validation['feature_list'])}")
    print(f"   Data validation passed: {validation['has_target_variable'] and not validation['missing_features']}")
    
    # 3. Data preprocessing
    print("\n3. Data preprocessing...")
    processed_data = loader.preprocess(
        sort_by_date=True,
        remove_duplicates=True,
        fill_missing='interpolate'
    )
    
    # Add time-based features
    enriched_data = loader.create_time_features()
    
    # Update feature list to include new time features
    loader.add_features(['quarter', 'week_of_year'])
    
    print(f"   Processed data shape: {processed_data.shape}")
    print(f"   Total features after enrichment: {len(loader.feature_list)}")
    
    # 4. Split data for training and testing
    print("\n4. Splitting data...")
    train_size = int(len(enriched_data) * 0.8)
    train_data = enriched_data[:train_size]
    test_data = enriched_data[train_size:]
    
    print(f"   Training data: {train_data.shape}")
    print(f"   Test data: {test_data.shape}")
    
    # 5. Model Training with different approaches
    print("\n5. Training models with flexible API...")
    
    # Model A: Using mock model (since real ML libraries might not be installed)
    try:
        # Register a simple mock model for demonstration
        from models.forecasting_model import ModelFactory
        from models.base_model import BaseForecastingModel
        
        class DemoModel(BaseForecastingModel):
            def __init__(self, **kwargs):
                super().__init__("DemoModel", **kwargs)
                self.target_variable = None
                self.feature_list = None
                
            def fit(self, train_data, target_variable, date_column='date', feature_list=None, **kwargs):
                self.target_variable = target_variable
                self.feature_list = feature_list or []
                self.is_fitted = True
                self.fit_time = 0.1
                
                # Store some basic statistics
                self.mean_target = train_data[target_variable].mean()
                self.std_target = train_data[target_variable].std()
                return self
                
            def predict(self, periods, confidence_level=0.95, include_history=False, **kwargs):
                # Simple prediction: random walk with trend
                last_date = pd.to_datetime('2023-12-31')  # From sample data
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=periods, freq='D')
                
                # Generate realistic predictions
                predictions = np.random.normal(self.mean_target, self.std_target * 0.1, periods)
                trend = np.linspace(0, periods * 2, periods)  # Small upward trend
                predictions += trend
                
                lower_bound = predictions - 1.96 * self.std_target * 0.1
                upper_bound = predictions + 1.96 * self.std_target * 0.1
                
                return pd.DataFrame({
                    'date': future_dates,
                    'forecast': predictions,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
                
            def predict_with_exog(self, periods, exog_future=None, confidence_level=0.95, **kwargs):
                return self.predict(periods, confidence_level, **kwargs)
        
        # Register the demo model
        ModelFactory.register_model('demo', DemoModel)
        
        # Create and train model using flexible API
        model = ForecastingModel('demo')
        
        # Fit with new flexible parameters
        model.fit(
            train_data=train_data,
            target_variable=loader.target_variable,  # Clear target
            date_column=loader.date_column,
            feature_list=loader.feature_list        # Explicit features
        )
        
        print(f"   Model trained successfully!")
        print(f"   Target used: {model.metadata['target_variable']}")
        print(f"   Features used: {len(model.metadata['feature_list'])}")
        
        # 6. Generate predictions
        print("\n6. Generating predictions...")
        
        # Basic forecast
        forecast = model.predict(
            periods=30,
            confidence_level=0.95,
            include_history=False
        )
        
        print(f"   Generated {len(forecast)} predictions")
        print(f"   Forecast columns: {list(forecast.columns)}")
        print(f"   Average forecast: {forecast['forecast'].mean():.2f}")
        
        # What-if scenarios
        print("\n7. What-if scenario analysis...")
        
        # Create scenarios with different feature values
        scenario_data = pd.DataFrame({
            'promotion_active': [1] * 30,  # High promotion scenario
            'weather_score': [85] * 30     # Good weather scenario
        })
        
        scenarios = {
            'baseline': pd.DataFrame({'promotion_active': [0] * 30}),
            'high_promo': scenario_data
        }
        
        scenario_results = model.predict_with_scenarios(
            periods=30,
            scenarios=scenarios,
            confidence_level=0.95
        )
        
        for scenario_name, result in scenario_results.items():
            if result is not None:
                avg_forecast = result['forecast'].mean()
                print(f"   {scenario_name}: Avg forecast = {avg_forecast:.2f}")
        
        # 8. Model evaluation and insights  
        print("\n8. Model insights...")
        
        model_metadata = model.get_metadata()
        print(f"   Model type: {model_metadata['model_type']}")
        print(f"   Is fitted: {model_metadata['is_fitted']}")
        print(f"   Fit time: {model_metadata.get('fit_time', 'N/A')} seconds")
        print(f"   Target variable: {model_metadata.get('target_variable', 'N/A')}")
        print(f"   Feature count: {len(model_metadata.get('feature_list', []))}")
        
        # 9. Data insights
        print("\n9. Data insights...")
        data_info = loader.get_data_info()
        
        print(f"   Dataset shape: {data_info['shape']}")
        print(f"   Target variable: {data_info['target_variable']}")
        print(f"   Feature count: {data_info['feature_count']}")
        
        if 'target_statistics' in data_info:
            target_stats = data_info['target_statistics']
            print(f"   Target mean: {target_stats['mean']:.2f}")
            print(f"   Target std: {target_stats['std']:.2f}")
            print(f"   Target range: {target_stats['min']:.2f} - {target_stats['max']:.2f}")
        
        print("\n=== Workflow completed successfully! ===")
        
        # 10. Summary of benefits
        print("\n🎯 Key Benefits of Flexible API:")
        print("✅ Clear target_variable and feature_list separation")
        print("✅ Automatic feature detection or manual specification")
        print("✅ Dynamic feature management during development") 
        print("✅ Works across different domains (sales, revenue, engagement)")
        print("✅ Comprehensive validation and metadata tracking")
        print("✅ Easy scenario analysis with specific features")
        print("✅ Backward compatibility with existing code")
        print("✅ Scalable to complex feature engineering pipelines")
        
    except Exception as e:
        print(f"   Error in modeling: {str(e)}")
        print("   This is expected if ML libraries are not installed")

if __name__ == "__main__":
    complete_forecasting_workflow()