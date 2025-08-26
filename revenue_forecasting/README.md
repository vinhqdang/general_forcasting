# Revenue Forecasting Framework

A comprehensive Python library for revenue forecasting using multiple algorithms including statistical models, machine learning, and deep learning approaches. Features automatic CPU/GPU optimization, extensible architecture, and comprehensive testing.

## 🚀 Features

- **Multiple Forecasting Algorithms**: ARIMA, SARIMAX, XGBoost, LightGBM, Prophet, N-BEATS, TCN
- **Automatic Device Optimization**: CPU/GPU detection and configuration
- **Complete Data Pipeline**: Loading, validation, preprocessing, and feature engineering
- **What-if Scenario Modeling**: Multi-scenario predictions with external variables
- **Confidence Intervals**: Prediction uncertainty quantification
- **Sample Data Generation**: Realistic synthetic datasets for testing
- **Extensible Architecture**: Easy to add new models and features
- **Comprehensive Testing**: 87+ unit tests ensuring reliability

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for acceleration)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vinhqdang/general_forcasting.git
cd general_forcasting/revenue_forecasting
```

### 2. Set Up Conda Environment
```bash
# Create and activate conda environment
conda create -n py310 python=3.10 -y
conda activate py310

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Additional ML Libraries (Optional)
```bash
# For full functionality, install these libraries:
pip install xgboost lightgbm prophet statsmodels

# For deep learning models:
pip install torch torchvision torchaudio

# For additional features:
pip install scikit-learn matplotlib seaborn plotly
```

## 🎯 Quick Start

### Basic Usage

```python
import sys
sys.path.append('src')

from data.data_loader import DataLoader
from models.forecasting_model import ForecastingModel
import pandas as pd

# 1. Load your data
loader = DataLoader(date_column='date', target_column='revenue')
df = pd.read_csv('your_revenue_data.csv')
loader.load_from_dataframe(df)

# 2. Preprocess data
processed_data = loader.preprocess(
    sort_by_date=True,
    remove_duplicates=True,
    fill_missing='interpolate'
)

# Add time features
data_with_features = loader.create_time_features()

# 3. Create and train a forecasting model
model = ForecastingModel('prophet')  # or 'xgboost', 'arima', etc.
model.fit(data_with_features, 'revenue', 'date')

# 4. Generate predictions
predictions = model.predict(
    periods=30,  # Forecast 30 periods ahead
    confidence_level=0.95,
    include_history=False
)

print(predictions.head())
```

### Using Sample Data

```python
from data.sample_data_generator import create_sample_datasets

# Generate sample datasets
datasets = create_sample_datasets()

# Use daily e-commerce revenue data
daily_data = datasets['daily_ecommerce_revenue']
print(f"Dataset shape: {daily_data.shape}")
print(daily_data.head())
```

### What-if Scenarios

```python
# Create scenarios with different promotional campaigns
scenarios = {
    'baseline': pd.DataFrame({
        'promotion_active': [0] * 30,
        'marketing_spend': [1000] * 30
    }),
    'high_promo': pd.DataFrame({
        'promotion_active': [1] * 30,
        'marketing_spend': [5000] * 30
    })
}

# Generate predictions for each scenario
scenario_results = model.predict_with_scenarios(
    periods=30,
    scenarios=scenarios,
    confidence_level=0.95
)

for scenario_name, forecast in scenario_results.items():
    print(f"\n{scenario_name} forecast:")
    print(forecast[['date', 'forecast', 'lower_bound', 'upper_bound']].head())
```

### Device Configuration

```python
from utils.device_config import configure_device

# Configure for automatic device selection
config = configure_device(
    preferred_device='auto',  # 'cpu', 'cuda', 'mps', or 'auto'
    gpu_memory_fraction=0.8,
    enable_mixed_precision=True
)

# Print device information
config.print_device_info()
```

## 📊 Available Models

### Statistical Models
- **ARIMA**: AutoRegressive Integrated Moving Average
- **SARIMAX**: Seasonal ARIMA with eXogenous variables

### Machine Learning Models
- **XGBoost**: Gradient boosting with GPU support
- **LightGBM**: Fast gradient boosting with GPU support

### Deep Learning Models
- **N-BEATS**: Neural Basis Expansion Analysis for Time Series
- **TCN**: Temporal Convolutional Networks

### Specialized Models
- **Prophet**: Facebook's forecasting tool with holidays and seasonality

## 🏗️ Project Structure

```
revenue_forecasting/
├── src/
│   ├── data/
│   │   ├── data_loader.py          # Data loading and preprocessing
│   │   └── sample_data_generator.py # Generate synthetic datasets
│   ├── models/
│   │   ├── base_model.py           # Abstract base class
│   │   ├── forecasting_model.py    # Main model factory
│   │   ├── statistical_models.py   # ARIMA, SARIMAX
│   │   ├── ml_models.py            # XGBoost, LightGBM
│   │   ├── deep_learning_models.py # N-BEATS, TCN
│   │   └── prophet_model.py        # Prophet implementation
│   └── utils/
│       └── device_config.py        # CPU/GPU configuration
├── tests/                          # Comprehensive unit tests
├── data/samples/                   # Sample datasets
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🧪 Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_forecasting_model.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📈 Sample Datasets

The framework includes 5 pre-generated sample datasets:

1. **daily_ecommerce_revenue**: 4 years of daily e-commerce data with promotions
2. **hourly_retail_revenue**: 3 months of hourly retail data with business patterns
3. **simple_daily_revenue**: 6 months of basic daily revenue for quick testing
4. **weekly_revenue**: Weekly aggregated data with promotional effects
5. **monthly_revenue**: Monthly aggregated data for long-term analysis

```python
# Access sample data
from data.sample_data_generator import create_sample_datasets
datasets = create_sample_datasets()

# View available datasets
for name, df in datasets.items():
    print(f"{name}: {df.shape} - {df['date'].min()} to {df['date'].max()}")
```

## ⚙️ Configuration

### Device Configuration
The framework automatically detects and configures the best available computing device:

```python
from utils.device_config import configure_device

# Auto-detect best device
config = configure_device('auto')

# Force CPU usage
config = configure_device('cpu')

# Use GPU if available
config = configure_device('cuda')  # NVIDIA GPU
config = configure_device('mps')   # Apple Silicon
```

### Model Configuration
Each model type supports specific parameters:

```python
# XGBoost with GPU acceleration
model = ForecastingModel('xgboost', 
                        n_estimators=1000,
                        max_depth=8,
                        learning_rate=0.01,
                        lags=[1, 2, 3, 7, 14, 30])

# Prophet with holidays
model = ForecastingModel('prophet',
                        seasonality_mode='multiplicative',
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        country_holidays='US')

# ARIMA with custom order
model = ForecastingModel('arima', order=(2, 1, 2))
```

## 🔄 Extending the Framework

### Adding a New Model

1. Create a new model class inheriting from `BaseForecastingModel`:

```python
from models.base_model import BaseForecastingModel

class MyCustomModel(BaseForecastingModel):
    def __init__(self, **kwargs):
        super().__init__("MyCustomModel", **kwargs)
    
    def fit(self, train_data, target_column, date_column, **kwargs):
        # Implement fitting logic
        self.is_fitted = True
        return self
    
    def predict(self, periods, confidence_level=0.95, **kwargs):
        # Implement prediction logic
        return predictions_dataframe
    
    def predict_with_exog(self, periods, exog_future=None, **kwargs):
        # Implement prediction with external variables
        return predictions_dataframe
```

2. Register the model:

```python
from models.forecasting_model import ModelFactory
ModelFactory.register_model('mycustom', MyCustomModel)

# Now you can use it
model = ForecastingModel('mycustom', param1=value1)
```

### Adding Custom Features

Extend the `DataLoader` class to add custom preprocessing:

```python
class MyDataLoader(DataLoader):
    def add_custom_features(self):
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Add your custom features
        self.data['custom_feature'] = self.data['revenue'].rolling(7).std()
        return self.data
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to add `src` to your Python path:
   ```python
   import sys
   sys.path.append('src')
   ```

2. **CUDA Issues**: If GPU detection fails, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Memory Issues**: Reduce `gpu_memory_fraction` in device configuration:
   ```python
   configure_device('cuda', gpu_memory_fraction=0.5)
   ```

4. **Missing Dependencies**: Install optional dependencies as needed:
   ```bash
   pip install xgboost lightgbm prophet statsmodels torch
   ```

## 📊 Performance Tips

1. **Use GPU acceleration** when available for XGBoost, LightGBM, and deep learning models
2. **Enable mixed precision** training for deep learning models to save memory
3. **Reduce data size** for initial experimentation using sample datasets
4. **Parallel processing** is automatically enabled for supported models
5. **Memory optimization** is handled automatically based on available resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `python -m pytest tests/ -v`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with support from Claude Code
- Utilizes industry-standard forecasting libraries
- Inspired by modern MLOps practices
- Designed for production-ready deployment

## 📬 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the test files for usage examples
- Review the comprehensive docstrings in the source code

---

**Ready to forecast your revenue? Get started with the quick start guide above!** 🚀