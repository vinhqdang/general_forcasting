"""
Simplified unit tests for device configuration (without external dependencies).
"""

import pytest

import sys
sys.path.append('src')

from utils.device_config import (
    DeviceConfig, 
    get_device_config, 
    set_device_config,
    configure_device,
    get_model_device_params
)


class TestDeviceConfigBasic:
    """Basic test cases for DeviceConfig class without external dependencies."""
    
    def test_init_default(self):
        """Test default initialization."""
        config = DeviceConfig()
        
        assert config.preferred_device == 'auto'
        assert config.gpu_memory_fraction == 0.8
        assert config.enable_mixed_precision is True
        assert isinstance(config.device_info, dict)
        assert 'cpu' in config.device_info
        assert config.selected_device is not None
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = DeviceConfig(
            preferred_device='cpu',
            gpu_memory_fraction=0.5,
            enable_mixed_precision=False
        )
        
        assert config.preferred_device == 'cpu'
        assert config.gpu_memory_fraction == 0.5
        assert config.enable_mixed_precision is False
        assert config.selected_device == 'cpu'
    
    def test_cpu_always_available(self):
        """Test that CPU is always detected as available."""
        config = DeviceConfig()
        
        assert 'cpu' in config.device_info
        assert config.device_info['cpu']['available'] is True
        assert config.device_info['cpu']['type'] == 'cpu'
    
    def test_device_selection_cpu(self):
        """Test device selection with CPU preference."""
        config = DeviceConfig(preferred_device='cpu')
        assert config.selected_device == 'cpu'
    
    def test_device_selection_cuda_fallback(self):
        """Test CUDA preference falls back to CPU when unavailable."""
        config = DeviceConfig(preferred_device='cuda')
        # Without torch/CUDA, should fallback to CPU
        assert config.selected_device == 'cpu'
    
    def test_device_selection_auto_cpu_only(self):
        """Test auto device selection with only CPU available."""
        config = DeviceConfig(preferred_device='auto')
        # Without GPU libraries, should select CPU
        assert config.selected_device == 'cpu'
    
    def test_get_xgboost_params_cpu(self):
        """Test getting XGBoost parameters for CPU."""
        config = DeviceConfig(preferred_device='cpu')
        params = config.get_xgboost_params()
        
        # Should be empty for CPU without XGBoost
        assert isinstance(params, dict)
    
    def test_get_lightgbm_params_cpu(self):
        """Test getting LightGBM parameters for CPU."""
        config = DeviceConfig(preferred_device='cpu')
        params = config.get_lightgbm_params()
        
        # Should be empty for CPU without LightGBM
        assert isinstance(params, dict)
    
    def test_get_device_summary(self):
        """Test getting device summary."""
        config = DeviceConfig(preferred_device='cpu')
        summary = config.get_device_summary()
        
        assert 'selected_device' in summary
        assert 'available_devices' in summary
        assert 'device_info' in summary
        assert 'gpu_memory_fraction' in summary
        assert 'mixed_precision_enabled' in summary
        
        assert summary['selected_device'] == 'cpu'
        assert isinstance(summary['available_devices'], list)
    
    def test_invalid_device_preference(self):
        """Test invalid device preference."""
        config = DeviceConfig(preferred_device='invalid_device')
        # Should fallback to CPU
        assert config.selected_device == 'cpu'
    
    def test_gpu_memory_fraction_bounds(self):
        """Test GPU memory fraction with different values."""
        # Test valid fraction
        config = DeviceConfig(gpu_memory_fraction=0.5)
        assert config.gpu_memory_fraction == 0.5
        
        # Test edge values
        config_low = DeviceConfig(gpu_memory_fraction=0.1)
        assert config_low.gpu_memory_fraction == 0.1
        
        config_high = DeviceConfig(gpu_memory_fraction=1.0)
        assert config_high.gpu_memory_fraction == 1.0
    
    def test_case_insensitive_device_names(self):
        """Test case insensitive device names."""
        config_upper = DeviceConfig(preferred_device='CPU')
        config_lower = DeviceConfig(preferred_device='cpu')
        
        assert config_upper.selected_device == config_lower.selected_device


class TestGlobalDeviceConfigBasic:
    """Test global device configuration management."""
    
    def test_get_device_config_default(self):
        """Test getting default global device config."""
        # Reset global config
        set_device_config(None)
        
        config = get_device_config()
        assert isinstance(config, DeviceConfig)
        assert config.preferred_device == 'auto'
    
    def test_set_and_get_device_config(self):
        """Test setting and getting global device config."""
        custom_config = DeviceConfig(preferred_device='cpu')
        set_device_config(custom_config)
        
        retrieved_config = get_device_config()
        assert retrieved_config is custom_config
        assert retrieved_config.preferred_device == 'cpu'
    
    def test_configure_device(self):
        """Test configuring device globally."""
        config = configure_device(
            preferred_device='cpu',
            gpu_memory_fraction=0.6,
            enable_mixed_precision=False
        )
        
        assert isinstance(config, DeviceConfig)
        assert config.preferred_device == 'cpu'
        assert config.gpu_memory_fraction == 0.6
        assert config.enable_mixed_precision is False
        
        # Should be set as global config
        global_config = get_device_config()
        assert global_config is config
    
    def test_get_model_device_params_xgboost(self):
        """Test getting model device params for XGBoost."""
        configure_device(preferred_device='cpu')
        params = get_model_device_params('xgboost')
        
        assert isinstance(params, dict)
    
    def test_get_model_device_params_lightgbm(self):
        """Test getting model device params for LightGBM."""
        configure_device(preferred_device='cpu')
        params = get_model_device_params('lightgbm')
        
        assert isinstance(params, dict)
    
    def test_get_model_device_params_pytorch(self):
        """Test getting model device params for PyTorch."""
        configure_device(preferred_device='cpu')
        params = get_model_device_params('pytorch')
        
        assert isinstance(params, dict)
    
    def test_get_model_device_params_unknown(self):
        """Test getting model device params for unknown model."""
        params = get_model_device_params('unknown_model')
        assert params == {}


class TestDeviceConfigFunctionality:
    """Test device configuration functionality."""
    
    def test_print_device_info(self, capsys):
        """Test printing device information."""
        config = DeviceConfig(preferred_device='cpu')
        config.print_device_info()
        
        captured = capsys.readouterr()
        assert "Device Configuration" in captured.out
        assert "Selected Device: cpu" in captured.out
        assert "Available Devices:" in captured.out
    
    def test_get_torch_device_no_torch(self):
        """Test getting PyTorch device when torch not available."""
        config = DeviceConfig()
        result = config.get_torch_device()
        # Without torch, should return None
        assert result is None
    
    def test_configure_torch_memory_no_cuda(self):
        """Test configuring torch memory without CUDA."""
        config = DeviceConfig(preferred_device='cpu')
        # Should not raise error even without torch/CUDA
        config.configure_torch_memory()
        # No assertions needed, just testing it doesn't crash


if __name__ == '__main__':
    pytest.main([__file__])