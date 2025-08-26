"""
Unit tests for device configuration.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.append('src')

from utils.device_config import (
    DeviceConfig, 
    get_device_config, 
    set_device_config,
    configure_device,
    get_model_device_params
)


class TestDeviceConfig:
    """Test cases for DeviceConfig class."""
    
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
    
    def test_cuda_detection_available(self):
        """Test CUDA detection when available."""
        # Since torch is not installed, this will test CPU fallback
        config = DeviceConfig(preferred_device='cuda')
        
        # Should fallback to CPU
        assert config.selected_device == 'cpu'
        assert 'cpu' in config.device_info
    
    @patch('utils.device_config.torch')
    def test_cuda_detection_unavailable(self, mock_torch):
        """Test CUDA detection when unavailable."""
        mock_torch.cuda.is_available.return_value = False
        
        config = DeviceConfig()
        
        # Should only have CPU
        cuda_devices = [d for d in config.device_info if d.startswith('cuda')]
        assert len(cuda_devices) == 0
    
    def test_device_selection_cpu(self):
        """Test device selection with CPU preference."""
        config = DeviceConfig(preferred_device='cpu')
        assert config.selected_device == 'cpu'
    
    @patch('utils.device_config.torch')
    def test_device_selection_cuda_available(self, mock_torch):
        """Test device selection with CUDA preference when available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        mock_props = MagicMock()
        mock_props.name = 'Test GPU'
        mock_props.total_memory = 8 * 1024**3
        mock_props.major = 7
        mock_props.minor = 5
        
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        config = DeviceConfig(preferred_device='cuda')
        assert config.selected_device == 'cuda:0'
    
    def test_device_selection_cuda_unavailable(self):
        """Test device selection with CUDA preference when unavailable."""
        config = DeviceConfig(preferred_device='cuda')
        # Should fallback to CPU
        assert config.selected_device == 'cpu'
    
    @patch('utils.device_config.torch')
    def test_device_selection_auto_with_cuda(self, mock_torch):
        """Test auto device selection with CUDA available."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        
        mock_props = MagicMock()
        mock_props.name = 'Test GPU'
        mock_props.total_memory = 8 * 1024**3
        mock_props.major = 7
        mock_props.minor = 5
        
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        config = DeviceConfig(preferred_device='auto')
        assert config.selected_device == 'cuda:0'
    
    def test_device_selection_auto_cpu_only(self):
        """Test auto device selection with only CPU."""
        config = DeviceConfig(preferred_device='auto')
        assert config.selected_device == 'cpu'
    
    @patch('utils.device_config.torch')
    def test_get_torch_device(self, mock_torch):
        """Test getting PyTorch device object."""
        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device
        
        config = DeviceConfig(preferred_device='cpu')
        result = config.get_torch_device()
        
        mock_torch.device.assert_called_once_with('cpu')
        assert result == mock_device
    
    def test_get_torch_device_no_torch(self):
        """Test getting PyTorch device when torch not available."""
        with patch('utils.device_config.torch', side_effect=ImportError):
            config = DeviceConfig()
            result = config.get_torch_device()
            assert result is None
    
    @patch('utils.device_config.xgb')
    def test_get_xgboost_params_gpu(self, mock_xgb):
        """Test getting XGBoost parameters for GPU.""" 
        with patch('utils.device_config.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            
            mock_props = MagicMock()
            mock_props.name = 'Test GPU'
            mock_props.total_memory = 8 * 1024**3
            mock_props.major = 7
            mock_props.minor = 5
            
            mock_torch.cuda.get_device_properties.return_value = mock_props
            
            config = DeviceConfig(preferred_device='cuda')
            # Manually set xgboost_gpu flag for testing
            config.device_info['cuda:0']['xgboost_gpu'] = True
            
            params = config.get_xgboost_params()
            
            assert 'tree_method' in params
            assert params['tree_method'] == 'gpu_hist'
            assert 'gpu_id' in params
    
    def test_get_xgboost_params_cpu(self):
        """Test getting XGBoost parameters for CPU."""
        config = DeviceConfig(preferred_device='cpu')
        params = config.get_xgboost_params()
        
        # Should be empty for CPU
        assert 'tree_method' not in params
        assert 'gpu_id' not in params
    
    @patch('utils.device_config.lgb')
    def test_get_lightgbm_params_gpu(self, mock_lgb):
        """Test getting LightGBM parameters for GPU."""
        with patch('utils.device_config.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            
            mock_props = MagicMock()
            mock_props.name = 'Test GPU'
            mock_props.total_memory = 8 * 1024**3
            mock_props.major = 7
            mock_props.minor = 5
            
            mock_torch.cuda.get_device_properties.return_value = mock_props
            
            config = DeviceConfig(preferred_device='cuda')
            # Manually set lightgbm_gpu flag for testing
            config.device_info['cuda:0']['lightgbm_gpu'] = True
            
            params = config.get_lightgbm_params()
            
            assert 'device' in params
            assert params['device'] == 'gpu'
            assert 'gpu_device_id' in params
    
    def test_get_lightgbm_params_cpu(self):
        """Test getting LightGBM parameters for CPU."""
        config = DeviceConfig(preferred_device='cpu')
        params = config.get_lightgbm_params()
        
        # Should be empty for CPU
        assert 'device' not in params
        assert 'gpu_device_id' not in params
    
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


class TestGlobalDeviceConfig:
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
        # CPU config should return empty params
    
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


class TestDeviceConfigEdgeCases:
    """Test edge cases and error conditions."""
    
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
    
    @patch('utils.device_config.torch')
    def test_torch_cuda_exception(self, mock_torch):
        """Test handling of torch CUDA exceptions."""
        mock_torch.cuda.is_available.side_effect = Exception("CUDA error")
        
        # Should not crash, should fallback gracefully
        config = DeviceConfig()
        assert config.selected_device == 'cpu'


if __name__ == '__main__':
    pytest.main([__file__])