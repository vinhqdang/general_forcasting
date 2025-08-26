"""
Device configuration utilities for CPU/GPU selection.
"""

import logging
from typing import Optional, Dict, Any
import warnings

logger = logging.getLogger(__name__)


class DeviceConfig:
    """Device configuration manager for CPU/GPU selection."""
    
    def __init__(self, 
                 preferred_device: str = 'auto',
                 gpu_memory_fraction: float = 0.8,
                 enable_mixed_precision: bool = True):
        """
        Initialize device configuration.
        
        Args:
            preferred_device: 'auto', 'cpu', 'cuda', 'mps'
            gpu_memory_fraction: Fraction of GPU memory to use
            enable_mixed_precision: Whether to enable mixed precision training
        """
        self.preferred_device = preferred_device.lower()
        self.gpu_memory_fraction = gpu_memory_fraction
        self.enable_mixed_precision = enable_mixed_precision
        
        self.device_info = self._detect_available_devices()
        self.selected_device = self._select_device()
        
        logger.info(f"Selected device: {self.selected_device}")
        if self.device_info:
            logger.info(f"Available devices: {list(self.device_info.keys())}")
    
    def _detect_available_devices(self) -> Dict[str, Dict[str, Any]]:
        """Detect available computing devices."""
        devices = {}
        
        # Always have CPU
        devices['cpu'] = {
            'available': True,
            'type': 'cpu',
            'name': 'CPU',
            'memory': 'System RAM'
        }
        
        # Check for CUDA (NVIDIA GPU)
        try:
            import torch
            if torch.cuda.is_available():
                cuda_info = {}
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    cuda_info[f'cuda:{i}'] = {
                        'available': True,
                        'type': 'cuda',
                        'name': props.name,
                        'memory': f'{props.total_memory / 1024**3:.1f} GB',
                        'compute_capability': f'{props.major}.{props.minor}'
                    }
                devices.update(cuda_info)
        except ImportError:
            logger.warning("PyTorch not available - CUDA detection skipped")
        except Exception as e:
            logger.warning(f"CUDA detection failed: {str(e)}")
        
        # Check for MPS (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices['mps'] = {
                    'available': True,
                    'type': 'mps',
                    'name': 'Apple MPS',
                    'memory': 'Unified Memory'
                }
        except (ImportError, AttributeError):
            pass
        except Exception as e:
            logger.warning(f"MPS detection failed: {str(e)}")
        
        # Check for XGBoost GPU support
        try:
            import xgboost as xgb
            if 'gpu_hist' in xgb.XGBRegressor().get_params():
                if 'cuda:0' in devices:
                    devices['cuda:0']['xgboost_gpu'] = True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"XGBoost GPU detection failed: {str(e)}")
        
        # Check for LightGBM GPU support
        try:
            import lightgbm as lgb
            # LightGBM GPU support detection is more complex
            # For now, assume it's available if CUDA is available
            if any('cuda' in device for device in devices):
                for device_name in devices:
                    if device_name.startswith('cuda'):
                        devices[device_name]['lightgbm_gpu'] = True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"LightGBM GPU detection failed: {str(e)}")
        
        return devices
    
    def _select_device(self) -> str:
        """Select the best available device based on preferences."""
        if self.preferred_device == 'cpu':
            return 'cpu'
        
        elif self.preferred_device == 'cuda':
            cuda_devices = [d for d in self.device_info if d.startswith('cuda')]
            if cuda_devices:
                return cuda_devices[0]  # Use first CUDA device
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
        
        elif self.preferred_device == 'mps':
            if 'mps' in self.device_info:
                return 'mps'
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return 'cpu'
        
        elif self.preferred_device == 'auto':
            # Auto-select best device
            # Priority: CUDA > MPS > CPU
            cuda_devices = [d for d in self.device_info if d.startswith('cuda')]
            if cuda_devices:
                return cuda_devices[0]
            elif 'mps' in self.device_info:
                return 'mps'
            else:
                return 'cpu'
        
        else:
            # Try to use the specified device directly
            if self.preferred_device in self.device_info:
                return self.preferred_device
            else:
                logger.warning(f"Device '{self.preferred_device}' not available, using CPU")
                return 'cpu'
    
    def get_torch_device(self):
        """Get PyTorch device object."""
        try:
            import torch
            return torch.device(self.selected_device)
        except ImportError:
            logger.warning("PyTorch not available")
            return None
    
    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters for the selected device."""
        params = {}
        
        if self.selected_device.startswith('cuda'):
            try:
                import xgboost as xgb
                # Check if GPU training is supported
                if self.device_info.get(self.selected_device, {}).get('xgboost_gpu', False):
                    params['tree_method'] = 'gpu_hist'
                    params['gpu_id'] = int(self.selected_device.split(':')[1]) if ':' in self.selected_device else 0
                    logger.info("Using XGBoost GPU acceleration")
                else:
                    logger.info("XGBoost GPU not available, using CPU")
            except ImportError:
                logger.warning("XGBoost not available")
        
        return params
    
    def get_lightgbm_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters for the selected device."""
        params = {}
        
        if self.selected_device.startswith('cuda'):
            try:
                import lightgbm as lgb
                if self.device_info.get(self.selected_device, {}).get('lightgbm_gpu', False):
                    params['device'] = 'gpu'
                    params['gpu_device_id'] = int(self.selected_device.split(':')[1]) if ':' in self.selected_device else 0
                    logger.info("Using LightGBM GPU acceleration")
                else:
                    logger.info("LightGBM GPU not available, using CPU")
            except ImportError:
                logger.warning("LightGBM not available")
        
        return params
    
    def configure_torch_memory(self):
        """Configure PyTorch memory settings."""
        if not self.selected_device.startswith('cuda'):
            return
        
        try:
            import torch
            if torch.cuda.is_available():
                # Set memory fraction
                device_id = int(self.selected_device.split(':')[1]) if ':' in self.selected_device else 0
                total_memory = torch.cuda.get_device_properties(device_id).total_memory
                target_memory = int(total_memory * self.gpu_memory_fraction)
                
                # Set memory limit (this is a simplified approach)
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device_id)
                
                logger.info(f"Set GPU memory fraction to {self.gpu_memory_fraction}")
                
                # Enable memory mapping for large datasets
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Failed to configure torch memory: {str(e)}")
    
    def get_device_summary(self) -> Dict[str, Any]:
        """Get a summary of device configuration."""
        return {
            'selected_device': self.selected_device,
            'available_devices': list(self.device_info.keys()),
            'device_info': self.device_info,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'mixed_precision_enabled': self.enable_mixed_precision
        }
    
    def print_device_info(self):
        """Print detailed device information."""
        print("=== Device Configuration ===")
        print(f"Selected Device: {self.selected_device}")
        print(f"Mixed Precision: {'Enabled' if self.enable_mixed_precision else 'Disabled'}")
        
        if self.selected_device.startswith('cuda'):
            print(f"GPU Memory Fraction: {self.gpu_memory_fraction}")
        
        print("\nAvailable Devices:")
        for device_name, info in self.device_info.items():
            print(f"  {device_name}: {info['name']} ({info['memory']})")
            if 'compute_capability' in info:
                print(f"    Compute Capability: {info['compute_capability']}")
            if info.get('xgboost_gpu'):
                print(f"    XGBoost GPU: Supported")
            if info.get('lightgbm_gpu'):
                print(f"    LightGBM GPU: Supported")


# Global device configuration instance
_global_device_config = None


def get_device_config() -> DeviceConfig:
    """Get the global device configuration."""
    global _global_device_config
    if _global_device_config is None:
        _global_device_config = DeviceConfig()
    return _global_device_config


def set_device_config(device_config: DeviceConfig):
    """Set the global device configuration."""
    global _global_device_config
    _global_device_config = device_config


def configure_device(preferred_device: str = 'auto', 
                    gpu_memory_fraction: float = 0.8,
                    enable_mixed_precision: bool = True) -> DeviceConfig:
    """
    Configure global device settings.
    
    Args:
        preferred_device: Preferred device ('auto', 'cpu', 'cuda', 'mps')
        gpu_memory_fraction: Fraction of GPU memory to use
        enable_mixed_precision: Whether to enable mixed precision
        
    Returns:
        DeviceConfig instance
    """
    config = DeviceConfig(
        preferred_device=preferred_device,
        gpu_memory_fraction=gpu_memory_fraction,
        enable_mixed_precision=enable_mixed_precision
    )
    set_device_config(config)
    return config


def get_model_device_params(model_type: str) -> Dict[str, Any]:
    """
    Get device-specific parameters for a model type.
    
    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'pytorch', etc.)
        
    Returns:
        Dictionary of device-specific parameters
    """
    config = get_device_config()
    
    if model_type.lower() == 'xgboost':
        return config.get_xgboost_params()
    elif model_type.lower() == 'lightgbm':
        return config.get_lightgbm_params()
    elif model_type.lower() in ['pytorch', 'nbeats', 'tcn', 'deepar']:
        device = config.get_torch_device()
        return {'device': device} if device else {}
    else:
        return {}


if __name__ == "__main__":
    # Example usage
    config = configure_device('auto')
    config.print_device_info()
    
    print("\nXGBoost params:", get_model_device_params('xgboost'))
    print("LightGBM params:", get_model_device_params('lightgbm'))
    print("PyTorch params:", get_model_device_params('pytorch'))