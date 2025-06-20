"""
Standardized Exception Handling System
Provides consistent error handling and user-friendly error messages
"""

import traceback
import logging
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager
from enum import Enum


class ErrorCategory(Enum):
    """Error categories for better error classification"""
    CONFIGURATION = "configuration"
    GPU_HARDWARE = "gpu_hardware"
    MODEL_IO = "model_io"
    ENVIRONMENT = "environment"
    TRAINING = "training"
    VIDEO_PROCESSING = "video_processing"
    NETWORK = "network"
    DATA_PROCESSING = "data_processing"


class DQNDDPGException(Exception):
    """Base exception for DQN-DDPG project
    
    All project-specific exceptions should inherit from this class.
    Provides structured error information and user-friendly messages.
    """
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory = ErrorCategory.CONFIGURATION,
                 details: Optional[Dict[str, Any]] = None,
                 user_message: Optional[str] = None,
                 suggestions: Optional[list] = None):
        """Initialize project exception
        
        Args:
            message: Technical error message
            category: Error category for classification
            details: Additional error details
            user_message: User-friendly error message
            suggestions: List of suggested solutions
        """
        super().__init__(message)
        self.category = category
        self.details = details or {}
        self.user_message = user_message or self._generate_user_message()
        self.suggestions = suggestions or []
        
    def _generate_user_message(self) -> str:
        """Generate user-friendly message based on category"""
        category_messages = {
            ErrorCategory.CONFIGURATION: "Configuration error occurred",
            ErrorCategory.GPU_HARDWARE: "GPU hardware issue detected",
            ErrorCategory.MODEL_IO: "Model loading/saving error",
            ErrorCategory.ENVIRONMENT: "Environment setup error",
            ErrorCategory.TRAINING: "Training process error",
            ErrorCategory.VIDEO_PROCESSING: "Video processing error",
            ErrorCategory.NETWORK: "Network architecture error",
            ErrorCategory.DATA_PROCESSING: "Data processing error"
        }
        return category_messages.get(self.category, "An error occurred")
    
    def get_error_info(self) -> Dict[str, Any]:
        """Get structured error information"""
        return {
            'category': self.category.value,
            'message': str(self),
            'user_message': self.user_message,
            'details': self.details,
            'suggestions': self.suggestions
        }


class ConfigurationError(DQNDDPGException):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        if config_path:
            kwargs.setdefault('details', {}).update({'config_path': config_path})
        super().__init__(message, **kwargs)


class GPUNotAvailableError(DQNDDPGException):
    """GPU not available or incompatible"""
    
    def __init__(self, message: str = "GPU is not available", **kwargs):
        kwargs.setdefault('category', ErrorCategory.GPU_HARDWARE)
        kwargs.setdefault('user_message', "GPU is not available. Switching to CPU mode.")
        kwargs.setdefault('suggestions', [
            "Check if CUDA is properly installed",
            "Verify GPU drivers are up to date",
            "Try running with --no-gpu flag",
            "Check if GPU is being used by another process"
        ])
        super().__init__(message, **kwargs)


class GPUMemoryError(DQNDDPGException):
    """GPU memory-related errors"""
    
    def __init__(self, message: str = "GPU memory error", memory_used: Optional[float] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.GPU_HARDWARE)
        kwargs.setdefault('user_message', "Insufficient GPU memory. Try reducing batch size.")
        if memory_used:
            kwargs.setdefault('details', {}).update({'memory_used_gb': memory_used})
        kwargs.setdefault('suggestions', [
            "Reduce batch size",
            "Enable gradient checkpointing",
            "Use CPU for training",
            "Close other GPU applications"
        ])
        super().__init__(message, **kwargs)


class ModelLoadError(DQNDDPGException):
    """Model loading errors"""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL_IO)
        kwargs.setdefault('user_message', "Failed to load model. File may be corrupted.")
        if model_path:
            kwargs.setdefault('details', {}).update({'model_path': model_path})
        kwargs.setdefault('suggestions', [
            "Check if model file exists",
            "Verify model file is not corrupted",
            "Ensure model architecture matches",
            "Try loading a different checkpoint"
        ])
        super().__init__(message, **kwargs)


class ModelSaveError(DQNDDPGException):
    """Model saving errors"""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL_IO)
        kwargs.setdefault('user_message', "Failed to save model. Check disk space and permissions.")
        if model_path:
            kwargs.setdefault('details', {}).update({'model_path': model_path})
        kwargs.setdefault('suggestions', [
            "Check available disk space",
            "Verify write permissions",
            "Ensure directory exists",
            "Try saving to a different location"
        ])
        super().__init__(message, **kwargs)


class EnvironmentError(DQNDDPGException):
    """Environment-related errors"""
    
    def __init__(self, message: str, env_name: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.ENVIRONMENT)
        kwargs.setdefault('user_message', "Environment setup failed. Check environment name and requirements.")
        if env_name:
            kwargs.setdefault('details', {}).update({'environment': env_name})
        kwargs.setdefault('suggestions', [
            "Check if environment name is correct",
            "Verify required dependencies are installed",
            "Try using a different environment",
            "Check Gym/Gymnasium version compatibility"
        ])
        super().__init__(message, **kwargs)


class TrainingError(DQNDDPGException):
    """Training process errors"""
    
    def __init__(self, message: str, episode: Optional[int] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.TRAINING)
        kwargs.setdefault('user_message', "Training error occurred. Check hyperparameters and data.")
        if episode is not None:
            kwargs.setdefault('details', {}).update({'episode': episode})
        kwargs.setdefault('suggestions', [
            "Check hyperparameters",
            "Verify training data",
            "Try reducing learning rate",
            "Check for numerical instabilities"
        ])
        super().__init__(message, **kwargs)


class VideoProcessingError(DQNDDPGException):
    """Video processing errors"""
    
    def __init__(self, message: str, video_path: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VIDEO_PROCESSING)
        kwargs.setdefault('user_message', "Video processing failed. Trying alternative method.")
        if video_path:
            kwargs.setdefault('details', {}).update({'video_path': video_path})
        kwargs.setdefault('suggestions', [
            "Check if FFmpeg is installed",
            "Try using a different video codec",
            "Reduce video quality/resolution",
            "Check available disk space"
        ])
        super().__init__(message, **kwargs)


class NetworkArchitectureError(DQNDDPGException):
    """Network architecture errors"""
    
    def __init__(self, message: str, network_type: Optional[str] = None, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('user_message', "Network architecture error. Check network configuration.")
        if network_type:
            kwargs.setdefault('details', {}).update({'network_type': network_type})
        kwargs.setdefault('suggestions', [
            "Check network architecture configuration",
            "Verify input/output dimensions",
            "Check activation functions",
            "Try simpler network architecture"
        ])
        super().__init__(message, **kwargs)


class ErrorHandler:
    """Centralized error handling utilities"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler
        
        Args:
            logger: Logger instance for error logging
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(self, 
                    error: Exception, 
                    context: Optional[str] = None,
                    reraise: bool = True) -> Optional[Dict[str, Any]]:
        """Handle error with logging and structured information
        
        Args:
            error: Exception to handle
            context: Additional context information
            reraise: Whether to reraise the exception
            
        Returns:
            Error information dictionary if not reraising
        """
        error_info = {}
        
        if isinstance(error, DQNDDPGException):
            error_info = error.get_error_info()
        else:
            error_info = {
                'category': 'unknown',
                'message': str(error),
                'user_message': f"An unexpected error occurred: {type(error).__name__}",
                'details': {'error_type': type(error).__name__},
                'suggestions': ["Check the logs for more details", "Try restarting the process"]
            }
        
        # Add context if provided
        if context:
            error_info['context'] = context
        
        # Log the error
        self.logger.error(
            f"Error in {context or 'unknown context'}: {error_info['message']}",
            extra={'error_info': error_info, 'traceback': traceback.format_exc()}
        )
        
        if reraise:
            raise error
        else:
            return error_info
    
    @contextmanager
    def error_context(self, 
                     operation: str,
                     fallback_result: Any = None,
                     suppress_errors: bool = False):
        """Context manager for error handling
        
        Args:
            operation: Description of the operation
            fallback_result: Result to return if error occurs and suppressed
            suppress_errors: Whether to suppress errors and return fallback
        """
        try:
            yield
        except Exception as e:
            error_info = self.handle_error(e, context=operation, reraise=False)
            
            if suppress_errors:
                self.logger.warning(f"Suppressed error in {operation}, using fallback")
                return fallback_result
            else:
                # Convert to appropriate project exception if needed
                if not isinstance(e, DQNDDPGException):
                    raise DQNDDPGException(
                        message=str(e),
                        category=ErrorCategory.CONFIGURATION,
                        details={'original_error': type(e).__name__}
                    ) from e
                else:
                    raise


class RecoveryStrategy:
    """Error recovery strategies"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """Initialize recovery strategy
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
    
    def with_retry(self, 
                   operation,
                   operation_name: str = "operation",
                   exceptions: tuple = (Exception,),
                   fallback=None):
        """Execute operation with retry logic
        
        Args:
            operation: Callable to execute
            operation_name: Name for logging
            exceptions: Exception types to catch and retry
            fallback: Fallback function if all retries fail
            
        Returns:
            Operation result or fallback result
        """
        import time
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation()
                if attempt > 0:
                    self.logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result
            
            except exceptions as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    self.logger.warning(
                        f"{operation_name} failed on attempt {attempt + 1}, "
                        f"retrying in {wait_time:.1f}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"{operation_name} failed after {self.max_retries + 1} attempts")
        
        # All retries failed
        if fallback is not None:
            self.logger.info(f"Using fallback for {operation_name}")
            return fallback()
        else:
            raise last_exception


# Utility functions for common error scenarios
def handle_gpu_error(func):
    """Decorator for handling GPU-related errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise GPUNotAvailableError(str(e)) from e
            elif "memory" in str(e).lower():
                raise GPUMemoryError(str(e)) from e
            else:
                raise
    return wrapper


def handle_model_io_error(func):
    """Decorator for handling model I/O errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise ModelLoadError(f"Model file not found: {e}") from e
        except PermissionError as e:
            raise ModelSaveError(f"Permission denied: {e}") from e
        except Exception as e:
            if "load" in func.__name__.lower():
                raise ModelLoadError(f"Failed to load model: {e}") from e
            elif "save" in func.__name__.lower():
                raise ModelSaveError(f"Failed to save model: {e}") from e
            else:
                raise
    return wrapper


def handle_environment_error(func):
    """Decorator for handling environment errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            env_name = None
            if args and hasattr(args[0], '__class__'):
                env_name = args[0].__class__.__name__
            raise EnvironmentError(str(e), env_name=env_name) from e
    return wrapper