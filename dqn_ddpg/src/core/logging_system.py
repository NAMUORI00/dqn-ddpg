"""
Structured Logging System
Provides centralized, configurable logging with JSON support
"""

import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from .config_manager import ConfigManager


@dataclass
class LogRecord:
    """Structured log record"""
    timestamp: str
    level: str
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_log_record(cls, record: logging.LogRecord) -> 'LogRecord':
        """Create structured log record from logging.LogRecord"""
        return cls(
            timestamp=datetime.fromtimestamp(record.created).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
            extra=getattr(record, 'extra', None)
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_record = LogRecord.from_log_record(record)
        return json.dumps(asdict(log_record), ensure_ascii=False)


class StructuredLogger:
    """Structured logger with file and console handlers"""
    
    def __init__(self, 
                 name: str,
                 config_manager: Optional[ConfigManager] = None,
                 log_dir: Optional[str] = None):
        """Initialize structured logger
        
        Args:
            name: Logger name
            config_manager: Configuration manager for settings
            log_dir: Log directory override
        """
        self.name = name
        self.config_manager = config_manager or ConfigManager()
        
        # Get logging configuration
        try:
            log_config = self.config_manager.get_config('logging')
        except KeyError:
            # Default configuration if not found
            log_config = {
                'level': 'INFO',
                'console': True,
                'file': True,
                'structured': False,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'max_file_size': '10MB',
                'backup_count': 5
            }
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set up handlers
        if log_config.get('console', True):
            self._setup_console_handler(log_config)
        
        if log_config.get('file', True):
            self._setup_file_handler(log_config, log_dir)
    
    def _setup_console_handler(self, config: Dict[str, Any]) -> None:
        """Set up console handler"""
        console_handler = logging.StreamHandler()
        
        if config.get('structured', False):
            formatter = JSONFormatter()
        else:
            format_string = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(format_string)
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, config: Dict[str, Any], log_dir: Optional[str] = None) -> None:
        """Set up file handler with rotation"""
        # Determine log directory
        if log_dir is None:
            output_config = self.config_manager.get_output_config()
            log_dir = output_config.get_path('logs')
        else:
            log_dir = Path(log_dir)
        
        # Ensure log directory exists
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up rotating file handler
        log_file = log_dir / f"{self.name}.log"
        
        # Parse max file size
        max_size_str = config.get('max_file_size', '10MB')
        if max_size_str.endswith('MB'):
            max_bytes = int(max_size_str[:-2]) * 1024 * 1024
        elif max_size_str.endswith('KB'):
            max_bytes = int(max_size_str[:-2]) * 1024
        else:
            max_bytes = int(max_size_str)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=config.get('backup_count', 5),
            encoding='utf-8'
        )
        
        # Use JSON formatter for file logging if structured logging enabled
        if config.get('structured', False):
            formatter = JSONFormatter()
        else:
            format_string = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter(format_string)
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.logger.debug(message, extra={'extra': kwargs} if kwargs else None)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.logger.info(message, extra={'extra': kwargs} if kwargs else None)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.logger.warning(message, extra={'extra': kwargs} if kwargs else None)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self.logger.error(message, extra={'extra': kwargs} if kwargs else None)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self.logger.critical(message, extra={'extra': kwargs} if kwargs else None)
    
    def log_training_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
        """Log training metrics"""
        self.info("Training metrics", episode=episode, metrics=metrics)
    
    def log_performance(self, component: str, duration: float, **kwargs) -> None:
        """Log performance metrics"""
        self.info(f"Performance: {component}", duration=duration, **kwargs)
    
    def log_gpu_info(self, memory_used: float, memory_total: float, **kwargs) -> None:
        """Log GPU information"""
        self.info("GPU status", memory_used_gb=memory_used, memory_total_gb=memory_total, **kwargs)
    
    def log_error_with_context(self, error: Exception, context: str, **kwargs) -> None:
        """Log error with structured context"""
        self.error(
            f"Error in {context}: {str(error)}",
            error_type=type(error).__name__,
            context=context,
            **kwargs
        )


class LoggingManager:
    """Central logging management"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize logging manager"""
        self.config_manager = config_manager or ConfigManager()
        self._loggers: Dict[str, StructuredLogger] = {}
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create logger for component
        
        Args:
            name: Logger name (e.g., 'training', 'dqn_agent', 'video')
            
        Returns:
            StructuredLogger instance
        """
        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self.config_manager)
        return self._loggers[name]
    
    def get_training_logger(self) -> StructuredLogger:
        """Get logger for training components"""
        return self.get_logger('training')
    
    def get_agent_logger(self, agent_type: str) -> StructuredLogger:
        """Get logger for specific agent type"""
        return self.get_logger(f'{agent_type}_agent')
    
    def get_video_logger(self) -> StructuredLogger:
        """Get logger for video components"""
        return self.get_logger('video')
    
    def get_environment_logger(self) -> StructuredLogger:
        """Get logger for environment components"""
        return self.get_logger('environment')
    
    def set_global_level(self, level: str) -> None:
        """Set logging level for all loggers"""
        log_level = getattr(logging, level.upper())
        for logger in self._loggers.values():
            logger.logger.setLevel(log_level)
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up old log files"""
        import time
        
        output_config = self.config_manager.get_output_config()
        log_dir = output_config.get_path('logs')
        
        if not log_dir.exists():
            return
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for log_file in log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    print(f"Cleaned up old log file: {log_file}")
                except OSError:
                    pass  # Ignore errors


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logger(name: str, config_manager: Optional[ConfigManager] = None) -> StructuredLogger:
    """Get logger instance (convenience function)
    
    Args:
        name: Logger name
        config_manager: Configuration manager (optional)
        
    Returns:
        StructuredLogger instance
    """
    global _logging_manager
    
    if _logging_manager is None:
        _logging_manager = LoggingManager(config_manager)
    
    return _logging_manager.get_logger(name)


def setup_logging(config_manager: Optional[ConfigManager] = None) -> LoggingManager:
    """Set up logging system
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        LoggingManager instance
    """
    global _logging_manager
    _logging_manager = LoggingManager(config_manager)
    return _logging_manager


# Context manager for performance logging
class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: StructuredLogger, operation: str, **kwargs):
        """Initialize performance logger
        
        Args:
            logger: Logger instance
            operation: Operation description
            **kwargs: Additional context
        """
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time = None
    
    def __enter__(self):
        """Start timing"""
        import time
        self.start_time = time.time()
        self.logger.debug(f"Started: {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results"""
        import time
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            if exc_type is None:
                self.logger.log_performance(self.operation, duration, **self.context)
            else:
                self.logger.error(
                    f"Failed: {self.operation}",
                    duration=duration,
                    error_type=exc_type.__name__,
                    error_message=str(exc_val),
                    **self.context
                )


# Decorator for automatic performance logging
def log_performance(operation: str = None, logger_name: str = "performance"):
    """Decorator for automatic performance logging"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or f"{func.__module__}.{func.__name__}"
            logger = get_logger(logger_name)
            
            with PerformanceLogger(logger, op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator