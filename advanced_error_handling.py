"""
Advanced Error Handling System for Crypto Bot
Comprehensive error management, retry mechanisms, and graceful degradation
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import asyncio
import json
import threading
from collections import defaultdict, deque
import functools

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"    # Bot cannot continue
    HIGH = "high"           # Feature unavailable
    MEDIUM = "medium"       # Degraded performance
    LOW = "low"             # Minor issues

class ErrorCategory(Enum):
    """Error categories for better organization"""
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    DATA_PROCESSING = "data_processing"
    AI_ANALYSIS = "ai_analysis"
    CHART_GENERATION = "chart_generation"
    USER_INPUT = "user_input"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"
    RATE_LIMIT = "rate_limit"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[int] = None
    symbol: Optional[str] = None
    function_name: str = ""
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    timeout: float = 30.0

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == 'HALF_OPEN':
                logger.info("Circuit breaker returning to CLOSED state")
                self.state = 'CLOSED'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                self.last_failure_time = time.time()
                logger.error(f"Circuit breaker OPENED for {func.__name__}")
            
            raise e

class ErrorRecoveryStrategy:
    """Different strategies for error recovery"""
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True) -> float:
        """Calculate exponential backoff delay"""
        delay = base_delay * (2 ** attempt)
        delay = min(delay, max_delay)
        
        if jitter:
            # Add random jitter to prevent thundering herd
            import random
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    @staticmethod
    def linear_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate linear backoff delay"""
        delay = base_delay * (attempt + 1)
        return min(delay, max_delay)
    
    @staticmethod
    def immediate_retry() -> float:
        """Immediate retry without delay"""
        return 0.0

class AdvancedErrorManager:
    """Comprehensive error management system"""
    
    def __init__(self):
        self.error_history = deque(maxlen=1000)
        self.error_stats = defaultdict(int)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        self.error_contexts = {}
        self.fallback_functions = {}
        
        # Initialize default circuit breakers
        self._init_circuit_breakers()
        
        # Initialize default recovery strategies
        self._init_recovery_strategies()
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for different services"""
        self.circuit_breakers = {
            'binance_api': CircuitBreaker(failure_threshold=3, recovery_timeout=120.0),
            'groq_api': CircuitBreaker(failure_threshold=5, recovery_timeout=60.0),
            'database': CircuitBreaker(failure_threshold=10, recovery_timeout=30.0),
            'chart_generation': CircuitBreaker(failure_threshold=5, recovery_timeout=90.0),
            'ai_analysis': CircuitBreaker(failure_threshold=3, recovery_timeout=180.0),
            'external_news': CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        }
    
    def _init_recovery_strategies(self):
        """Initialize recovery strategies for different error types"""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: RetryConfig(max_attempts=3, base_delay=2.0, max_delay=30.0),
            ErrorCategory.API: RetryConfig(max_attempts=2, base_delay=5.0, max_delay=60.0),
            ErrorCategory.DATABASE: RetryConfig(max_attempts=2, base_delay=1.0, max_delay=10.0),
            ErrorCategory.RATE_LIMIT: RetryConfig(max_attempts=5, base_delay=10.0, max_delay=300.0),
            ErrorCategory.EXTERNAL_SERVICE: RetryConfig(max_attempts=3, base_delay=5.0, max_delay=120.0),
            ErrorCategory.DATA_PROCESSING: RetryConfig(max_attempts=2, base_delay=1.0, max_delay=20.0),
            ErrorCategory.AI_ANALYSIS: RetryConfig(max_attempts=2, base_delay=3.0, max_delay=45.0)
        }
    
    def create_error_context(self, 
                           category: ErrorCategory, 
                           severity: ErrorSeverity,
                           message: str,
                           details: Dict[str, Any] = None,
                           **kwargs) -> ErrorContext:
        """Create error context for tracking"""
        error_id = f"{category.value}_{int(time.time() * 1000)}_{len(self.error_history)}"
        
        context = ErrorContext(
            error_id=error_id,
            category=category,
            severity=severity,
            message=message,
            details=details or {},
            function_name=kwargs.get('function_name', ''),
            user_id=kwargs.get('user_id'),
            symbol=kwargs.get('symbol')
        )
        
        self.error_contexts[error_id] = context
        return context
    
    def log_error(self, error_context: ErrorContext, exception: Exception = None):
        """Log error with context"""
        # Add exception details if provided
        if exception:
            error_context.details['exception_type'] = type(exception).__name__
            error_context.details['exception_message'] = str(exception)
            error_context.details['traceback'] = traceback.format_exc()
        
        # Update statistics
        self.error_stats[error_context.category.value] += 1
        self.error_stats[f"{error_context.severity.value}_{error_context.category.value}"] += 1
        
        # Add to history
        self.error_history.append(error_context)
        
        # Log based on severity
        log_message = f"[{error_context.category.value.upper()}] {error_context.message}"
        if error_context.details:
            log_message += f" | Details: {error_context.details}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def register_fallback(self, service: str, fallback_func: Callable):
        """Register fallback function for service"""
        self.fallback_functions[service] = fallback_func
    
    def get_fallback(self, service: str, *args, **kwargs):
        """Execute fallback function for service"""
        if service in self.fallback_functions:
            try:
                return self.fallback_functions[service](*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback function failed for {service}: {e}")
        return None

# Global error manager instance
error_manager = AdvancedErrorManager()

# Decorators for different error handling patterns
def handle_errors(category: ErrorCategory, 
                  severity: ErrorSeverity = ErrorSeverity.HIGH,
                  retry_config: RetryConfig = None,
                  circuit_breaker: str = None,
                  fallback_service: str = None):
    """Decorator for comprehensive error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create error context
            error_context = error_manager.create_error_context(
                category=category,
                severity=severity,
                message=f"Error in {func.__name__}",
                function_name=func.__name__,
                details={'args': str(args)[:200], 'kwargs': str(kwargs)[:200]}
            )
            
            # Determine retry config
            retry_cfg = retry_config or error_manager.recovery_strategies.get(category, RetryConfig())
            
            # Get circuit breaker if specified
            cb = error_manager.circuit_breakers.get(circuit_breaker) if circuit_breaker else None
            
            # Retry logic
            last_exception = None
            for attempt in range(retry_cfg.max_attempts):
                try:
                    # Execute with circuit breaker if specified
                    if cb:
                        result = cb.call(func, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Success - reset retry count
                    error_context.retry_count = attempt
                    return result
                    
                except Exception as e:
                    last_exception = e
                    error_context.details['last_exception'] = str(e)
                    
                    # Check if this is the last attempt
                    if attempt == retry_cfg.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = ErrorRecoveryStrategy.exponential_backoff(
                        attempt, retry_cfg.base_delay, retry_cfg.max_delay, retry_cfg.jitter
                    )
                    
                    error_context.details[f'attempt_{attempt+1}_delay'] = delay
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            # All retries failed - try fallback if available
            if fallback_service and fallback_service in error_manager.fallback_functions:
                logger.warning(f"All retries failed for {func.__name__}. Trying fallback for {fallback_service}")
                try:
                    fallback_result = error_manager.get_fallback(fallback_service, *args, **kwargs)
                    if fallback_result is not None:
                        logger.info(f"Fallback successful for {func.__name__}")
                        return fallback_result
                except Exception as fallback_e:
                    logger.error(f"Fallback failed for {func.__name__}: {fallback_e}")
            
            # Log final error
            error_context.details['final_exception'] = str(last_exception)
            error_context.details['total_attempts'] = retry_cfg.max_attempts
            error_manager.log_error(error_context, last_exception)
            
            # Raise based on severity
            if severity == ErrorSeverity.CRITICAL:
                raise
            elif severity == ErrorSeverity.HIGH:
                # Return None for high severity errors instead of raising
                return None
            else:
                # Return default value for medium/low severity
                return None
        
        return wrapper
    return decorator

def graceful_degradation(fallback_values: Dict[str, Any] = None):
    """Decorator for graceful degradation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to get fallback value
                fallback_key = func.__name__
                fallback_value = fallback_values.get(fallback_key, fallback_values.get('default', None))
                
                if fallback_value is not None:
                    logger.warning(f"Using fallback value for {func.__name__}: {fallback_value}")
                    return fallback_value
                else:
                    logger.warning(f"No fallback available for {func.__name__}")
                    return None
        return wrapper
    return decorator

def time_limit(seconds: float):
    """Decorator to enforce time limits on functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator

def circuit_breaker(service_name: str):
    """Decorator to apply circuit breaker pattern"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cb = error_manager.circuit_breakers.get(service_name)
            if not cb:
                logger.warning(f"No circuit breaker found for service {service_name}")
                return func(*args, **kwargs)
            
            try:
                return cb.call(func, *args, **kwargs)
            except Exception as e:
                error_context = error_manager.create_error_context(
                    category=ErrorCategory.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    message=f"Circuit breaker triggered for {service_name}",
                    function_name=func.__name__,
                    details={'exception': str(e)}
                )
                error_manager.log_error(error_context)
                
                # Try fallback
                if service_name in error_manager.fallback_functions:
                    logger.info(f"Attempting fallback for {service_name}")
                    return error_manager.get_fallback(service_name, *args, **kwargs)
                
                raise e
        
        return wrapper
    return decorator

# Utility functions for common error scenarios
def handle_api_error(error_context: ErrorContext, api_name: str, response=None):
    """Handle API-specific errors with smart retry logic"""
    if response:
        status_code = response.status_code if hasattr(response, 'status_code') else None
        error_context.details['status_code'] = status_code
        
        if status_code == 429:  # Rate limit
            error_context.category = ErrorCategory.RATE_LIMIT
            error_context.severity = ErrorSeverity.MEDIUM
        elif status_code >= 500:  # Server error
            error_context.severity = ErrorSeverity.MEDIUM
        elif status_code >= 400:  # Client error
            error_context.severity = ErrorSeverity.LOW
            error_context.details['client_error'] = True
    
    error_manager.log_error(error_context)

def handle_database_error(error_context: ErrorContext, query: str = None, params: tuple = None):
    """Handle database-specific errors"""
    if query:
        error_context.details['query'] = query[:200]  # Truncate for safety
        error_context.details['params'] = str(params)[:200] if params else None
    
    error_context.category = ErrorCategory.DATABASE
    error_context.severity = ErrorSeverity.HIGH
    error_manager.log_error(error_context)

def handle_data_error(error_context: ErrorContext, data_source: str = None):
    """Handle data processing errors"""
    if data_source:
        error_context.details['data_source'] = data_source
    
    error_context.category = ErrorCategory.DATA_PROCESSING
    error_context.severity = ErrorSeverity.MEDIUM
    error_manager.log_error(error_context)

# Context managers for error handling
class ErrorContext:
    """Context manager for error handling with automatic cleanup"""
    
    def __init__(self, category: ErrorCategory, severity: ErrorSeverity, message: str, **kwargs):
        self.category = category
        self.severity = severity
        self.message = message
        self.kwargs = kwargs
        self.context = None
    
    def __enter__(self):
        self.context = error_manager.create_error_context(
            category=self.category,
            severity=self.severity,
            message=self.message,
            **self.kwargs
        )
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.context.details['exception_type'] = exc_type.__name__
            self.context.details['exception_message'] = str(exc_val)
            error_manager.log_error(self.context, exc_val)
        else:
            logger.info(f"Context completed successfully: {self.message}")

# Performance monitoring
class PerformanceMonitor:
    """Monitor function performance and detect issues"""
    
    def __init__(self):
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self.error_counts = defaultdict(int)
        self.slow_operations = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = func.__name__
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                self.call_counts[func_name] += 1
                self.total_times[func_name] += execution_time
                
                # Track slow operations
                if execution_time > 5.0:  # Operations taking > 5 seconds
                    self.slow_operations.append({
                        'function': func_name,
                        'execution_time': execution_time,
                        'timestamp': time.time()
                    })
                    
                    # Keep only last 100 slow operations
                    if len(self.slow_operations) > 100:
                        self.slow_operations = self.slow_operations[-100:]
                
                return result
                
            except Exception as e:
                # Record error
                self.error_counts[func_name] += 1
                execution_time = time.time() - start_time
                
                logger.error(f"Error in {func_name} after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    
    @property
    def stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for func_name in self.call_counts:
            stats[func_name] = {
                'call_count': self.call_counts[func_name],
                'total_time': self.total_times[func_name],
                'avg_time': self.total_times[func_name] / self.call_counts[func_name],
                'error_count': self.error_counts[func_name],
                'error_rate': self.error_counts[func_name] / self.call_counts[func_name] * 100
            }
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Example usage
if __name__ == "__main__":
    # Example of error handling with decorators
    
    @handle_errors(ErrorCategory.API, severity=ErrorSeverity.HIGH, retry_config=RetryConfig(max_attempts=3))
    @circuit_breaker('binance_api')
    @performance_monitor
    def fetch_crypto_data(symbol: str):
        """Example function with comprehensive error handling"""
        if symbol == 'INVALID':
            raise ValueError("Invalid symbol provided")
        # Simulate API call
        return {'symbol': symbol, 'price': 50000.0}
    
    # Test the error handling
    try:
        result = fetch_crypto_data('BTC')
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test with invalid symbol
    try:
        result = fetch_crypto_data('INVALID')
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Show performance stats
    print("\nPerformance Statistics:")
    for func, stats in performance_monitor.stats.items():
        print(f"{func}: {stats}")