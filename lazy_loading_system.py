"""
Lazy Loading System for Crypto Bot
Advanced lazy loading for indicators, data processing, and UI components
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps, lru_cache
import weakref
import gc
from concurrent.futures import ThreadPoolExecutor, Future
import queue
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)

class LoadingPriority(Enum):
    """Priority levels for lazy loading"""
    CRITICAL = 1      # Essential for immediate response
    HIGH = 2          # Important for main functionality
    MEDIUM = 3        # Nice to have features
    LOW = 4           # Background features
    BACKGROUND = 5    # Non-essential features

class LoadingState(Enum):
    """States for loading components"""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    EXPIRED = "expired"

@dataclass
class LazyComponent:
    """Represents a lazily loaded component"""
    name: str
    loader_func: Callable
    priority: LoadingPriority
    ttl: float = 300.0  # Time to live in seconds
    dependencies: List[str] = field(default_factory=list)
    auto_refresh: bool = False
    refresh_interval: float = 600.0  # 10 minutes
    
    # State tracking
    state: LoadingState = LoadingState.NOT_LOADED
    last_loaded: float = 0.0
    load_attempts: int = 0
    max_attempts: int = 3
    error_count: int = 0
    last_error: str = ""
    
    # Data storage
    data: Any = None
    loading_start_time: float = 0.0
    
    # Threading
    loading_future: Optional[Future] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

class LazyLoadingManager:
    """Manager for lazy loading components"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.components: Dict[str, LazyComponent] = {}
        self.component_states = defaultdict(list)
        self.loading_queue = queue.PriorityQueue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = True
        
        # Memory management
        self.max_memory_mb = 500  # Maximum memory usage in MB
        self.cleanup_threshold = 0.8  # Cleanup when 80% of memory limit reached
        
        # Background thread for cleanup and auto-refresh
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        # Cache for computed values
        self.computation_cache = OrderedDict(maxsize=1000)
        
        # Statistics
        self.stats = {
            'total_loads': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_load_time': 0.0,
            'components_created': 0,
            'components_cleaned': 0
        }
    
    def register_component(self, 
                          name: str,
                          loader_func: Callable,
                          priority: LoadingPriority = LoadingPriority.MEDIUM,
                          ttl: float = 300.0,
                          dependencies: List[str] = None,
                          auto_refresh: bool = False,
                          refresh_interval: float = 600.0):
        """Register a component for lazy loading"""
        
        component = LazyComponent(
            name=name,
            loader_func=loader_func,
            priority=priority,
            ttl=ttl,
            dependencies=dependencies or [],
            auto_refresh=auto_refresh,
            refresh_interval=refresh_interval
        )
        
        self.components[name] = component
        self.component_states[priority].append(name)
        self.stats['components_created'] += 1
        
        logger.info(f"Registered lazy component: {name} (priority: {priority.name})")
    
    def get_component(self, name: str, force_refresh: bool = False) -> Any:
        """Get component data, loading it if necessary"""
        
        if name not in self.components:
            raise ValueError(f"Component {name} not registered")
        
        component = self.components[name]
        
        with component.lock:
            # Check if component needs loading
            if self._should_load_component(component, force_refresh):
                self._load_component_async(component)
            
            # Wait for loading if in progress
            if component.state == LoadingState.LOADING:
                self._wait_for_loading(component)
            
            # Return data if successfully loaded
            if component.state == LoadingState.LOADED:
                self.stats['cache_hits'] += 1
                return component.data
            
            # If loading failed, try again once
            if component.state == LoadingState.FAILED and component.load_attempts < component.max_attempts:
                logger.warning(f"Retrying load for component {name}")
                self._load_component_async(component, force_refresh=True)
                self._wait_for_loading(component)
                
                if component.state == LoadingState.LOADED:
                    return component.data
            
            self.stats['cache_misses'] += 1
            return None
    
    def _should_load_component(self, component: LazyComponent, force_refresh: bool) -> bool:
        """Determine if component should be loaded"""
        
        current_time = time.time()
        
        # Force refresh takes precedence
        if force_refresh:
            return True
        
        # Load if not loaded yet
        if component.state == LoadingState.NOT_LOADED:
            return True
        
        # Load if expired
        if component.state == LoadingState.LOADED:
            time_since_load = current_time - component.last_loaded
            if time_since_load > component.ttl:
                component.state = LoadingState.EXPIRED
                return True
        
        # Load if auto-refresh is needed
        if (component.auto_refresh and 
            component.state == LoadingState.LOADED and
            current_time - component.last_loaded > component.refresh_interval):
            return True
        
        return False
    
    def _load_component_async(self, component: LazyComponent, force_refresh: bool = False):
        """Start loading component asynchronously"""
        
        if component.state == LoadingState.LOADING:
            return  # Already loading
        
        with component.lock:
            if component.state == LoadingState.LOADING:
                return  # Double check after acquiring lock
            
            # Check dependencies first
            if component.dependencies and not force_refresh:
                for dep_name in component.dependencies:
                    if dep_name in self.components:
                        dep_component = self.components[dep_name]
                        dep_data = self.get_component(dep_name)
                        if dep_data is None and dep_component.state == LoadingState.FAILED:
                            logger.warning(f"Dependency {dep_name} failed for {component.name}")
                            return
            
            # Mark as loading
            component.state = LoadingState.LOADING
            component.loading_start_time = time.time()
            component.load_attempts += 1
            
            # Submit to executor
            component.loading_future = self.executor.submit(self._load_component_sync, component)
    
    def _load_component_sync(self, component: LazyComponent):
        """Synchronous component loading"""
        
        try:
            start_time = time.time()
            
            # Call loader function
            if asyncio.iscoroutinefunction(component.loader_func):
                # For async functions, we need to run in event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're in async context, create new task
                        data = loop.create_task(component.loader_func())
                        # For now, just return the coroutine object
                        # In a real implementation, you'd await this
                        component.data = f"Async result placeholder for {component.name}"
                    else:
                        # Run in new event loop
                        data = asyncio.run(component.loader_func())
                except:
                    data = component.loader_func()  # Fallback to sync execution
            else:
                data = component.loader_func()
            
            # Update component state
            with component.lock:
                component.data = data
                component.state = LoadingState.LOADED
                component.last_loaded = time.time()
                component.error_count = 0
                component.last_error = ""
                
                # Update statistics
                load_time = time.time() - start_time
                self.stats['total_loads'] += 1
                self.stats['successful_loads'] += 1
                self.stats['total_load_time'] += load_time
                
                logger.debug(f"Component {component.name} loaded successfully in {load_time:.2f}s")
                
                # Trigger auto-refresh if needed
                if component.auto_refresh:
                    self._schedule_auto_refresh(component)
        
        except Exception as e:
            # Handle loading error
            with component.lock:
                component.state = LoadingState.FAILED
                component.error_count += 1
                component.last_error = str(e)
                
                self.stats['total_loads'] += 1
                self.stats['failed_loads'] += 1
                
                logger.error(f"Failed to load component {component.name}: {e}")
    
    def _wait_for_loading(self, component: LazyComponent, timeout: float = 30.0):
        """Wait for component loading to complete"""
        
        if component.loading_future:
            try:
                component.loading_future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Error waiting for {component.name}: {e}")
    
    def _schedule_auto_refresh(self, component: LazyComponent):
        """Schedule automatic refresh for component"""
        if component.auto_refresh:
            delay = max(60.0, component.refresh_interval * 0.8)  # Refresh 80% through TTL
            self.loading_queue.put((LoadingPriority.BACKGROUND, time.time() + delay, component.name))
    
    def _cleanup_loop(self):
        """Background cleanup and maintenance loop"""
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Process auto-refresh queue
                while not self.loading_queue.empty():
                    try:
                        priority, scheduled_time, component_name = self.loading_queue.get_nowait()
                        
                        if current_time >= scheduled_time and component_name in self.components:
                            component = self.components[component_name]
                            if (component.state == LoadingState.LOADED and 
                                current_time - component.last_loaded > component.refresh_interval * 0.8):
                                self._load_component_async(component)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error processing loading queue: {e}")
                
                # Cleanup expired components
                self._cleanup_expired_components()
                
                # Memory cleanup
                self._check_memory_usage()
                
                # Sleep before next cycle
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(30.0)
    
    def _cleanup_expired_components(self):
        """Clean up expired components"""
        
        current_time = time.time()
        expired_components = []
        
        for name, component in self.components.items():
            if component.state in [LoadingState.EXPIRED, LoadingState.FAILED]:
                time_since_load = current_time - component.last_loaded
                if time_since_load > component.ttl * 2:  # Remove after 2x TTL
                    expired_components.append(name)
        
        for name in expired_components:
            self._remove_component(name)
            self.stats['components_cleaned'] += 1
    
    def _remove_component(self, name: str):
        """Remove component and free memory"""
        
        if name in self.components:
            component = self.components[name]
            with component.lock:
                # Cancel loading if in progress
                if component.loading_future and not component.loading_future.done():
                    component.loading_future.cancel()
                
                # Clear data
                component.data = None
                component.state = LoadingState.NOT_LOADED
            
            # Remove from tracking
            del self.components[name]
            for priority_list in self.component_states.values():
                if name in priority_list:
                    priority_list.remove(name)
            
            logger.debug(f"Removed expired component: {name}")
    
    def _check_memory_usage(self):
        """Check memory usage and perform cleanup if needed"""
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb * self.cleanup_threshold:
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB. Triggering cleanup.")
                self._aggressive_cleanup()
        except ImportError:
            # psutil not available, skip memory check
            pass
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    def _aggressive_cleanup(self):
        """Perform aggressive cleanup of components"""
        
        # Remove low-priority expired components first
        for priority in [LoadingPriority.BACKGROUND, LoadingPriority.LOW]:
            for component_name in list(self.component_states[priority]):
                if component_name in self.components:
                    component = self.components[component_name]
                    if component.state in [LoadingState.EXPIRED, LoadingState.FAILED]:
                        self._remove_component(component_name)
                        self.stats['components_cleaned'] += 1
        
        # Force garbage collection
        gc.collect()
        
        # Clear computation cache if it exists
        if hasattr(self, 'computation_cache'):
            self.computation_cache.clear()
    
    def preload_component(self, name: str):
        """Preload a component in the background"""
        
        if name in self.components:
            component = self.components[name]
            if component.state == LoadingState.NOT_LOADED:
                logger.info(f"Preloading component: {name}")
                self._load_component_async(component)
    
    def preload_priority_components(self, priority: LoadingPriority, max_components: int = 5):
        """Preload multiple components of a specific priority"""
        
        if priority in self.component_states:
            components_to_preload = [
                name for name in self.component_states[priority][:max_components]
                if name in self.components and self.components[name].state == LoadingState.NOT_LOADED
            ]
            
            logger.info(f"Preloading {len(components_to_preload)} components of priority {priority.name}")
            
            for component_name in components_to_preload:
                self.preload_component(component_name)
    
    def get_component_status(self, name: str) -> Dict[str, Any]:
        """Get status information for a component"""
        
        if name not in self.components:
            return {'error': 'Component not found'}
        
        component = self.components[name]
        
        with component.lock:
            current_time = time.time()
            time_since_load = current_time - component.last_loaded if component.last_loaded > 0 else 0
            
            return {
                'name': name,
                'state': component.state.value,
                'priority': component.priority.name,
                'last_loaded': component.last_loaded,
                'time_since_load': time_since_load,
                'ttl': component.ttl,
                'is_expired': time_since_load > component.ttl,
                'load_attempts': component.load_attempts,
                'error_count': component.error_count,
                'last_error': component.last_error,
                'dependencies': component.dependencies,
                'auto_refresh': component.auto_refresh,
                'data_size': len(str(component.data)) if component.data else 0
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        # Component counts by state
        state_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        
        for component in self.components.values():
            state_counts[component.state.value] += 1
            priority_counts[component.priority.name] += 1
        
        return {
            'total_components': len(self.components),
            'state_counts': dict(state_counts),
            'priority_counts': dict(priority_counts),
            'stats': self.stats.copy(),
            'loading_queue_size': self.loading_queue.qsize(),
            'executor_info': {
                'max_workers': self.max_workers,
                'active_threads': threading.active_count()
            }
        }
    
    def shutdown(self):
        """Shutdown the lazy loading manager"""
        
        logger.info("Shutting down lazy loading manager...")
        
        self.is_running = False
        
        # Cancel all loading futures
        for component in self.components.values():
            if component.loading_future and not component.loading_future.done():
                component.loading_future.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Lazy loading manager shutdown complete")

# Decorator for lazy loading
def lazy_load(priority: LoadingPriority = LoadingPriority.MEDIUM, 
              ttl: float = 300.0,
              dependencies: List[str] = None,
              auto_refresh: bool = False,
              refresh_interval: float = 600.0):
    """Decorator to make functions lazy-loadable"""
    
    def decorator(func):
        func_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            # Get the lazy loading manager (assumed to be global)
            manager = globals().get('lazy_manager')
            if manager:
                return manager.get_component(func_name)
            else:
                # Fallback to direct execution
                return func(*args, **kwargs)
        
        # Register the function
        def register():
            manager = globals().get('lazy_manager')
            if manager:
                manager.register_component(
                    name=func_name,
                    loader_func=lambda: func(),
                    priority=priority,
                    ttl=ttl,
                    dependencies=dependencies,
                    auto_refresh=auto_refresh,
                    refresh_interval=refresh_interval
                )
        
        # Schedule registration after module load
        import atexit
        atexit.register(register)
        
        return wrapper
    return decorator

# Global lazy loading manager
lazy_manager = LazyLoadingManager()

# Example usage and testing
if __name__ == "__main__":
    
    # Example lazy-loaded functions
    
    @lazy_load(priority=LoadingPriority.HIGH, ttl=180.0)
    def load_market_data():
        """Simulated market data loading"""
        print("Loading market data...")
        time.sleep(2)  # Simulate network call
        return {'btc_price': 45000, 'eth_price': 3000, 'timestamp': time.time()}
    
    @lazy_load(priority=LoadingPriority.MEDIUM, ttl=300.0, 
               dependencies=['load_technical_indicators'])
    def load_smc_analysis():
        """Simulated SMC analysis"""
        print("Loading SMC analysis...")
        time.sleep(1)
        return {'signal': 'BUY', 'confidence': 85, 'order_blocks': []}
    
    @lazy_load(priority=LoadingPriority.MEDIUM, ttl=120.0)
    def load_technical_indicators():
        """Simulated technical indicators"""
        print("Loading technical indicators...")
        time.sleep(1.5)
        return {'rsi': 45, 'macd': 0.5, 'bb_position': 'middle'}
    
    # Test the system
    print("Testing Lazy Loading System")
    print("=" * 50)
    
    # First access - should load
    start_time = time.time()
    market_data = load_market_data()
    load_time = time.time() - start_time
    print(f"First access took: {load_time:.2f}s")
    print(f"Market data: {market_data}")
    
    # Second access - should use cache
    start_time = time.time()
    market_data2 = load_market_data()
    cache_time = time.time() - start_time
    print(f"Second access took: {cache_time:.2f}s")
    
    # Access dependent component
    print("\nAccessing SMC analysis (depends on technical indicators)...")
    start_time = time.time()
    smc_data = load_smc_analysis()
    smc_time = time.time() - start_time
    print(f"SMC analysis access took: {smc_time:.2f}s")
    print(f"SMC data: {smc_data}")
    
    # Show system status
    print("\nSystem Status:")
    status = lazy_manager.get_system_status()
    print(f"Total components: {status['total_components']}")
    print(f"Stats: {status['stats']}")
    
    # Shutdown
    lazy_manager.shutdown()