"""
Advanced Memory Management System for Crypto Bot
Memory monitoring, optimization, and automatic cleanup
"""

import gc
import sys
import threading
import time
import logging
import psutil
import os
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import weakref
import tracemalloc
from functools import wraps, lru_cache
import resource

logger = logging.getLogger(__name__)

class MemoryLevel(Enum):
    """Memory usage levels"""
    EXCELLENT = "excellent"      # < 30% of limit
    GOOD = "good"                # 30-60% of limit
    WARNING = "warning"          # 60-80% of limit
    CRITICAL = "critical"        # 80-90% of limit
    DANGER = "danger"            # > 90% of limit

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    current_mb: float
    peak_mb: float
    level: MemoryLevel
    available_mb: float
    total_mb: float
    percentage: float
    swap_used_mb: float
    process_memory_mb: float
    gc_stats: Dict[str, Any]

@dataclass
class MemoryMonitor:
    """Memory monitoring data for a specific component"""
    name: str
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: float = 300.0  # Time to live in seconds
    priority: int = 0   # Lower number = higher priority (0 = highest)

class MemoryOptimizer:
    """Advanced memory optimization system"""
    
    def __init__(self, max_memory_mb: int = 1000, warning_threshold: float = 0.8):
        self.max_memory_mb = max_memory_mb
        self.warning_threshold = warning_threshold
        self.critical_threshold = 0.9
        
        # Memory tracking
        self.tracked_objects = {}  # name -> MemoryMonitor
        self.memory_history = deque(maxlen=1000)  # Keep last 1000 readings
        self.monitored_components = {}  # Component name -> memory usage
        self.memory_pools = {}  # Pool name -> pooled objects
        
        # Threading
        self.monitor_lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread = None
        
        # Garbage collection settings
        self.gc_threshold = (700, 10, 10)  # Default thresholds
        self.auto_gc_enabled = True
        self.gc_stats_history = deque(maxlen=100)
        
        # Caching settings
        self.max_cache_size = 1000
        self.cache_ttl = 3600  # 1 hour
        self.auto_cleanup_enabled = True
        
        # Statistics
        self.stats = {
            'optimizations_performed': 0,
            'objects_cleaned': 0,
            'memory_freed_mb': 0.0,
            'gc_collections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'components_tracked': 0
        }
        
        # Initialize memory monitoring
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize memory monitoring system"""
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        
        # Set garbage collection thresholds
        gc.set_threshold(*self.gc_threshold)
        
        logger.info("Memory optimization system initialized")
    
    def start_tracking(self, 
                      name: str, 
                      obj: Any, 
                      ttl: float = 300.0, 
                      priority: int = 0) -> bool:
        """Start tracking an object's memory usage"""
        
        try:
            import sys
            size_bytes = sys.getsizeof(obj)
            
            monitor = MemoryMonitor(
                name=name,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                priority=priority
            )
            
            with self.monitor_lock:
                self.tracked_objects[name] = monitor
                self.stats['components_tracked'] += 1
            
            logger.debug(f"Started tracking object: {name} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking object {name}: {e}")
            return False
    
    def stop_tracking(self, name: str) -> bool:
        """Stop tracking an object"""
        
        with self.monitor_lock:
            if name in self.tracked_objects:
                del self.tracked_objects[name]
                logger.debug(f"Stopped tracking object: {name}")
                return True
        
        return False
    
    def access_tracked_object(self, name: str) -> Optional[Any]:
        """Access a tracked object and update its metadata"""
        
        with self.monitor_lock:
            if name in self.tracked_objects:
                monitor = self.tracked_objects[name]
                monitor.last_accessed = time.time()
                monitor.access_count += 1
                return True
        
        return False
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        
        try:
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Current memory usage
            current_mb = memory.used / 1024 / 1024  # MB
            available_mb = memory.available / 1024 / 1024  # MB
            total_mb = memory.total / 1024 / 1024  # MB
            percentage = memory.percent
            
            # Peak memory (for this session)
            if hasattr(self, '_peak_memory_mb'):
                peak_mb = self._peak_memory_mb
            else:
                peak_mb = current_mb
                self._peak_memory_mb = peak_mb
            
            # Update peak memory
            if current_mb > peak_mb:
                peak_mb = current_mb
                self._peak_memory_mb = peak_mb
            
            # Determine memory level
            usage_ratio = current_mb / self.max_memory_mb
            if usage_ratio < 0.3:
                level = MemoryLevel.EXCELLENT
            elif usage_ratio < 0.6:
                level = MemoryLevel.GOOD
            elif usage_ratio < 0.8:
                level = MemoryLevel.WARNING
            elif usage_ratio < 0.9:
                level = MemoryLevel.CRITICAL
            else:
                level = MemoryLevel.DANGER
            
            # GC statistics
            gc_stats = gc.get_stats()
            
            return MemoryStats(
                current_mb=current_mb,
                peak_mb=peak_mb,
                level=level,
                available_mb=available_mb,
                total_mb=total_mb,
                percentage=percentage,
                swap_used_mb=swap.used / 1024 / 1024,
                process_memory_mb=process_memory,
                gc_stats=gc_stats
            )
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            # Return default stats
            return MemoryStats(
                current_mb=0.0,
                peak_mb=0.0,
                level=MemoryLevel.EXCELLENT,
                available_mb=0.0,
                total_mb=0.0,
                percentage=0.0,
                swap_used_mb=0.0,
                process_memory_mb=0.0,
                gc_stats={}
            )
    
    def _monitor_memory(self):
        """Background memory monitoring loop"""
        
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                
                # Add to history
                self.memory_history.append({
                    'timestamp': time.time(),
                    'current_mb': stats.current_mb,
                    'percentage': stats.percentage,
                    'level': stats.level.value
                })
                
                # Check for high memory usage
                if stats.percentage > 70:  # System memory > 70%
                    logger.warning(f"High memory usage detected: {stats.percentage:.1f}%")
                    self._perform_optimization(stats)
                
                # Auto cleanup if enabled
                if self.auto_cleanup_enabled:
                    self._auto_cleanup()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(30)
    
    def _perform_optimization(self, stats: MemoryStats):
        """Perform memory optimization based on current stats"""
        
        optimizations_performed = 0
        memory_freed = 0.0
        
        try:
            # 1. Garbage collection
            if self.auto_gc_enabled:
                collected = gc.collect()
                if collected > 0:
                    memory_freed += collected * 100  # Rough estimate
                    optimizations_performed += 1
                    self.stats['gc_collections'] += 1
            
            # 2. Clear expired tracked objects
            with self.monitor_lock:
                current_time = time.time()
                expired_objects = []
                
                for name, monitor in self.tracked_objects.items():
                    if current_time - monitor.last_accessed > monitor.ttl:
                        expired_objects.append(name)
                
                for name in expired_objects:
                    del self.tracked_objects[name]
                    memory_freed += self.tracked_objects[name].size_bytes if name in self.tracked_objects else 0
                    optimizations_performed += 1
            
            # 3. Clear weak references
            self._clear_weak_references()
            
            # 4. Clear cached data if memory is critical
            if stats.level == MemoryLevel.CRITICAL or stats.level == MemoryLevel.DANGER:
                self._clear_caches()
                optimizations_performed += 1
            
            # 5. Free memory pools
            for pool_name, pool_objects in self.memory_pools.items():
                if len(pool_objects) > 10:  # Keep only last 10 objects
                    excess_count = len(pool_objects) - 10
                    for _ in range(excess_count):
                        if pool_objects:
                            pool_objects.pop()
                    optimizations_performed += 1
            
            # Update statistics
            self.stats['optimizations_performed'] += optimizations_performed
            self.stats['memory_freed_mb'] += memory_freed / 1024 / 1024
            
            if optimizations_performed > 0:
                logger.info(f"Memory optimization completed: {optimizations_performed} actions, "
                          f"{memory_freed/1024/1024:.1f}MB freed")
            
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
    
    def _clear_weak_references(self):
        """Clear weak references to free memory"""
        
        # Get all weak references
        import weakref
        refs = gc.get_referrers(*gc.get_objects())
        
        cleared = 0
        for ref in refs:
            try:
                if hasattr(ref, '__self__') and hasattr(ref, '__func__'):
                    # Method reference - clear it
                    del ref
                    cleared += 1
            except:
                pass
        
        if cleared > 0:
            logger.debug(f"Cleared {cleared} weak references")
    
    def _clear_caches(self):
        """Clear various caches to free memory"""
        
        cleared_caches = 0
        
        # Clear lru_cache
        try:
            import functools
            import inspect
            
            for name, obj in globals().items():
                if hasattr(obj, 'cache_clear'):
                    try:
                        obj.cache_clear()
                        cleared_caches += 1
                    except:
                        pass
        except Exception as e:
            logger.debug(f"Error clearing LRU caches: {e}")
        
        # Clear computation cache if it exists
        if hasattr(self, 'computation_cache'):
            cache_size = len(self.computation_cache)
            self.computation_cache.clear()
            cleared_caches += cache_size
        
        logger.debug(f"Cleared {cleared_caches} cached items")
    
    def _auto_cleanup(self):
        """Automatic cleanup based on memory usage patterns"""
        
        current_time = time.time()
        
        with self.monitor_lock:
            # Remove objects not accessed recently
            for name in list(self.tracked_objects.keys()):
                monitor = self.tracked_objects[name]
                
                # Remove if not accessed for 2x TTL
                if current_time - monitor.last_accessed > monitor.ttl * 2:
                    del self.tracked_objects[name]
                    self.stats['objects_cleaned'] += 1
        
        # Trigger garbage collection if memory usage is high
        stats = self.get_memory_stats()
        if stats.percentage > 60:
            gc.collect()
    
    def create_memory_pool(self, pool_name: str, factory_func: Callable, initial_size: int = 10):
        """Create a memory pool for frequently used objects"""
        
        self.memory_pools[pool_name] = []
        
        # Pre-populate pool
        for _ in range(initial_size):
            try:
                obj = factory_func()
                self.memory_pools[pool_name].append(obj)
            except Exception as e:
                logger.error(f"Error creating pool object: {e}")
        
        logger.info(f"Created memory pool '{pool_name}' with {initial_size} objects")
    
    def get_from_pool(self, pool_name: str, factory_func: Callable = None) -> Optional[Any]:
        """Get object from memory pool"""
        
        if pool_name not in self.memory_pools:
            if factory_func:
                self.create_memory_pool(pool_name, factory_func)
            else:
                return None
        
        pool = self.memory_pools[pool_name]
        
        if pool:
            return pool.pop()
        elif factory_func:
            # Create new object if pool is empty
            try:
                return factory_func()
            except Exception as e:
                logger.error(f"Error creating object from pool {pool_name}: {e}")
        
        return None
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to memory pool"""
        
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = []
        
        pool = self.memory_pools[pool_name]
        
        # Limit pool size to prevent memory bloat
        if len(pool) < 50:
            pool.append(obj)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        
        current_stats = self.get_memory_stats()
        
        # Tracked objects summary
        tracked_summary = {
            'total_tracked': len(self.tracked_objects),
            'total_size_mb': sum(monitor.size_bytes for monitor in self.tracked_objects.values()) / 1024 / 1024,
            'expired_objects': 0
        }
        
        current_time = time.time()
        for monitor in self.tracked_objects.values():
            if current_time - monitor.last_accessed > monitor.ttl:
                tracked_summary['expired_objects'] += 1
        
        # Memory pools summary
        pools_summary = {}
        for pool_name, pool_objects in self.memory_pools.items():
            pools_summary[pool_name] = {
                'size': len(pool_objects),
                'estimated_size_mb': len(pool_objects) * 0.1  # Rough estimate
            }
        
        # System information
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'process_id': os.getpid(),
            'cpu_count': os.cpu_count()
        }
        
        return {
            'current_stats': {
                'current_mb': current_stats.current_mb,
                'peak_mb': current_stats.peak_mb,
                'level': current_stats.level.value,
                'percentage': current_stats.percentage,
                'available_mb': current_stats.available_mb,
                'process_memory_mb': current_stats.process_memory_mb
            },
            'tracked_objects': tracked_summary,
            'memory_pools': pools_summary,
            'statistics': self.stats.copy(),
            'system_info': system_info,
            'gc_settings': {
                'threshold': self.gc_threshold,
                'auto_gc_enabled': self.auto_gc_enabled
            },
            'history_points': len(self.memory_history)
        }
    
    def optimize_now(self) -> Dict[str, Any]:
        """Perform immediate optimization"""
        
        before_stats = self.get_memory_stats()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear expired tracked objects
        with self.monitor_lock:
            current_time = time.time()
            expired_objects = []
            
            for name, monitor in self.tracked_objects.items():
                if current_time - monitor.last_accessed > monitor.ttl:
                    expired_objects.append(name)
            
            for name in expired_objects:
                del self.tracked_objects[name]
        
        # Clear caches
        self._clear_caches()
        
        after_stats = self.get_memory_stats()
        
        return {
            'before': {
                'memory_mb': before_stats.current_mb,
                'percentage': before_stats.percentage,
                'tracked_objects': len(self.tracked_objects)
            },
            'after': {
                'memory_mb': after_stats.current_mb,
                'percentage': after_stats.percentage,
                'tracked_objects': len(self.tracked_objects)
            },
            'improvements': {
                'memory_freed_mb': before_stats.current_mb - after_stats.current_mb,
                'objects_cleared': len(expired_objects),
                'gc_collections': collected
            }
        }
    
    def shutdown(self):
        """Shutdown memory optimization system"""
        
        logger.info("Shutting down memory optimization system...")
        
        # Stop monitoring
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Clear all tracked objects
        with self.monitor_lock:
            self.tracked_objects.clear()
        
        # Clear memory pools
        self.memory_pools.clear()
        
        # Force final garbage collection
        gc.collect()
        
        logger.info("Memory optimization system shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# Decorators for automatic memory management
def memory_tracked(ttl: float = 300.0, priority: int = 0):
    """Decorator to automatically track function result in memory"""
    
    def decorator(func):
        func_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the memory optimizer (assumed to be global)
            optimizer = globals().get('memory_optimizer')
            
            if optimizer:
                # Check if result is cached
                if optimizer.access_tracked_object(func_name):
                    # This is a cached result
                    return None  # Should be handled by cache system
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Track the result
                optimizer.start_tracking(func_name, result, ttl, priority)
                
                return result
            else:
                # No optimizer available, just execute
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def memory_efficient(cache_size: int = 128, ttl: int = 3600):
    """Decorator for memory-efficient function execution"""
    
    def decorator(func):
        # Use lru_cache with memory-aware settings
        cached_func = lru_cache(maxsize=cache_size)(func)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cached_func(*args, **kwargs)
        
        # Add cleanup method
        def clear_cache():
            cached_func.cache_clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()

# Example usage and testing
if __name__ == "__main__":
    
    print("Testing Memory Management System")
    print("=" * 50)
    
    # Test memory tracking
    test_data = [i for i in range(10000)]  # Large list
    memory_optimizer.start_tracking("test_data", test_data, ttl=60, priority=1)
    
    # Test memory pools
    def create_string():
        return "a" * 1000  # 1KB string
    
    memory_optimizer.create_memory_pool("strings", create_string, 5)
    
    # Get from pool
    string_obj = memory_optimizer.get_from_pool("strings", create_string)
    print(f"Got string from pool: {len(string_obj)} characters")
    
    # Return to pool
    memory_optimizer.return_to_pool("strings", string_obj)
    
    # Get memory stats
    stats = memory_optimizer.get_memory_stats()
    print(f"Memory Stats:")
    print(f"  Current: {stats.current_mb:.1f}MB")
    print(f"  Peak: {stats.peak_mb:.1f}MB")
    print(f"  Level: {stats.level.value}")
    print(f"  System Usage: {stats.percentage:.1f}%")
    
    # Test optimization
    print("\nTesting Optimization:")
    before_stats = memory_optimizer.get_memory_stats()
    optimization_result = memory_optimizer.optimize_now()
    
    print(f"Memory freed: {optimization_result['improvements']['memory_freed_mb']:.2f}MB")
    print(f"Objects cleared: {optimization_result['improvements']['objects_cleared']}")
    print(f"GC collections: {optimization_result['improvements']['gc_collections']}")
    
    # Get comprehensive report
    report = memory_optimizer.get_optimization_report()
    print(f"\nOptimization Report:")
    print(f"Tracked objects: {report['tracked_objects']['total_tracked']}")
    print(f"Memory pools: {len(report['memory_pools'])}")
    print(f"Total optimizations: {report['statistics']['optimizations_performed']}")
    
    # Shutdown
    memory_optimizer.shutdown()