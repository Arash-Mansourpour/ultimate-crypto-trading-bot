"""
API Performance Optimization System for Crypto Bot
Advanced rate limiting, connection pooling, caching, and performance monitoring
"""

import asyncio
import aiohttp
import time
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from urllib.parse import urlparse
import hashlib
import gzip
import ssl
import certifi
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """Supported API providers"""
    BINANCE = "binance"
    COINGECKO = "coingecko"
    CRYPTOCOMPARE = "cryptocompare"
    GROQ = "groq"
    GOOGLE = "google"
    BLOCKCHAIN_INFO = "blockchain_info"

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    provider: APIProvider
    base_url: str
    rate_limit: int  # Requests per time window
    rate_window: float  # Time window in seconds
    timeout: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Performance tracking
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: float = 0.0

@dataclass
class APIRequest:
    """API request representation"""
    endpoint: APIEndpoint
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = None
    data: Any = None
    timeout: float = None
    priority: int = 0  # Lower number = higher priority
    
    # Response tracking
    request_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    retry_count: int = 0

@dataclass
class APIResponse:
    """API response representation"""
    request_id: str
    status_code: int
    data: Any
    headers: Dict[str, str]
    response_time: float
    cached: bool = False
    error: str = None

class ConnectionPool:
    """Advanced connection pool for HTTP requests"""
    
    def __init__(self, 
                 max_connections: int = 100,
                 max_keepalive_connections: int = 20,
                 keepalive_expiry: float = 5.0,
                 ssl_context: ssl.SSLContext = None):
        
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        
        # SSL context
        if ssl_context is None:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context = ssl_context
        
        # Connection tracking
        self.active_connections = 0
        self.keepalive_connections = {}
        self.connection_stats = defaultdict(int)
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(f"Connection pool initialized (max: {max_connections})")
    
    @asynccontextmanager
    async def get_connection(self, session: aiohttp.ClientSession):
        """Get a connection from the pool"""
        
        connection = None
        try:
            # Try to get a keepalive connection first
            with self.lock:
                if self.keepalive_connections:
                    # Get the oldest connection
                    oldest_key = min(self.keepalive_connections.keys())
                    connection = self.keepalive_connections.pop(oldest_key)
            
            if connection:
                # Test if connection is still valid
                try:
                    # Simple test - send a ping
                    async with connection.get('http://localhost') as resp:
                        pass
                    # Connection is good
                except:
                    # Connection is dead, discard it
                    connection = None
            
            if not connection:
                # Create new connection
                connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=self.max_keepalive_connections,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=self.keepalive_expiry,
                    enable_cleanup_closed=True
                )
                
                session._connector = connector
                self.connection_stats['new_connections'] += 1
            
            yield connection
            
        finally:
            if connection:
                # Return connection to keepalive pool
                with self.lock:
                    if len(self.keepalive_connections) < self.max_keepalive_connections:
                        self.keepalive_connections[time.time()] = connection
                    else:
                        connection.close()

class RateLimiter:
    """Advanced rate limiter with multiple strategies"""
    
    def __init__(self, 
                 strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
                 default_rate: int = 10,
                 default_window: float = 60.0):
        
        self.strategy = strategy
        self.default_rate = default_rate
        self.default_window = default_window
        
        # Rate limiting data
        self.rate_limits = {}  # endpoint_name -> (rate, window)
        self.request_times = defaultdict(deque)  # endpoint_name -> list of request times
        
        # Token bucket for advanced rate limiting
        self.token_buckets = {}  # endpoint_name -> {'tokens': float, 'last_update': float}
        
        # Statistics
        self.blocked_requests = defaultdict(int)
        self.total_requests = defaultdict(int)
        
        # Threading
        self.lock = threading.Lock()
    
    def configure_endpoint(self, endpoint_name: str, rate: int, window: float):
        """Configure rate limit for specific endpoint"""
        
        with self.lock:
            self.rate_limits[endpoint_name] = (rate, window)
            
            # Initialize token bucket if using token bucket strategy
            if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
                self.token_buckets[endpoint_name] = {
                    'tokens': float(rate),
                    'last_update': time.time()
                }
        
        logger.info(f"Configured rate limit for {endpoint_name}: {rate} req/{window}s")
    
    async def acquire(self, endpoint_name: str, rate: int = None, window: float = None) -> bool:
        """Acquire permission to make a request"""
        
        rate = rate or self.default_rate
        window = window or self.default_window
        
        with self.lock:
            self.total_requests[endpoint_name] += 1
        
        # Check rate limit based on strategy
        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(endpoint_name, rate, window)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(endpoint_name, rate, window)
        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(endpoint_name, rate)
        else:
            return True  # No rate limiting
    
    async def _check_sliding_window(self, endpoint_name: str, rate: int, window: float) -> bool:
        """Check rate limit using sliding window algorithm"""
        
        current_time = time.time()
        cutoff_time = current_time - window
        
        with self.lock:
            request_times = self.request_times[endpoint_name]
            
            # Remove old requests outside the window
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
            
            # Check if we can make another request
            if len(request_times) < rate:
                request_times.append(current_time)
                return True
            else:
                self.blocked_requests[endpoint_name] += 1
                return False
    
    async def _check_fixed_window(self, endpoint_name: str, rate: int, window: float) -> bool:
        """Check rate limit using fixed window algorithm"""
        
        current_time = time.time()
        window_start = int(current_time / window) * window
        
        with self.lock:
            request_times = self.request_times[endpoint_name]
            
            # Remove old windows
            keys_to_remove = []
            for timestamp in request_times:
                if int(timestamp / window) * window < window_start:
                    keys_to_remove.append(timestamp)
            
            for key in keys_to_remove:
                if key in request_times:
                    request_times.remove(key)
            
            # Check if we can make another request in current window
            window_requests = sum(1 for t in request_times if int(t / window) * window == window_start)
            
            if window_requests < rate:
                request_times.append(current_time)
                return True
            else:
                self.blocked_requests[endpoint_name] += 1
                return False
    
    async def _check_token_bucket(self, endpoint_name: str, rate: int) -> bool:
        """Check rate limit using token bucket algorithm"""
        
        current_time = time.time()
        
        with self.lock:
            if endpoint_name not in self.token_buckets:
                self.token_buckets[endpoint_name] = {
                    'tokens': float(rate),
                    'last_update': current_time
                }
            
            bucket = self.token_buckets[endpoint_name]
            
            # Add tokens based on time passed
            time_passed = current_time - bucket['last_update']
            tokens_to_add = time_passed * (rate / 60.0)  # tokens per second
            
            bucket['tokens'] = min(rate, bucket['tokens'] + tokens_to_add)
            bucket['last_update'] = current_time
            
            # Check if we have enough tokens
            if bucket['tokens'] >= 1.0:
                bucket['tokens'] -= 1.0
                return True
            else:
                self.blocked_requests[endpoint_name] += 1
                return False
    
    def get_stats(self, endpoint_name: str = None) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        
        if endpoint_name:
            return {
                'total_requests': self.total_requests[endpoint_name],
                'blocked_requests': self.blocked_requests[endpoint_name],
                'blocked_rate': (self.blocked_requests[endpoint_name] / 
                               max(self.total_requests[endpoint_name], 1)) * 100
            }
        
        stats = {}
        for endpoint in self.total_requests.keys():
            stats[endpoint] = self.get_stats(endpoint)
        
        return stats

class APIResponseCache:
    """Advanced response caching with compression and TTL"""
    
    def __init__(self, 
                 max_size_mb: int = 100,
                 default_ttl: float = 300.0,
                 compression_enabled: bool = True):
        
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.compression_enabled = compression_enabled
        
        # Cache storage
        self.cache = {}  # cache_key -> cache_entry
        self.cache_order = deque()  # For LRU eviction
        self.current_size_mb = 0.0
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0,
            'avg_response_time_saved': 0.0
        }
        
        # Threading
        self.lock = threading.Lock()
        
        logger.info(f"Response cache initialized (max: {max_size_mb}MB)")
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request"""
        
        # Create hash from URL, method, and sorted params
        key_data = {
            'url': request.url,
            'method': request.method,
            'params': request.params or {},
            'headers': {k: v for k, v in request.headers.items() if k.lower() in ['authorization', 'content-type']}
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response"""
        
        cache_key = self._generate_cache_key(request)
        
        with self.lock:
            self.stats['total_requests'] += 1
            
            if cache_key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            entry = self.cache[cache_key]
            
            # Check if entry is expired
            if time.time() > entry['expires_at']:
                del self.cache[cache_key]
                self.cache_order.remove(cache_key)
                self.stats['misses'] += 1
                return None
            
            # Move to end of LRU order
            self.cache_order.remove(cache_key)
            self.cache_order.append(cache_key)
            
            self.stats['hits'] += 1
            
            # Return cached response
            return APIResponse(
                request_id=cache_key,
                status_code=entry['status_code'],
                data=entry['data'],
                headers=entry['headers'],
                response_time=0.0,  # No actual response time for cached
                cached=True
            )
    
    def put(self, request: APIRequest, response: APIResponse, ttl: float = None):
        """Cache a response"""
        
        ttl = ttl or self.default_ttl
        cache_key = self._generate_cache_key(request)
        
        # Prepare data for storage
        data_to_store = response.data
        
        if self.compression_enabled and isinstance(data_to_store, (str, bytes)):
            try:
                # Compress large responses
                if isinstance(data_to_store, str):
                    data_to_store = data_to_store.encode('utf-8')
                
                if len(data_to_store) > 1024:  # Compress if larger than 1KB
                    data_to_store = gzip.compress(data_to_store)
                    compressed = True
                else:
                    compressed = False
            except Exception:
                compressed = False
        else:
            compressed = False
        
        # Calculate size
        import sys
        size_bytes = sys.getsizeof(data_to_store)
        
        with self.lock:
            # Evict entries if cache is full
            while (self.current_size_mb + size_bytes / 1024 / 1024 > self.max_size_mb and 
                   self.cache):
                self._evict_oldest_entry()
            
            # Store the entry
            expires_at = time.time() + ttl
            
            self.cache[cache_key] = {
                'data': data_to_store,
                'headers': response.headers,
                'status_code': response.status_code,
                'expires_at': expires_at,
                'compressed': compressed,
                'created_at': time.time()
            }
            
            self.cache_order.append(cache_key)
            self.current_size_mb += size_bytes / 1024 / 1024
            
            # Update stats
            self.stats['avg_response_time_saved'] = (
                (self.stats['avg_response_time_saved'] * self.stats['hits'] + response.response_time) /
                (self.stats['hits'] + 1)
            )
    
    def _evict_oldest_entry(self):
        """Evict the oldest entry from cache"""
        
        if not self.cache_order:
            return
        
        oldest_key = self.cache_order.popleft()
        entry = self.cache[oldest_key]
        
        # Calculate size
        import sys
        size_bytes = sys.getsizeof(entry['data'])
        
        del self.cache[oldest_key]
        self.current_size_mb -= size_bytes / 1024 / 1024
        self.stats['evictions'] += 1
    
    def clear_expired(self):
        """Remove expired entries from cache"""
        
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for cache_key, entry in self.cache.items():
                if current_time > entry['expires_at']:
                    expired_keys.append(cache_key)
            
            for cache_key in expired_keys:
                del self.cache[cache_key]
                if cache_key in self.cache_order:
                    self.cache_order.remove(cache_key)
        
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        with self.lock:
            hit_rate = (self.stats['hits'] / max(self.stats['total_requests'], 1)) * 100
            
            return {
                **self.stats.copy(),
                'hit_rate': hit_rate,
                'current_size_mb': self.current_size_mb,
                'max_size_mb': self.max_size_mb,
                'entries_count': len(self.cache),
                'utilization': (self.current_size_mb / self.max_size_mb) * 100
            }

class OptimizedAPIManager:
    """Advanced API manager with performance optimizations"""
    
    def __init__(self):
        self.endpoints = {}  # endpoint_name -> APIEndpoint
        self.connection_pools = {}  # provider -> ConnectionPool
        self.rate_limiter = RateLimiter()
        self.response_cache = APIResponseCache()
        
        # Session management
        self.sessions = {}  # provider -> aiohttp.ClientSession
        self.session_lock = threading.Lock()
        
        # Performance monitoring
        self.request_history = deque(maxlen=1000)
        self.provider_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'last_request': 0.0
        })
        
        # Circuit breakers
        self.circuit_breakers = {}  # provider -> {'failures': int, 'last_failure': float, 'state': str}
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default endpoints
        self._initialize_default_endpoints()
    
    def _initialize_default_endpoints(self):
        """Initialize default API endpoints"""
        
        endpoints = [
            APIEndpoint(
                name="binance_price",
                provider=APIProvider.BINANCE,
                base_url="https://api.binance.com/api/v3",
                rate_limit=10,  # 10 requests per minute
                rate_window=60.0,
                timeout=10.0
            ),
            APIEndpoint(
                name="binance_ohlcv",
                provider=APIProvider.BINANCE,
                base_url="https://api.binance.com/api/v3",
                rate_limit=1200,  # 1200 requests per minute
                rate_window=60.0,
                timeout=15.0
            ),
            APIEndpoint(
                name="coingecko_price",
                provider=APIProvider.COINGECKO,
                base_url="https://api.coingecko.com/api/v3",
                rate_limit=10,  # 10 requests per minute
                rate_window=60.0,
                timeout=10.0
            ),
            APIEndpoint(
                name="groq_chat",
                provider=APIProvider.GROQ,
                base_url="https://api.groq.com/openai/v1",
                rate_limit=30,  # 30 requests per minute
                rate_window=60.0,
                timeout=30.0
            ),
            APIEndpoint(
                name="google_search",
                provider=APIProvider.GOOGLE,
                base_url="https://customsearch.googleapis.com",
                rate_limit=100,  # 100 requests per day
                rate_window=86400.0,  # 24 hours
                timeout=10.0
            )
        ]
        
        for endpoint in endpoints:
            self.register_endpoint(endpoint)
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """Register an API endpoint"""
        
        self.endpoints[endpoint.name] = endpoint
        
        # Configure rate limiter
        self.rate_limiter.configure_endpoint(
            endpoint.name, 
            endpoint.rate_limit, 
            endpoint.rate_window
        )
        
        # Initialize circuit breaker for provider
        provider_key = endpoint.provider.value
        self.circuit_breakers[provider_key] = {
            'failures': 0,
            'last_failure': 0.0,
            'state': 'closed'
        }
        
        logger.info(f"Registered endpoint: {endpoint.name} ({endpoint.provider.value})")
    
    async def get_session(self, provider: APIProvider) -> aiohttp.ClientSession:
        """Get or create a session for the provider"""
        
        provider_key = provider.value
        
        with self.session_lock:
            if provider_key not in self.sessions:
                # Create new session
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=10,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={'User-Agent': 'CryptoBot/2.0'}
                )
                
                self.sessions[provider_key] = session
        
        return self.sessions[provider_key]
    
    async def make_request(self, 
                          endpoint_name: str,
                          path: str,
                          method: str = "GET",
                          params: Dict[str, Any] = None,
                          data: Any = None,
                          headers: Dict[str, str] = None,
                          timeout: float = None,
                          cache_ttl: float = None) -> APIResponse:
        
        """Make an optimized API request"""
        
        if endpoint_name not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint_name} not registered")
        
        endpoint = self.endpoints[endpoint_name]
        provider = endpoint.provider
        
        # Check circuit breaker
        if not self._check_circuit_breaker(provider.value):
            raise Exception(f"Circuit breaker OPEN for {provider.value}")
        
        # Check rate limit
        rate_limited = await self.rate_limiter.acquire(endpoint_name, endpoint.rate_limit, endpoint.rate_window)
        if not rate_limited:
            raise Exception(f"Rate limit exceeded for {endpoint_name}")
        
        # Check cache for GET requests
        if method.upper() == "GET":
            request = APIRequest(
                endpoint=endpoint,
                url=f"{endpoint.base_url}/{path.lstrip('/')}",
                method=method,
                headers=headers or {},
                params=params,
                timeout=timeout or endpoint.timeout
            )
            
            cached_response = self.response_cache.get(request)
            if cached_response:
                logger.debug(f"Cache hit for {endpoint_name}")
                return cached_response
        
        # Make the actual request
        start_time = time.time()
        request_id = f"{endpoint_name}_{int(start_time * 1000)}"
        
        try:
            session = await self.get_session(provider)
            
            url = f"{endpoint.base_url}/{path.lstrip('/')}"
            request_kwargs = {
                'url': url,
                'method': method,
                'headers': headers or {}
            }
            
            if params:
                request_kwargs['params'] = params
            
            if data and method.upper() in ['POST', 'PUT', 'PATCH']:
                if isinstance(data, dict):
                    request_kwargs['json'] = data
                else:
                    request_kwargs['data'] = data
            
            timeout_obj = aiohttp.ClientTimeout(total=timeout or endpoint.timeout)
            request_kwargs['timeout'] = timeout_obj
            
            async with session.request(**request_kwargs) as response:
                response_data = await response.text()
                response_headers = dict(response.headers)
                
                # Try to parse JSON
                try:
                    if response_headers.get('content-type', '').startswith('application/json'):
                        response_data = json.loads(response_data)
                except:
                    pass  # Keep as text if not valid JSON
                
                response_time = time.time() - start_time
                
                # Create response object
                api_response = APIResponse(
                    request_id=request_id,
                    status_code=response.status_code,
                    data=response_data,
                    headers=response_headers,
                    response_time=response_time
                )
                
                # Cache successful GET responses
                if (method.upper() == 'GET' and 
                    response.status_code == 200 and 
                    200 <= len(str(response_data)) < 1024 * 1024):  # Max 1MB
                    cache_request = APIRequest(
                        endpoint=endpoint,
                        url=url,
                        method=method,
                        headers=headers or {},
                        params=params,
                        timeout=timeout or endpoint.timeout
                    )
                    self.response_cache.put(cache_request, api_response, cache_ttl)
                
                # Update statistics
                self._update_success_stats(provider.value, response_time)
                
                logger.debug(f"API request successful: {endpoint_name} ({response.status_code}, {response_time:.2f}s)")
                
                return api_response
                
        except Exception as e:
            response_time = time.time() - start_time
            self._update_failure_stats(provider.value, str(e))
            
            logger.error(f"API request failed: {endpoint_name} - {e}")
            raise
    
    def _check_circuit_breaker(self, provider_key: str) -> bool:
        """Check if circuit breaker allows requests"""
        
        if provider_key not in self.circuit_breakers:
            return True
        
        cb = self.circuit_breakers[provider_key]
        current_time = time.time()
        
        if cb['state'] == 'open':
            # Check if it's time to try again
            if current_time - cb['last_failure'] > 60:  # Try after 1 minute
                cb['state'] = 'half-open'
                logger.info(f"Circuit breaker {provider_key} moved to HALF-OPEN")
                return True
            return False
        
        return True
    
    def _update_success_stats(self, provider_key: str, response_time: float):
        """Update success statistics"""
        
        stats = self.provider_stats[provider_key]
        stats['total_requests'] += 1
        stats['successful_requests'] += 1
        stats['last_request'] = time.time()
        
        # Update average response time
        if stats['avg_response_time'] == 0:
            stats['avg_response_time'] = response_time
        else:
            stats['avg_response_time'] = (stats['avg_response_time'] * 0.9) + (response_time * 0.1)
        
        # Reset circuit breaker on success
        if provider_key in self.circuit_breakers:
            cb = self.circuit_breakers[provider_key]
            if cb['state'] in ['half-open', 'open']:
                cb['state'] = 'closed'
                cb['failures'] = 0
                logger.info(f"Circuit breaker {provider_key} closed")
    
    def _update_failure_stats(self, provider_key: str, error: str):
        """Update failure statistics"""
        
        stats = self.provider_stats[provider_key]
        stats['total_requests'] += 1
        stats['failed_requests'] += 1
        stats['last_request'] = time.time()
        
        # Update circuit breaker
        if provider_key in self.circuit_breakers:
            cb = self.circuit_breakers[provider_key]
            cb['failures'] += 1
            cb['last_failure'] = time.time()
            
            if cb['failures'] >= 5:  # Threshold for opening circuit
                cb['state'] = 'open'
                logger.warning(f"Circuit breaker {provider_key} OPENED after {cb['failures']} failures")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        # Request history analysis
        recent_requests = [req for req in self.request_history 
                          if time.time() - req['timestamp'] < 3600]  # Last hour
        
        if recent_requests:
            avg_response_time = sum(req['response_time'] for req in recent_requests) / len(recent_requests)
            success_rate = (sum(1 for req in recent_requests if req['success']) / len(recent_requests)) * 100
        else:
            avg_response_time = 0
            success_rate = 0
        
        return {
            'overall_stats': {
                'avg_response_time': avg_response_time,
                'success_rate': success_rate,
                'total_requests_1h': len(recent_requests)
            },
            'provider_stats': dict(self.provider_stats),
            'rate_limiter_stats': self.rate_limiter.get_stats(),
            'cache_stats': self.response_cache.get_stats(),
            'circuit_breakers': {k: v for k, v in self.circuit_breakers.items()},
            'registered_endpoints': len(self.endpoints)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all endpoints"""
        
        health_status = {}
        current_time = time.time()
        
        for endpoint_name, endpoint in self.endpoints.items():
            provider_key = endpoint.provider.value
            provider_stats = self.provider_stats[provider_key]
            
            # Determine health based on recent performance
            last_request_age = current_time - provider_stats['last_request']
            if provider_stats['total_requests'] == 0:
                status = 'unknown'
            elif provider_stats['failed_requests'] / provider_stats['total_requests'] > 0.1:  # > 10% failure rate
                status = 'degraded'
            elif last_request_age > 300:  # 5 minutes since last request
                status = 'stale'
            else:
                status = 'healthy'
            
            health_status[endpoint_name] = {
                'status': status,
                'total_requests': provider_stats['total_requests'],
                'success_rate': (provider_stats['successful_requests'] / 
                               max(provider_stats['total_requests'], 1)) * 100,
                'avg_response_time': provider_stats['avg_response_time'],
                'last_request_age': last_request_age,
                'circuit_breaker_state': self.circuit_breakers[provider_key]['state']
            }
        
        return health_status
    
    async def shutdown(self):
        """Shutdown the API manager"""
        
        logger.info("Shutting down API manager...")
        
        # Close all sessions
        with self.session_lock:
            for session in self.sessions.values():
                await session.close()
            self.sessions.clear()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear response cache
        self.response_cache = None
        
        logger.info("API manager shutdown complete")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

# Example usage and testing
if __name__ == "__main__":
    
    async def test_api_manager():
        """Test the API manager"""
        
        print("Testing Optimized API Manager")
        print("=" * 50)
        
        async with OptimizedAPIManager() as api_manager:
            
            # Test health check
            print("\n1. Health Check:")
            health = await api_manager.health_check()
            for endpoint, status in health.items():
                print(f"   {endpoint}: {status['status']}")
            
            # Test making a request
            print("\n2. Testing API Request:")
            try:
                # Test Binance price endpoint (cached)
                response = await api_manager.make_request(
                    endpoint_name="binance_price",
                    path="/ticker/price",
                    params={"symbol": "BTCUSDT"}
                )
                
                print(f"   Status: {response.status_code}")
                print(f"   Response time: {response.response_time:.2f}s")
                print(f"   Data: {response.data}")
                
            except Exception as e:
                print(f"   Error: {e}")
            
            # Test performance stats
            print("\n3. Performance Stats:")
            stats = api_manager.get_performance_stats()
            print(f"   Average response time: {stats['overall_stats']['avg_response_time']:.2f}s")
            print(f"   Success rate: {stats['overall_stats']['success_rate']:.1f}%")
            print(f"   Cache hit rate: {stats['cache_stats']['hit_rate']:.1f}%")
            print(f"   Requests in last hour: {stats['overall_stats']['total_requests_1h']}")
    
    # Run the test
    asyncio.run(test_api_manager())