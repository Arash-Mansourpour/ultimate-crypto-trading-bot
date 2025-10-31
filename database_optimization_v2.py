"""
Enhanced Database Optimization for Crypto Bot
Advanced indexing, query optimization, and performance improvements
"""

import sqlite3
import threading
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Advanced database configuration"""
    db_path: str = "enhanced_crypto_bot.db"
    pool_size: int = 10
    max_retries: int = 3
    timeout: float = 30.0
    wal_mode: bool = True
    cache_size: int = -64000  # 64MB cache
    temp_store: str = "memory"  # Temporary tables in memory
    mmap_size: int = 256 * 1024 * 1024  # 256MB mmap
    page_size: int = 4096
    foreign_keys: bool = True

class OptimizedConnectionPool:
    """Advanced connection pool with WAL mode and optimization"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = []
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with optimizations"""
        conn = self._create_connection()
        
        # Enable WAL mode for better concurrency
        if self.config.wal_mode:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Set cache size
        conn.execute(f"PRAGMA cache_size={self.config.cache_size};")
        
        # Set temp store to memory
        conn.execute(f"PRAGMA temp_store={self.config.temp_store};")
        
        # Set mmap size
        conn.execute(f"PRAGMA mmap_size={self.config.mmap_size};")
        
        # Set page size
        conn.execute(f"PRAGMA page_size={self.config.page_size};")
        
        # Enable foreign keys
        conn.execute(f"PRAGMA foreign_keys={self.config.foreign_keys};")
        
        # Create all indexes
        self._create_optimized_indexes(conn)
        
        # Enable query planner optimizations
        conn.execute("PRAGMA optimize;")
        conn.execute("PRAGMA compile_options;")
        
        conn.commit()
        conn.close()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized database connection"""
        conn = sqlite3.connect(
            self.config.db_path,
            timeout=self.config.timeout,
            check_same_thread=False,
            isolation_level=None  # Enable autocommit mode
        )
        
        # Enable row factory
        conn.row_factory = sqlite3.Row
        
        # Enable extended result codes
        conn.execute("PRAGMA extended_result_codes=ON;")
        
        return conn
    
    def _create_optimized_indexes(self, conn: sqlite3.Connection):
        """Create comprehensive optimized indexes"""
        
        # User profiles indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id 
            ON user_profiles(user_id);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_account_level 
            ON user_profiles(account_level);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_experience 
            ON user_profiles(experience_level);
        """)
        
        # Trading signals indexes (most important)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_user_id_created 
            ON trading_signals(user_id, created_at DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_created 
            ON trading_signals(symbol, created_at DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_signal_confidence 
            ON trading_signals(signal, confidence DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_signal 
            ON trading_signals(symbol, signal, created_at DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_signals_created_date 
            ON trading_signals(created_at);
        """)
        
        # Signal feedback indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_feedback_signal_id 
            ON signal_feedback(signal_id);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_feedback_user_id 
            ON signal_feedback(user_id);
        """)
        
        # Price alerts indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_alerts_user_id_active 
            ON price_alerts(user_id, is_active);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol 
            ON price_alerts(symbol);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_alerts_target_price 
            ON price_alerts(target_price);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_alerts_condition_type 
            ON price_alerts(condition_type);
        """)
        
        # User portfolio indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_portfolio_user_id 
            ON user_portfolio(user_id);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_portfolio_symbol 
            ON user_portfolio(symbol);
        """)
        
        # Conversation history indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_history_user_id_timestamp 
            ON conversation_history(user_id, timestamp DESC);
        """)
        
        # Market data indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
            ON market_data(symbol, timestamp DESC);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
            ON market_data(timestamp);
        """)
        
        # Composite indexes for complex queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_user_symbol_date 
            ON trading_signals(user_id, symbol, created_at);
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_signal_date 
            ON trading_signals(symbol, signal, created_at);
        """)
        
        # Partial indexes for active records
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_active_price_alerts 
            ON price_alerts(user_id, symbol) 
            WHERE is_active = 1;
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recent_signals 
            ON trading_signals(user_id, confidence DESC) 
            WHERE created_at > datetime('now', '-30 days');
        """)
        
        # Performance monitoring indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_analysis_requests_user_time 
            ON analysis_requests(user_id, timestamp DESC);
        """)
        
        # Full-text search indexes (if needed)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS user_feedback_fts 
            USING fts5(feedback_text, content='signal_feedback', content_rowid='id');
        """)
        
        logger.info("✅ Database indexes created successfully")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get connection from pool"""
        with self.lock:
            if self.pool:
                return self.pool.pop()
            return self._create_connection()
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self.lock:
            if len(self.pool) < self.config.pool_size:
                self.pool.append(conn)
            else:
                conn.close()
    
    def close_all(self):
        """Close all connections in pool"""
        with self.lock:
            for conn in self.pool:
                conn.close()
            self.pool.clear()

class OptimizedDatabaseManager:
    """Enhanced database manager with advanced optimizations"""
    
    def __init__(self, db_path: str = "enhanced_crypto_bot.db"):
        self.config = DatabaseConfig(db_path=db_path)
        self.pool = OptimizedConnectionPool(self.config)
        self.lock = threading.Lock()
        
        # Performance metrics
        self.query_count = 0
        self.total_query_time = 0.0
        self.slow_queries = []
        
        # Query optimization settings
        self.enable_query_plan = False
        self.slow_query_threshold = 1.0  # seconds
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema with optimizations"""
        conn = self.pool.get_connection()
        
        try:
            # Create user profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    experience_level TEXT DEFAULT 'beginner',
                    risk_tolerance TEXT DEFAULT 'medium',
                    account_size REAL DEFAULT 10000,
                    account_level TEXT DEFAULT 'free',
                    preferred_language TEXT DEFAULT 'en',
                    timezone TEXT DEFAULT 'UTC',
                    notifications_enabled INTEGER DEFAULT 1,
                    risk_per_trade REAL DEFAULT 0.02,
                    max_concurrent_signals INTEGER DEFAULT 5,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create trading signals table with optimizations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    position_size REAL,
                    risk_reward_ratio REAL,
                    timeframe TEXT DEFAULT '1d',
                    reasons TEXT, -- JSON array
                    technical_score REAL,
                    sentiment_score REAL,
                    volume_score REAL,
                    social_sentiment REAL,
                    onchain_score REAL,
                    backtest_winrate REAL,
                    ichimoku_signal TEXT,
                    fibonacci_levels TEXT, -- JSON
                    elliott_wave TEXT,
                    multi_timeframe_alignment TEXT,
                    ml_prediction TEXT,
                    smc_analysis TEXT, -- JSON
                    vsa_signal TEXT,
                    wyckoff_analysis TEXT,
                    order_blocks TEXT, -- JSON
                    liquidity_zones TEXT, -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create signal feedback table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    feedback_text TEXT,
                    profitability REAL,
                    actual_outcome TEXT,
                    confidence_rating INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES trading_signals (id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create price alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    target_price REAL NOT NULL,
                    condition_type TEXT NOT NULL, -- 'above' or 'below'
                    is_active INTEGER DEFAULT 1,
                    triggered_at DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create user portfolio table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    value REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create conversation history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    response TEXT,
                    message_type TEXT DEFAULT 'text',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create market data table for caching
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume REAL,
                    market_cap REAL,
                    price_change_24h REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT DEFAULT 'binance'
                )
            """)
            
            # Create analysis requests table for performance tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    request_type TEXT DEFAULT 'analysis',
                    processing_time REAL,
                    success INTEGER DEFAULT 1,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create user settings table for advanced preferences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    setting_key TEXT NOT NULL,
                    setting_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, setting_key),
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id) ON DELETE CASCADE
                )
            """)
            
            # Create triggers for automatic updates
            self._create_database_triggers(conn)
            
            # Commit all changes
            conn.commit()
            logger.info("✅ Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            conn.rollback()
            raise
        finally:
            self.pool.return_connection(conn)
    
    def _create_database_triggers(self, conn: sqlite3.Connection):
        """Create database triggers for automatic updates"""
        
        # Trigger to update updated_at timestamp
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_user_profiles_timestamp
            AFTER UPDATE ON user_profiles
            FOR EACH ROW
            BEGIN
                UPDATE user_profiles SET updated_at = CURRENT_TIMESTAMP WHERE user_id = NEW.user_id;
            END
        """)
        
        # Trigger to update portfolio PnL
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS update_portfolio_pnl
            AFTER UPDATE OF current_price ON user_portfolio
            FOR EACH ROW
            BEGIN
                UPDATE user_portfolio 
                SET value = amount * NEW.current_price,
                    pnl = (NEW.current_price - entry_price) * amount,
                    pnl_percent = ((NEW.current_price - entry_price) / entry_price) * 100,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id;
            END
        """)
        
        # Trigger to automatically set expiration for signals
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS set_signal_expiration
            AFTER INSERT ON trading_signals
            FOR EACH ROW
            BEGIN
                UPDATE trading_signals 
                SET expires_at = datetime('now', '+7 days')
                WHERE id = NEW.id AND expires_at IS NULL;
            END
        """)
        
        # Trigger to deactivate alerts when triggered
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS deactivate_triggered_alert
            AFTER UPDATE ON price_alerts
            FOR EACH ROW
            BEGIN
                UPDATE price_alerts 
                SET is_active = 0, triggered_at = CURRENT_TIMESTAMP
                WHERE id = NEW.id AND is_active = 1;
            END
        """)
    
    @property
    def stats(self) -> Dict:
        """Get database performance statistics"""
        return {
            'query_count': self.query_count,
            'total_query_time': round(self.total_query_time, 3),
            'avg_query_time': round(self.total_query_time / max(self.query_count, 1), 3),
            'slow_queries': len(self.slow_queries)
        }
    
    def execute_optimized_query(self, query: str, params: tuple = (), fetch: str = 'all') -> Any:
        """Execute query with performance monitoring and optimization"""
        start_time = time.time()
        conn = self.pool.get_connection()
        
        try:
            # Add query plan if enabled
            if self.enable_query_plan:
                logger.info(f"Query Plan for: {query[:100]}...")
            
            cursor = conn.execute(query, params)
            
            if fetch == 'all':
                result = cursor.fetchall()
            elif fetch == 'one':
                result = cursor.fetchone()
            elif fetch == 'many':
                result = cursor.fetchmany()
            else:
                result = cursor.fetchall()
            
            # Record execution time
            execution_time = time.time() - start_time
            self._record_query_performance(query, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
        finally:
            self.pool.return_connection(conn)
    
    def _record_query_performance(self, query: str, execution_time: float):
        """Record query performance metrics"""
        with self.lock:
            self.query_count += 1
            self.total_query_time += execution_time
            
            # Track slow queries
            if execution_time > self.slow_query_threshold:
                self.slow_queries.append({
                    'query': query[:100] + '...' if len(query) > 100 else query,
                    'time': execution_time,
                    'timestamp': time.time()
                })
                
                # Keep only last 100 slow queries
                if len(self.slow_queries) > 100:
                    self.slow_queries = self.slow_queries[-100:]
                
                logger.warning(f"Slow query detected ({execution_time:.3f}s): {query[:100]}...")

# Usage example
if __name__ == "__main__":
    # Initialize optimized database
    db_manager = OptimizedDatabaseManager("enhanced_crypto_bot_v2.db")
    
    # Show performance stats
    print("Database Performance Stats:")
    print(f"Total queries: {db_manager.stats['query_count']}")
    print(f"Total time: {db_manager.stats['total_query_time']}s")
    print(f"Average time: {db_manager.stats['avg_query_time']}s")