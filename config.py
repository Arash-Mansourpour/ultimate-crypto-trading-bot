"""
Configuration file for Arshava V2.0 Crypto Trading Bot
Update these values with your actual API keys
"""

import os
from typing import List

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7946390053:AAFu9Ac-hamijaCDjVpESlLfQYuZ86HJ0PY')
if TELEGRAM_TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN_HERE':
    raise ValueError("❌ Please set your Telegram Bot Token! See instructions below.")

# API Keys Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
CMC_API_KEY = os.getenv('CMC_API_KEY', '6f754f9e-af16-4017-8993-6ae8cf67c1b1')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyA8NV_u2tlPSRY8-jFanhW1AFby-wlA7Qs')
SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID', 'd504953928a9f428f')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_news_api_key_here')

# Supported Cryptocurrencies
SUPPORTED_SYMBOLS = [
    'BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK', 
    'XRP', 'DOGE', 'SHIB', 'UNI', 'LTC', 'BCH', 'ETC', 'ATOM', 
    'FIL', 'TRX', 'VET', 'XLM', 'BNB', 'NEAR', 'APT', 'HBAR',
    'ALGO', 'FLOW', 'ICP', 'SAND', 'MANA', 'AXS'
]

# Database Configuration
DATABASE_PATH = 'arshava_v2.db'

# Rate Limiting Configuration
RATE_LIMITS = {
    'binance': 15,      # requests per minute
    'coingecko': 10,    # requests per minute  
    'cryptocompare': 5, # requests per minute
    'fear_greed': 2,    # requests per minute
    'news_search': 5    # requests per minute
}

# Cache Configuration (in seconds)
CACHE_TTL = {
    'price_data': 60,
    'ohlcv_data': 300,
    'indicators': 180,
    'fear_greed': 600,
    'news_data': 300,
    'ai_analysis': 120
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FILE = 'arshava_v2.log'

# Bot Settings
MAX_MESSAGE_LENGTH = 4000
CHUNK_SIZE = 3200
DEFAULT_TIMEFRAME = '1d'
DEFAULT_LIMIT = 200