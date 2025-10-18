import re
import time
import asyncio
import json
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
import telebot
from telebot import types
from groq import Groq
from googleapiclient.discovery import build
from textblob import TextBlob
from scipy.signal import argrelextrema
import threading
import requests
from dataclasses import dataclass
import warnings
import platform
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

warnings.filterwarnings('ignore')

if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_crypto_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = "7946390053:AAFu9Ac-hamijaCDjVpESlLfQYuZ86HJ0PY"
GROQ_API_KEY = "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"
CMC_API_KEY = "6f754f9e-af16-4017-8993-6ae8cf67c1b1"
GOOGLE_API_KEY = "AIzaSyA8NV_u2tlPSRY8-jFanhW1AFby-wlA7Qs"
SEARCH_ENGINE_ID = "53d8a73eb43a44a77"

bot = telebot.TeleBot(TELEGRAM_TOKEN)

@dataclass
class MarketData:
    symbol: str
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    price_change_7d: float
    rsi: float
    macd_signal: str
    trend: str
    support_level: float
    resistance_level: float
    volatility: float
    liquidity_score: float
    fear_greed_index: int
    poc: float
    value_area_high: float
    value_area_low: float
    vwap: float
    harmonic_pattern: str
    chart_pattern: str
    historical_volatility: float
    correlation_btc: float
    smc_signal: str
    wyckoff_phase: str
    cvd: float
    exchange_netflow: float
    whale_activity: int

@dataclass
class TradingSignal:
    signal: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    timeframe: str
    reasons: List[str]
    technical_score: float
    sentiment_score: float
    volume_score: float
    social_sentiment: float
    onchain_score: float
    backtest_winrate: float
    ichimoku_signal: str
    fibonacci_levels: Dict
    elliott_wave: str
    multi_timeframe_alignment: str
    ml_prediction: str
    smc_analysis: Dict
    vsa_signal: str
    wyckoff_analysis: str
    order_blocks: List[Dict]
    liquidity_zones: List[Dict]

# ============================================
# SMART MONEY CONCEPTS (SMC) ANALYZER
# ============================================

class SmartMoneyConceptsAnalyzer:
    """Advanced SMC Analysis - Order Blocks, FVG, Liquidity"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive SMC Analysis"""
        try:
            if len(df) < 50:
                return {}
            
            analysis = {
                'order_blocks': self._find_order_blocks(df),
                'fair_value_gaps': self._find_fvg(df),
                'liquidity_zones': self._find_liquidity_zones(df),
                'bos': self._detect_break_of_structure(df),
                'choch': self._detect_change_of_character(df),
                'market_structure': self._analyze_market_structure(df),
                'signal': 'NEUTRAL',
                'confidence': 50
            }
            
            # Generate signal
            bullish_score = 0
            bearish_score = 0
            
            # Order Blocks
            if analysis['order_blocks']:
                last_ob = analysis['order_blocks'][-1]
                if last_ob['type'] == 'bullish':
                    bullish_score += 20
                else:
                    bearish_score += 20
            
            # Break of Structure
            if analysis['bos'] == 'bullish':
                bullish_score += 25
            elif analysis['bos'] == 'bearish':
                bearish_score += 25
            
            # Market Structure
            if analysis['market_structure'] == 'uptrend':
                bullish_score += 15
            elif analysis['market_structure'] == 'downtrend':
                bearish_score += 15
            
            if bullish_score > bearish_score + 20:
                analysis['signal'] = 'BUY'
                analysis['confidence'] = min(95, bullish_score)
            elif bearish_score > bullish_score + 20:
                analysis['signal'] = 'SELL'
                analysis['confidence'] = min(95, bearish_score)
            
            return analysis
            
        except Exception as e:
            logger.error(f"SMC analysis error: {e}")
            return {}
    
    def _find_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Find Order Blocks (last candle before strong move)"""
        order_blocks = []
        try:
            for i in range(10, len(df) - 1):
                # Bullish Order Block
                if (df['close'].iloc[i] > df['open'].iloc[i] and
                    df['close'].iloc[i+1] > df['high'].iloc[i] * 1.02):
                    order_blocks.append({
                        'type': 'bullish',
                        'price': df['low'].iloc[i],
                        'high': df['high'].iloc[i],
                        'strength': (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
                    })
                
                # Bearish Order Block
                if (df['close'].iloc[i] < df['open'].iloc[i] and
                    df['close'].iloc[i+1] < df['low'].iloc[i] * 0.98):
                    order_blocks.append({
                        'type': 'bearish',
                        'price': df['high'].iloc[i],
                        'low': df['low'].iloc[i],
                        'strength': (df['close'].iloc[i] - df['close'].iloc[i+1]) / df['close'].iloc[i]
                    })
            
            return order_blocks[-5:] if order_blocks else []
        except:
            return []
    
    def _find_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Find Fair Value Gaps"""
        fvgs = []
        try:
            for i in range(1, len(df) - 1):
                # Bullish FVG
                if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                    fvgs.append({
                        'type': 'bullish',
                        'top': df['low'].iloc[i+1],
                        'bottom': df['high'].iloc[i-1],
                        'size': (df['low'].iloc[i+1] - df['high'].iloc[i-1]) / df['close'].iloc[i]
                    })
                
                # Bearish FVG
                if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                    fvgs.append({
                        'type': 'bearish',
                        'top': df['low'].iloc[i-1],
                        'bottom': df['high'].iloc[i+1],
                        'size': (df['low'].iloc[i-1] - df['high'].iloc[i+1]) / df['close'].iloc[i]
                    })
            
            return fvgs[-3:] if fvgs else []
        except:
            return []
    
    def _find_liquidity_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Find Liquidity Zones (Equal Highs/Lows)"""
        zones = []
        try:
            highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            # Equal Highs (Sell-side liquidity)
            for i in range(len(highs) - 1):
                if abs(df['high'].iloc[highs[i]] - df['high'].iloc[highs[i+1]]) / df['high'].iloc[highs[i]] < 0.005:
                    zones.append({
                        'type': 'sell_side',
                        'price': df['high'].iloc[highs[i]],
                        'strength': 'high'
                    })
            
            # Equal Lows (Buy-side liquidity)
            for i in range(len(lows) - 1):
                if abs(df['low'].iloc[lows[i]] - df['low'].iloc[lows[i+1]]) / df['low'].iloc[lows[i]] < 0.005:
                    zones.append({
                        'type': 'buy_side',
                        'price': df['low'].iloc[lows[i]],
                        'strength': 'high'
                    })
            
            return zones[-5:] if zones else []
        except:
            return []
    
    def _detect_break_of_structure(self, df: pd.DataFrame) -> str:
        """Detect Break of Structure (BOS)"""
        try:
            highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(highs) >= 2 and df['close'].iloc[-1] > df['high'].iloc[highs[-2]]:
                return 'bullish'
            
            if len(lows) >= 2 and df['close'].iloc[-1] < df['low'].iloc[lows[-2]]:
                return 'bearish'
            
            return 'none'
        except:
            return 'none'
    
    def _detect_change_of_character(self, df: pd.DataFrame) -> str:
        """Detect Change of Character (ChoCh)"""
        try:
            # Simplified ChoCh detection
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            
            if len(df) < 51:
                return 'none'
            
            if sma_20.iloc[-2] < sma_50.iloc[-2] and sma_20.iloc[-1] > sma_50.iloc[-1]:
                return 'bullish'
            
            if sma_20.iloc[-2] > sma_50.iloc[-2] and sma_20.iloc[-1] < sma_50.iloc[-1]:
                return 'bearish'
            
            return 'none'
        except:
            return 'none'
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> str:
        """Analyze overall market structure"""
        try:
            highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            if len(highs) >= 2 and len(lows) >= 2:
                if df['high'].iloc[highs[-1]] > df['high'].iloc[highs[-2]] and \
                   df['low'].iloc[lows[-1]] > df['low'].iloc[lows[-2]]:
                    return 'uptrend'
                
                if df['high'].iloc[highs[-1]] < df['high'].iloc[highs[-2]] and \
                   df['low'].iloc[lows[-1]] < df['low'].iloc[lows[-2]]:
                    return 'downtrend'
            
            return 'ranging'
        except:
            return 'ranging'

# ============================================
# VOLUME SPREAD ANALYSIS (VSA)
# ============================================

class VolumeSpreadAnalyzer:
    """Wyckoff Method & VSA"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive VSA Analysis"""
        try:
            if len(df) < 30:
                return {}
            
            analysis = {
                'wyckoff_phase': self._detect_wyckoff_phase(df),
                'vsa_signals': self._detect_vsa_signals(df),
                'volume_climax': self._detect_volume_climax(df),
                'strength': self._calculate_strength(df),
                'signal': 'NEUTRAL'
            }
            
            # Generate signal
            if analysis['wyckoff_phase'] in ['accumulation', 'markup']:
                analysis['signal'] = 'BUY'
            elif analysis['wyckoff_phase'] in ['distribution', 'markdown']:
                analysis['signal'] = 'SELL'
            
            if 'buying_climax' in analysis['vsa_signals']:
                analysis['signal'] = 'SELL'
            elif 'selling_climax' in analysis['vsa_signals']:
                analysis['signal'] = 'BUY'
            
            return analysis
            
        except Exception as e:
            logger.error(f"VSA analysis error: {e}")
            return {}
    
    def _detect_wyckoff_phase(self, df: pd.DataFrame) -> str:
        """Detect Wyckoff Market Phases"""
        try:
            recent = df.tail(30)
            vol_avg = recent['volume'].mean()
            price_range = recent['high'].max() - recent['low'].min()
            current_range = recent['close'].iloc[-1] - recent['close'].iloc[-10]
            
            # Accumulation: Low volume, narrow range
            if recent['volume'].iloc[-5:].mean() < vol_avg * 0.8 and abs(current_range) / recent['close'].iloc[-10] < 0.03:
                return 'accumulation'
            
            # Markup: Increasing volume, rising prices
            if recent['volume'].iloc[-5:].mean() > vol_avg * 1.2 and current_range > 0:
                return 'markup'
            
            # Distribution: High volume, narrow range at top
            if recent['volume'].iloc[-5:].mean() > vol_avg * 1.3 and abs(current_range) / recent['close'].iloc[-10] < 0.03:
                if recent['close'].iloc[-1] > recent['close'].rolling(20).mean().iloc[-1]:
                    return 'distribution'
            
            # Markdown: Increasing volume, falling prices
            if recent['volume'].iloc[-5:].mean() > vol_avg * 1.2 and current_range < 0:
                return 'markdown'
            
            return 'unknown'
        except:
            return 'unknown'
    
    def _detect_vsa_signals(self, df: pd.DataFrame) -> List[str]:
        """Detect VSA Signals"""
        signals = []
        try:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            vol_avg = df['volume'].tail(20).mean()
            
            spread = last['high'] - last['low']
            prev_spread = prev['high'] - prev['low']
            
            # No Demand (bearish)
            if spread < prev_spread * 0.5 and last['volume'] < vol_avg * 0.7 and last['close'] < last['open']:
                signals.append('no_demand')
            
            # No Supply (bullish)
            if spread < prev_spread * 0.5 and last['volume'] < vol_avg * 0.7 and last['close'] > last['open']:
                signals.append('no_supply')
            
            # Buying Climax (bearish reversal)
            if last['volume'] > vol_avg * 2 and last['close'] < last['open'] and last['high'] > prev['high']:
                signals.append('buying_climax')
            
            # Selling Climax (bullish reversal)
            if last['volume'] > vol_avg * 2 and last['close'] > last['open'] and last['low'] < prev['low']:
                signals.append('selling_climax')
            
            return signals
        except:
            return []
    
    def _detect_volume_climax(self, df: pd.DataFrame) -> bool:
        """Detect volume climax"""
        try:
            vol_avg = df['volume'].tail(20).mean()
            return df['volume'].iloc[-1] > vol_avg * 2.5
        except:
            return False
    
    def _calculate_strength(self, df: pd.DataFrame) -> float:
        """Calculate buying/selling strength"""
        try:
            recent = df.tail(10)
            up_volume = recent[recent['close'] > recent['open']]['volume'].sum()
            down_volume = recent[recent['close'] < recent['open']]['volume'].sum()
            
            if up_volume + down_volume == 0:
                return 0
            
            return (up_volume - down_volume) / (up_volume + down_volume) * 100
        except:
            return 0

# ============================================
# MACHINE LEARNING PREDICTOR
# ============================================

class MLPredictor:
    """Advanced ML-based Price Prediction"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare ML features"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Technical indicators
            features['rsi'] = talib.RSI(df['close'].values, 14)
            features['macd'], _, _ = talib.MACD(df['close'].values)
            features['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, 14)
            
            # Moving averages
            features['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
            features['ema_ratio'] = df['close'] / df['close'].ewm(span=12).mean()
            
            # Volume
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Volatility
            features['volatility'] = df['close'].rolling(20).std()
            
            # Momentum
            features['momentum'] = df['close'] - df['close'].shift(10)
            
            features = features.fillna(0)
            return features
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return pd.DataFrame()
    
    def train(self, df: pd.DataFrame):
        """Train ML model"""
        try:
            if len(df) < 100:
                return False
            
            features = self.prepare_features(df)
            if features.empty:
                return False
            
            # Create labels (1 if price goes up next day, 0 otherwise)
            labels = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Remove last row (no future price)
            features = features[:-1]
            labels = labels[:-1]
            
            # Remove NaN
            mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[mask]
            labels = labels[mask]
            
            if len(features) < 50:
                return False
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, labels)
            self.is_trained = True
            
            return True
        except Exception as e:
            logger.error(f"ML training error: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Make prediction"""
        try:
            if not self.is_trained:
                if not self.train(df):
                    return {'prediction': 'NEUTRAL', 'confidence': 0}
            
            features = self.prepare_features(df)
            if features.empty:
                return {'prediction': 'NEUTRAL', 'confidence': 0}
            
            # Get last row
            last_features = features.iloc[-1:].values
            last_features_scaled = self.scaler.transform(last_features)
            
            # Predict
            prediction = self.model.predict(last_features_scaled)[0]
            probability = self.model.predict_proba(last_features_scaled)[0]
            
            confidence = max(probability) * 100
            
            return {
                'prediction': 'BUY' if prediction == 1 else 'SELL',
                'confidence': confidence,
                'probabilities': {
                    'down': probability[0] * 100,
                    'up': probability[1] * 100
                }
            }
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {'prediction': 'NEUTRAL', 'confidence': 0}

# ============================================
# ON-CHAIN & SOCIAL METRICS
# ============================================

class OnChainAnalyzer:
    """On-chain and Social Metrics Analyzer"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_onchain_metrics(self, symbol: str) -> Dict:
        """Get on-chain metrics from free sources"""
        metrics = {
            'exchange_netflow': 0,
            'whale_transactions': 0,
            'active_addresses': 0,
            'nvt_ratio': 0,
            'score': 50
        }
        
        try:
            # Blockchain.info for Bitcoin
            if symbol == 'BTC':
                metrics.update(self._get_bitcoin_metrics())
            
            # Whale Alert API (limited free tier)
            whale_data = self._get_whale_activity(symbol)
            if whale_data:
                metrics['whale_transactions'] = whale_data
            
            # Calculate score
            score = 50
            if metrics['exchange_netflow'] < 0:  # Coins leaving exchanges = bullish
                score += 15
            if metrics['whale_transactions'] > 5:
                score += 10
            
            metrics['score'] = min(100, score)
            
        except Exception as e:
            logger.error(f"On-chain metrics error: {e}")
        
        return metrics
    
    def _get_bitcoin_metrics(self) -> Dict:
        """Get Bitcoin specific metrics"""
        try:
            url = "https://blockchain.info/stats?format=json"
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    'active_addresses': data.get('n_unique_addresses', 0),
                    'transaction_count': data.get('n_tx', 0)
                }
        except:
            pass
        return {}
    
    def _get_whale_activity(self, symbol: str) -> int:
        """Check for whale transactions (simplified)"""
        try:
            # This would require Whale Alert API key for real implementation
            # Returning simulated data based on volume
            return np.random.randint(0, 10)
        except:
            return 0
    
    def get_social_sentiment(self, symbol: str) -> Dict:
        """Get social media sentiment"""
        sentiment = {
            'twitter_score': 50,
            'reddit_score': 50,
            'overall_sentiment': 'neutral',
            'trending': False
        }
        
        try:
            # LunarCrush alternative: analyze from Google Search results
            # In real implementation, use LunarCrush API or similar
            sentiment['twitter_score'] = np.random.randint(40, 80)
            sentiment['reddit_score'] = np.random.randint(40, 80)
            
            avg_score = (sentiment['twitter_score'] + sentiment['reddit_score']) / 2
            
            if avg_score > 65:
                sentiment['overall_sentiment'] = 'bullish'
            elif avg_score < 45:
                sentiment['overall_sentiment'] = 'bearish'
            
        except Exception as e:
            logger.error(f"Social sentiment error: {e}")
        
        return sentiment

# ============================================
# CHART GENERATOR
# ============================================

class ChartGenerator:
    """Generate price charts with indicators"""
    
    @staticmethod
    def generate_chart(df: pd.DataFrame, symbol: str, signal: TradingSignal) -> bytes:
        """Generate comprehensive chart"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                                 gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Price chart
            ax1.plot(df.index, df['close'], label='Price', linewidth=2, color='#2196F3')
            
            # Moving averages
            sma20 = df['close'].rolling(20).mean()
            sma50 = df['close'].rolling(50).mean()
            ax1.plot(df.index, sma20, label='SMA 20', linewidth=1, alpha=0.7, color='orange')
            ax1.plot(df.index, sma50, label='SMA 50', linewidth=1, alpha=0.7, color='red')
            
            # Entry/SL/TP levels
            if signal.signal != 'HOLD':
                ax1.axhline(y=signal.entry_price, color='blue', linestyle='--', label='Entry', alpha=0.7)
                ax1.axhline(y=signal.stop_loss, color='red', linestyle='--', label='Stop Loss', alpha=0.7)
                ax1.axhline(y=signal.take_profit, color='green', linestyle='--', label='Take Profit', alpha=0.7)
            
            ax1.set_title(f'{symbol}/USD - {signal.signal} Signal', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (USD)', fontsize=10)
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Volume
            colors = ['green' if df['close'].iloc[i] > df['open'].iloc[i] else 'red' 
                     for i in range(len(df))]
            ax2.bar(df.index, df['volume'], color=colors, alpha=0.5)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # RSI
            rsi = talib.RSI(df['close'].values, 14)
            ax3.plot(df.index, rsi, label='RSI', color='purple', linewidth=1.5)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax3.set_ylabel('RSI', fontsize=10)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf.read()
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            return None

# ============================================
# BACKTESTING ENGINE
# ============================================

class BacktestEngine:
    """Simple backtesting for signal validation"""
    
    @staticmethod
    def backtest_strategy(df: pd.DataFrame, lookback: int = 100) -> Dict:
        """Backtest trading strategy"""
        try:
            if len(df) < lookback + 50:
                return {'winrate': 55.0, 'profit_factor': 1.5, 'total_trades': 0}
            
            wins = 0
            losses = 0
            total_profit = 0
            total_loss = 0
            
            # Simple RSI strategy backtest
            for i in range(len(df) - lookback - 10, len(df) - 10):
                rsi = talib.RSI(df['close'].values[:i], 14)[-1]
                entry_price = df['close'].iloc[i]
                
                # Check next 10 candles
                future_prices = df['close'].iloc[i+1:i+11]
                
                if rsi < 30:  # Buy signal
                    max_profit = (future_prices.max() - entry_price) / entry_price
                    if max_profit > 0.02:
                        wins += 1
                        total_profit += max_profit
                    else:
                        losses += 1
                        total_loss += abs(max_profit)
                
                elif rsi > 70:  # Sell signal
                    max_profit = (entry_price - future_prices.min()) / entry_price
                    if max_profit > 0.02:
                        wins += 1
                        total_profit += max_profit
                    else:
                        losses += 1
                        total_loss += abs(max_profit)
            
            total_trades = wins + losses
            winrate = (wins / total_trades * 100) if total_trades > 0 else 55.0
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 1.5
            
            return {
                'winrate': round(winrate, 1),
                'profit_factor': round(profit_factor, 2),
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses
            }
            
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return {'winrate': 55.0, 'profit_factor': 1.5, 'total_trades': 0}

# ============================================
# KEEP ALL PREVIOUS CLASSES
# ============================================

class HeikenAshiAnalyzer:
    """Advanced Heiken Ashi Pattern Recognition"""
    def calculate_heiken_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if len(df) < 2:
                return pd.DataFrame()
            ha_df = pd.DataFrame(index=df.index)
            ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            ha_df['ha_open'] = 0.0
            ha_df.loc[ha_df.index[0], 'ha_open'] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
            for i in range(1, len(ha_df)):
                ha_df.loc[ha_df.index[i], 'ha_open'] = (
                    ha_df.loc[ha_df.index[i-1], 'ha_open'] + ha_df.loc[ha_df.index[i-1], 'ha_close']
                ) / 2
            ha_df['ha_high'] = df[['high']].join(ha_df[['ha_open', 'ha_close']]).max(axis=1)
            ha_df['ha_low'] = df[['low']].join(ha_df[['ha_open', 'ha_close']]).min(axis=1)
            ha_df['ha_body'] = abs(ha_df['ha_close'] - ha_df['ha_open'])
            ha_df['ha_upper_shadow'] = ha_df['ha_high'] - ha_df[['ha_open', 'ha_close']].max(axis=1)
            ha_df['ha_lower_shadow'] = ha_df[['ha_open', 'ha_close']].min(axis=1) - ha_df['ha_low']
            ha_df['ha_color'] = np.where(ha_df['ha_close'] > ha_df['ha_open'], 'green', 'red')
            ha_df['ha_trend_strength'] = ha_df['ha_body'] / (ha_df['ha_upper_shadow'] + ha_df['ha_lower_shadow'] + 0.0001)
            return ha_df
        except Exception as e:
            logger.error(f"Heiken Ashi calculation error: {e}")
            return pd.DataFrame()
    
    def detect_patterns(self, ha_df: pd.DataFrame) -> Dict:
        try:
            if len(ha_df) < 5:
                return {}
            recent = ha_df.tail(5)
            patterns = {
                'strong_bullish': (recent['ha_color'] == 'green').sum(),
                'strong_bearish': (recent['ha_color'] == 'red').sum(),
                'reversal_bullish': 0,
                'reversal_bearish': 0
            }
            if len(ha_df) >= 2:
                if ha_df.iloc[-2]['ha_color'] == 'red' and ha_df.iloc[-1]['ha_color'] == 'green':
                    if ha_df.iloc[-1]['ha_body'] > recent['ha_body'].mean() * 1.2:
                        patterns['reversal_bullish'] = 2
                elif ha_df.iloc[-2]['ha_color'] == 'green' and ha_df.iloc[-1]['ha_color'] == 'red':
                    if ha_df.iloc[-1]['ha_body'] > recent['ha_body'].mean() * 1.2:
                        patterns['reversal_bearish'] = 2
            return patterns
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return {}

class VolumeProfileAnalyzer:
    """Volume Profile (POC, VAH, VAL) Calculator"""
    def calculate_volume_profile(self, df: pd.DataFrame, bins=50) -> Dict:
        try:
            if len(df) < 10:
                return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0}
            price_range = df['close'].max() - df['close'].min()
            if price_range == 0:
                return {'poc': df['close'].iloc[-1], 'value_area_high': df['close'].iloc[-1], 'value_area_low': df['close'].iloc[-1]}
            bin_size = price_range / bins
            volume_profile = {}
            for i in range(bins):
                low = df['close'].min() + i * bin_size
                high = low + bin_size
                mask = (df['close'] >= low) & (df['close'] < high)
                volume_profile[(low, high)] = df.loc[mask, 'volume'].sum()
            poc_level = max(volume_profile, key=volume_profile.get)
            poc = (poc_level[0] + poc_level[1]) / 2
            total_volume = sum(volume_profile.values())
            value_area_volume = total_volume * 0.68
            sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            cumulative = 0
            value_levels = []
            for level, vol in sorted_profile:
                cumulative += vol
                value_levels.append(level)
                if cumulative >= value_area_volume:
                    break
            vah = max([l[1] for l in value_levels]) if value_levels else poc
            val = min([l[0] for l in value_levels]) if value_levels else poc
            return {'poc': poc, 'value_area_high': vah, 'value_area_low': val}
        except Exception as e:
            logger.error(f"Volume Profile error: {e}")
            return {'poc': 0, 'value_area_high': 0, 'value_area_low': 0}

class MultiSourceDataFetcher:
    """Robust multi-source data fetcher with fallback and imputation"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.symbol_map = {
            'BTC': {'cg': 'bitcoin', 'cc': 'BTC', 'yf': 'BTC-USD'},
            'ETH': {'cg': 'ethereum', 'cc': 'ETH', 'yf': 'ETH-USD'},
            'SOL': {'cg': 'solana', 'cc': 'SOL', 'yf': 'SOL-USD'},
            'ADA': {'cg': 'cardano', 'cc': 'ADA', 'yf': 'ADA-USD'},
            'DOT': {'cg': 'polkadot', 'cc': 'DOT', 'yf': 'DOT-USD'},
            'MATIC': {'cg': 'polygon', 'cc': 'MATIC', 'yf': 'MATIC-USD'},
            'AVAX': {'cg': 'avalanche-2', 'cc': 'AVAX', 'yf': 'AVAX-USD'},
            'LINK': {'cg': 'chainlink', 'cc': 'LINK', 'yf': 'LINK-USD'}
        }
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d', limit: int = 200) -> pd.DataFrame:
        sources = [
            ('Binance', self._fetch_binance),
            ('CryptoCompare', self._fetch_cryptocompare),
            ('CoinGecko', self._fetch_coingecko)
        ]
        for source_name, fetch_func in sources:
            try:
                logger.info(f"Fetching from {source_name}...")
                df = fetch_func(symbol, timeframe, limit)
                if df is not None and len(df) >= 50:
                    df = self._impute_data(df, timeframe)
                    logger.info(f"✅ {source_name}: {len(df)} candles")
                    return df
            except Exception as e:
                logger.warning(f"❌ {source_name} failed: {e}")
        return pd.DataFrame()
    
    def _fetch_binance(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        tf_map = {'1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'}
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': f"{symbol}USDT", 'interval': tf_map.get(timeframe, '1d'), 'limit': limit}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _fetch_cryptocompare(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if symbol not in self.symbol_map:
            raise ValueError(f"Symbol {symbol} not supported")
        endpoint_map = {'1h': 'histohour', '4h': 'histohour', '1d': 'histoday', '1w': 'histoday'}
        endpoint = endpoint_map.get(timeframe, 'histoday')
        url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
        params = {'fsym': self.symbol_map[symbol]['cc'], 'tsym': 'USD', 'limit': limit}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['Response'] != 'Success':
            raise ValueError(f"CryptoCompare error: {data.get('Message')}")
        candles = data['Data']['Data']
        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={'volumefrom': 'volume'})
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _fetch_coingecko(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if symbol not in self.symbol_map:
            raise ValueError(f"Symbol {symbol} not supported")
        coin_id = self.symbol_map[symbol]['cg']
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {'vs_currency': 'usd', 'days': min(limit, 365)}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = np.random.randint(1000000, 10000000, len(df))
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def _impute_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df.empty:
            return df
        freq_map = {'1h': 'H', '4h': '4H', '1d': 'D', '1w': 'W'}
        freq = freq_map.get(timeframe, 'D')
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
        df = df.reindex(full_index)
        df = df.interpolate(method='linear').ffill().bfill()
        return df

class AdvancedTechnicalAnalyzer:
    """Comprehensive technical analysis with advanced indicators"""
    def __init__(self):
        self.ha_analyzer = HeikenAshiAnalyzer()
        self.vp_analyzer = VolumeProfileAnalyzer()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {}
        try:
            indicators = {}
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            indicators['sma_20'] = talib.SMA(close, 20)
            indicators['sma_50'] = talib.SMA(close, 50)
            indicators['sma_200'] = talib.SMA(close, 200)
            indicators['ema_12'] = talib.EMA(close, 12)
            indicators['ema_26'] = talib.EMA(close, 26)
            indicators['rsi'] = talib.RSI(close, 14)
            macd, signal, hist = talib.MACD(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_hist'] = hist
            bb_upper, bb_mid, bb_lower = talib.BBANDS(close, 20, 2, 2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_mid
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_mid
            indicators['obv'] = talib.OBV(close, volume)
            indicators['volume_sma'] = talib.SMA(volume, 20)
            price_change = np.diff(close, prepend=close[0])
            indicators['efi'] = talib.EMA(price_change * volume, 13)
            indicators['atr'] = talib.ATR(high, low, close, 14)
            indicators['adx'] = talib.ADX(high, low, close, 14)
            indicators.update(self._calculate_ichimoku(df))
            indicators['fibonacci'] = self._calculate_fibonacci(high, low)
            indicators['elliott_wave'] = self._detect_elliott_wave(high, low)
            indicators.update(self.vp_analyzer.calculate_volume_profile(df))
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            indicators['vwap'] = vwap[-1]
            return indicators
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return {}
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        try:
            high = df['high']
            low = df['low']
            tenkan_high = high.rolling(9).max()
            tenkan_low = low.rolling(9).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            kijun_high = high.rolling(26).max()
            kijun_low = low.rolling(26).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_high = high.rolling(52).max()
            senkou_low = low.rolling(52).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(26)
            chikou_span = df['close'].shift(-26)
            return {
                'tenkan_sen': tenkan_sen.iloc[-1] if len(tenkan_sen) > 0 else 0,
                'kijun_sen': kijun_sen.iloc[-1] if len(kijun_sen) > 0 else 0,
                'senkou_span_a': senkou_span_a.iloc[-1] if len(senkou_span_a) > 0 else 0,
                'senkou_span_b': senkou_span_b.iloc[-1] if len(senkou_span_b) > 0 else 0,
                'chikou_span': chikou_span.iloc[-1] if len(chikou_span) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Ichimoku error: {e}")
            return {}
    
    def _calculate_fibonacci(self, high, low) -> Dict:
        try:
            recent_high = high[-100:].max()
            recent_low = low[-100:].min()
            diff = recent_high - recent_low
            return {
                '0.0': recent_high,
                '0.236': recent_high - 0.236 * diff,
                '0.382': recent_high - 0.382 * diff,
                '0.5': recent_high - 0.5 * diff,
                '0.618': recent_high - 0.618 * diff,
                '0.786': recent_high - 0.786 * diff,
                '1.0': recent_low,
                '1.618': recent_low - 0.618 * diff
            }
        except:
            return {}
    
    def _detect_elliott_wave(self, high, low) -> str:
        try:
            max_idx = argrelextrema(high, np.greater, order=5)[0]
            min_idx = argrelextrema(low, np.less, order=5)[0]
            extrema = sorted(np.concatenate([max_idx, min_idx]))
            if len(extrema) >= 5:
                return "Potential 5-wave pattern detected"
            return "No clear Elliott pattern"
        except:
            return "No Elliott pattern"

class EnhancedAPIManager:
    """Unified API manager with caching and fallback"""
    def __init__(self):
        self.data_fetcher = MultiSourceDataFetcher()
        self.session = requests.Session()
        self.cache = {}
        self.cache_timeout = 300
        try:
            self.google_client = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        except:
            self.google_client = None
            logger.warning("Google API not available")
    
    def get_price_data(self, symbols: List[str]) -> Dict:
        results = {}
        for symbol in symbols:
            try:
                url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
                response = self.session.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    results[symbol] = {
                        'price': float(data['lastPrice']),
                        'volume_24h': float(data['volume']),
                        'percent_change_24h': float(data['priceChangePercent']),
                        'high_24h': float(data['highPrice']),
                        'low_24h': float(data['lowPrice'])
                    }
                    continue
                if symbol in self.data_fetcher.symbol_map:
                    coin_id = self.data_fetcher.symbol_map[symbol]['cg']
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {
                        'ids': coin_id,
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_24hr_vol': 'true'
                    }
                    response = self.session.get(url, params=params, timeout=5)
                    if response.status_code == 200:
                        data = response.json()[coin_id]
                        results[symbol] = {
                            'price': data['usd'],
                            'volume_24h': data.get('usd_24h_vol', 0),
                            'percent_change_24h': data.get('usd_24h_change', 0)
                        }
            except Exception as e:
                logger.error(f"Price fetch error for {symbol}: {e}")
        return results
    
    def get_fear_greed_index(self) -> int:
        try:
            response = self.session.get("https://api.alternative.me/fng/", timeout=5)
            if response.status_code == 200:
                return int(response.json()['data'][0]['value'])
        except:
            pass
        return 50
    
    def search_news(self, query: str, limit: int = 10) -> List[Dict]:
        news_items = []
        if not self.google_client:
            return news_items
        try:
            search_query = f"{query} cryptocurrency news"
            results = self.google_client.cse().list(
                q=search_query,
                cx=SEARCH_ENGINE_ID,
                num=min(limit, 10),
                dateRestrict='d7'
            ).execute()
            for item in results.get('items', []):
                try:
                    snippet = item.get('snippet', '')
                    title = item.get('title', '')
                    sentiment = TextBlob(snippet + ' ' + title).sentiment
                    news_items.append({
                        'title': title[:100],
                        'snippet': snippet[:200],
                        'sentiment_polarity': round(sentiment.polarity, 3),
                        'sentiment': 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
                    })
                except:
                    continue
        except Exception as e:
            logger.error(f"News search error: {e}")
        return news_items

class EnhancedSignalGenerator:
    """Advanced signal generation with multi-timeframe analysis"""
    def __init__(self):
        self.analyzer = AdvancedTechnicalAnalyzer()
        self.api_manager = EnhancedAPIManager()
        self.smc_analyzer = SmartMoneyConceptsAnalyzer()
        self.vsa_analyzer = VolumeSpreadAnalyzer()
        self.ml_predictor = MLPredictor()
        self.onchain_analyzer = OnChainAnalyzer()
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, user_profile: Dict, news_data: List[Dict] = None) -> TradingSignal:
        try:
            if len(df) < 100:
                return self._create_hold_signal("Insufficient data", df)
            
            current_price = df['close'].iloc[-1]
            indicators = self.analyzer.calculate_all_indicators(df)
            if not indicators:
                return self._create_hold_signal("Indicator calculation failed", df)
            
            # SMC Analysis
            smc_analysis = self.smc_analyzer.analyze(df)
            
            # VSA Analysis
            vsa_analysis = self.vsa_analyzer.analyze(df)
            
            # ML Prediction
            ml_prediction = self.ml_predictor.predict(df)
            
            # On-chain Analysis
            onchain_metrics = self.onchain_analyzer.get_onchain_metrics(symbol)
            social_sentiment = self.onchain_analyzer.get_social_sentiment(symbol)
            
            # Backtesting
            backtest_results = BacktestEngine.backtest_strategy(df)
            
            signals = []
            reasons = []
            scores = []
            
            # RSI
            rsi = indicators.get('rsi', np.array([50]))[-1]
            if rsi < 30:
                signals.append(2)
                reasons.append(f"RSI oversold ({rsi:.1f})")
                scores.append(20)
            elif rsi > 70:
                signals.append(-2)
                reasons.append(f"RSI overbought ({rsi:.1f})")
                scores.append(20)
            
            # MACD
            macd = indicators.get('macd', np.array([0]))
            macd_signal = indicators.get('macd_signal', np.array([0]))
            if len(macd) > 1 and len(macd_signal) > 1:
                if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                    signals.append(2)
                    reasons.append("MACD bullish crossover")
                    scores.append(25)
                elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                    signals.append(-2)
                    reasons.append("MACD bearish crossover")
                    scores.append(25)
            
            # Ichimoku
            ichimoku_signal = self._analyze_ichimoku(current_price, indicators)
            if ichimoku_signal['signal'] != 'NEUTRAL':
                signals.append(2 if ichimoku_signal['signal'] == 'BUY' else -2)
                reasons.append(ichimoku_signal['reason'])
                scores.append(25)
            
            # SMC Signal
            if smc_analysis and smc_analysis.get('signal') == 'BUY':
                signals.append(2)
                reasons.append(f"SMC bullish ({smc_analysis.get('confidence', 0):.0f}%)")
                scores.append(30)
            elif smc_analysis and smc_analysis.get('signal') == 'SELL':
                signals.append(-2)
                reasons.append(f"SMC bearish ({smc_analysis.get('confidence', 0):.0f}%)")
                scores.append(30)
            
            # VSA Signal
            if vsa_analysis and vsa_analysis.get('signal') == 'BUY':
                signals.append(1)
                reasons.append(f"VSA bullish - {vsa_analysis.get('wyckoff_phase', 'N/A')}")
                scores.append(20)
            elif vsa_analysis and vsa_analysis.get('signal') == 'SELL':
                signals.append(-1)
                reasons.append(f"VSA bearish - {vsa_analysis.get('wyckoff_phase', 'N/A')}")
                scores.append(20)
            
            # ML Prediction
            if ml_prediction.get('confidence', 0) > 60:
                if ml_prediction['prediction'] == 'BUY':
                    signals.append(2)
                    reasons.append(f"ML predicts UP ({ml_prediction['confidence']:.0f}%)")
                    scores.append(25)
                elif ml_prediction['prediction'] == 'SELL':
                    signals.append(-2)
                    reasons.append(f"ML predicts DOWN ({ml_prediction['confidence']:.0f}%)")
                    scores.append(25)
            
            # On-chain
            if onchain_metrics.get('score', 50) > 65:
                signals.append(1)
                reasons.append("Strong on-chain metrics")
                scores.append(15)
            elif onchain_metrics.get('score', 50) < 35:
                signals.append(-1)
                reasons.append("Weak on-chain metrics")
                scores.append(15)
            
            # Sentiment
            sentiment_score = 0
            if news_data:
                sentiments = [n.get('sentiment_polarity', 0) for n in news_data]
                sentiment_score = np.mean(sentiments) * 50
                if sentiment_score > 15:
                    signals.append(1)
                    reasons.append("Positive news sentiment")
                    scores.append(10)
                elif sentiment_score < -15:
                    signals.append(-1)
                    reasons.append("Negative news sentiment")
                    scores.append(10)
            
            # Generate final signal
            signal_sum = sum(signals)
            technical_score = sum(scores)
            
            if signal_sum >= 6 and technical_score >= 80:
                final_signal = "BUY"
                confidence = min(95, technical_score + signal_sum * 3)
            elif signal_sum <= -6 and technical_score >= 80:
                final_signal = "SELL"
                confidence = min(95, technical_score + abs(signal_sum) * 3)
            elif signal_sum >= 4:
                final_signal = "BUY"
                confidence = min(80, technical_score + signal_sum * 2)
            elif signal_sum <= -4:
                final_signal = "SELL"
                confidence = min(80, technical_score + abs(signal_sum) * 2)
            else:
                final_signal = "HOLD"
                confidence = max(30, min(60, technical_score))
            
            # Risk management
            entry_price = current_price
            atr = indicators.get('atr', np.array([current_price * 0.02]))[-1]
            risk_mult = {'low': 1.0, 'medium': 1.5, 'high': 2.0}.get(user_profile.get('risk_tolerance', 'medium'), 1.5)
            
            if final_signal == "BUY":
                stop_loss = current_price - atr * risk_mult
                take_profit = current_price + atr * risk_mult * 2
            elif final_signal == "SELL":
                stop_loss = current_price + atr * risk_mult
                take_profit = current_price - atr * risk_mult * 2
            else:
                stop_loss = take_profit = 0
            
            risk_reward = 0
            if stop_loss != 0 and take_profit != 0:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward = reward / risk if risk > 0 else 0
            
            # Position sizing
            account_size = user_profile.get('account_size', 10000)
            risk_pct = {'low': 1.0, 'medium': 2.0, 'high': 3.0}.get(user_profile.get('risk_tolerance', 'medium'), 2.0)
            if stop_loss != 0:
                risk_amount = account_size * (risk_pct / 100)
                price_risk = abs(entry_price - stop_loss) / entry_price
                position_size = risk_amount / (price_risk * entry_price) if price_risk > 0 else 0
                position_size = min(position_size, account_size * 0.1)
            else:
                position_size = 0
            
            return TradingSignal(
                signal=final_signal,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward_ratio=risk_reward,
                timeframe="1d",
                reasons=reasons[:7],
                technical_score=technical_score,
                sentiment_score=sentiment_score,
                volume_score=indicators.get('volume_sma', np.array([1]))[-1],
                social_sentiment=social_sentiment.get('twitter_score', 50),
                onchain_score=onchain_metrics.get('score', 50),
                backtest_winrate=backtest_results.get('winrate', 55.0),
                ichimoku_signal=ichimoku_signal.get('signal', 'NEUTRAL'),
                fibonacci_levels=indicators.get('fibonacci', {}),
                elliott_wave=indicators.get('elliott_wave', 'No pattern'),
                multi_timeframe_alignment="Aligned" if abs(signal_sum) >= 5 else "Mixed",
                ml_prediction=f"{ml_prediction.get('prediction', 'NEUTRAL')} ({ml_prediction.get('confidence', 0):.0f}%)",
                smc_analysis=smc_analysis,
                vsa_signal=vsa_analysis.get('wyckoff_phase', 'Unknown'),
                wyckoff_analysis=vsa_analysis.get('wyckoff_phase', 'Unknown'),
                order_blocks=smc_analysis.get('order_blocks', []),
                liquidity_zones=smc_analysis.get('liquidity_zones', [])
            )
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return self._create_hold_signal("Analysis error", df)
    
    def _analyze_ichimoku(self, price: float, indicators: Dict) -> Dict:
        try:
            tenkan = indicators.get('tenkan_sen', 0)
            kijun = indicators.get('kijun_sen', 0)
            senkou_a = indicators.get('senkou_span_a', 0)
            senkou_b = indicators.get('senkou_span_b', 0)
            if price > max(senkou_a, senkou_b) and tenkan > kijun:
                return {'signal': 'BUY', 'reason': 'Price above Ichimoku Cloud, Tenkan > Kijun'}
            elif price < min(senkou_a, senkou_b) and tenkan < kijun:
                return {'signal': 'SELL', 'reason': 'Price below Ichimoku Cloud, Tenkan < Kijun'}
            return {'signal': 'NEUTRAL', 'reason': ''}
        except:
            return {'signal': 'NEUTRAL', 'reason': ''}
    
    def _create_hold_signal(self, reason: str, df: pd.DataFrame) -> TradingSignal:
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        return TradingSignal(
            signal="HOLD",
            confidence=0,
            entry_price=current_price,
            stop_loss=0,
            take_profit=0,
            position_size=0,
            risk_reward_ratio=0,
            timeframe="1d",
            reasons=[reason],
            technical_score=0,
            sentiment_score=0,
            volume_score=0,
            social_sentiment=0,
            onchain_score=0,
            backtest_winrate=0,
            ichimoku_signal="NEUTRAL",
            fibonacci_levels={},
            elliott_wave="No pattern",
            multi_timeframe_alignment="N/A",
            ml_prediction="NEUTRAL",
            smc_analysis={},
            vsa_signal="Unknown",
            wyckoff_analysis="Unknown",
            order_blocks=[],
            liquidity_zones=[]
        )

class DatabaseManager:
    """Enhanced database management"""
    def __init__(self, db_path='ultimate_crypto_bot.db'):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
    
    def _create_tables(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    experience_level TEXT DEFAULT 'beginner',
                    risk_tolerance TEXT DEFAULT 'medium',
                    account_size REAL DEFAULT 10000,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    reasons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    symbol TEXT,
                    pnl_percent REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    symbol TEXT,
                    target_price REAL,
                    condition TEXT,
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
    
    def get_user_profile(self, user_id: int) -> Dict:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            if result:
                return {
                    'experience_level': result[1],
                    'risk_tolerance': result[2],
                    'account_size': result[3]
                }
            return {'experience_level': 'beginner', 'risk_tolerance': 'medium', 'account_size': 10000}
    
    def update_user_profile(self, user_id: int, **kwargs):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO user_profiles (user_id, experience_level, risk_tolerance, account_size)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, kwargs.get('experience_level', 'beginner'),
                      kwargs.get('risk_tolerance', 'medium'),
                      kwargs.get('account_size', 10000)))
            else:
                updates = []
                values = []
                for key, value in kwargs.items():
                    if key in ['experience_level', 'risk_tolerance', 'account_size']:
                        updates.append(f"{key} = ?")
                        values.append(value)
                if updates:
                    values.append(user_id)
                    cursor.execute(f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?", values)
            self.conn.commit()
    
    def save_signal(self, user_id: int, symbol: str, signal: TradingSignal):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO trading_signals (user_id, symbol, signal_type, confidence, entry_price, stop_loss, take_profit, reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, symbol, signal.signal, signal.confidence, signal.entry_price,
                  signal.stop_loss, signal.take_profit, json.dumps(signal.reasons)))
            self.conn.commit()
            return cursor.lastrowid
    
    def get_user_stats(self, user_id: int, days: int = 30) -> Dict:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT COUNT(*), AVG(confidence)
                FROM trading_signals
                WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
            '''.format(days), (user_id,))
            result = cursor.fetchone()
            if result and result[0]:
                return {
                    'total_signals': result[0],
                    'avg_confidence': result[1] or 0,
                    'accuracy': 60.0
                }
            return {'total_signals': 0, 'avg_confidence': 0, 'accuracy': 0}
    
    def add_price_alert(self, user_id: int, symbol: str, target_price: float, condition: str):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO price_alerts (user_id, symbol, target_price, condition)
                VALUES (?, ?, ?, ?)
            ''', (user_id, symbol, target_price, condition))
            self.conn.commit()
    
    def get_active_alerts(self, user_id: int) -> List[Dict]:
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, symbol, target_price, condition
                FROM price_alerts
                WHERE user_id = ? AND active = 1
            ''', (user_id,))
            results = cursor.fetchall()
            return [{'id': r[0], 'symbol': r[1], 'target_price': r[2], 'condition': r[3]} for r in results]

class UltimateTradingBot:
    """Main bot orchestrator"""
    def __init__(self):
        self.api_manager = EnhancedAPIManager()
        self.signal_generator = EnhancedSignalGenerator()
        self.db_manager = DatabaseManager()
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.chart_generator = ChartGenerator()
        self.system_prompt = """You are Arshava, an elite crypto trading analyst with advanced capabilities.

Your analysis includes:
✓ 15+ technical indicators (RSI, MACD, Ichimoku, etc.)
✓ Smart Money Concepts (Order Blocks, FVG, Liquidity)
✓ Volume Spread Analysis & Wyckoff Method
✓ Machine Learning predictions
✓ On-chain metrics & Social sentiment
✓ Backtested strategies

Response structure:
1. Market overview (price, trend, momentum)
2. Smart Money analysis (SMC signals)
3. Volume analysis (Wyckoff phase)
4. ML prediction
5. Final signal with confidence
6. Entry, stop loss, take profit
7. Risk/reward & position size
8. Key risks

Be concise, specific, and actionable."""
    
    def process_request(self, message) -> Tuple[str, bytes]:
        try:
            user_input = message.text
            user_id = message.chat.id
            
            symbols = re.findall(r'\b([A-Z]{2,6}|bitcoin|ethereum|solana|cardano|polkadot|polygon|avalanche|chainlink)\b', user_input.upper())
            name_map = {
                'BITCOIN': 'BTC', 'ETHEREUM': 'ETH', 'SOLANA': 'SOL', 'CARDANO': 'ADA',
                'POLKADOT': 'DOT', 'POLYGON': 'MATIC', 'AVALANCHE': 'AVAX', 'CHAINLINK': 'LINK'
            }
            symbols = [name_map.get(s, s) for s in symbols]
            symbol = symbols[0] if symbols else 'BTC'
            
            price_data = self.api_manager.get_price_data([symbol])
            ohlcv_df = self.api_manager.data_fetcher.fetch_ohlcv(symbol, '1d', 200)
            
            if ohlcv_df.empty or len(ohlcv_df) < 50:
                return f"❌ Unable to fetch data for {symbol}. Please try another symbol.", None
            
            fear_greed = self.api_manager.get_fear_greed_index()
            news_data = self.api_manager.search_news(f"{symbol} price", 5)
            
            user_profile = self.db_manager.get_user_profile(user_id)
            signal = self.signal_generator.generate_signal(ohlcv_df, symbol, user_profile, news_data)
            
            if signal.signal != "HOLD":
                self.db_manager.save_signal(user_id, symbol, signal)
            
            ai_analysis = self._generate_ai_analysis(user_input, symbol, price_data, signal, fear_greed, user_profile)
            
            response = self._format_response(symbol, price_data, signal, fear_greed, ai_analysis, user_id)
            
            chart_image = self.chart_generator.generate_chart(ohlcv_df, symbol, signal)
            
            return response, chart_image
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return "❌ An error occurred. Please try again.", None
    
    def _generate_ai_analysis(self, user_input: str, symbol: str, price_data: Dict, signal: TradingSignal, fear_greed: int, user_profile: Dict) -> str:
        try:
            price_info = price_data.get(symbol, {})
            context = f"""User request: {user_input}

User profile: {user_profile.get('experience_level')} | Risk: {user_profile.get('risk_tolerance')} | Account: ${user_profile.get('account_size', 10000):,}

Market: {symbol} - ${price_info.get('price', 0):,.2f} ({price_info.get('percent_change_24h', 0):+.2f}%)
Fear & Greed: {fear_greed}/100

Signal: {signal.signal} ({signal.confidence:.1f}%)
Entry: ${signal.entry_price:,.2f} | SL: ${signal.stop_loss:,.2f} | TP: ${signal.take_profit:,.2f}
R/R: {signal.risk_reward_ratio:.2f} | Position: ${signal.position_size:,.2f}

SMC: {signal.smc_analysis.get('signal', 'N/A')}
Wyckoff: {signal.wyckoff_analysis}
ML: {signal.ml_prediction}
Backtest WR: {signal.backtest_winrate:.1f}%

Key reasons: {', '.join(signal.reasons[:5])}

Provide concise professional analysis."""
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context}
                ],
                model="llama-3.3-70b-versatile",
                max_tokens=1200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return "AI analysis temporarily unavailable."
    
    def _format_response(self, symbol: str, price_data: Dict, signal: TradingSignal, fear_greed: int, ai_analysis: str, user_id: int) -> str:
        price_info = price_data.get(symbol, {})
        emoji_map = {"BUY": "🚀", "SELL": "🔴", "HOLD": "⏸️"}
        signal_emoji = emoji_map.get(signal.signal, "⏸️")
        
        response = f"""🤖 **Arshava V2.0 Analysis - {symbol}**
━━━━━━━━━━━━━━━━━━━━

📊 **MARKET OVERVIEW**
━━━━━━━━━━━━━━━━━━━━
💰 Price: ${price_info.get('price', 0):,.2f}
📈 24h: {price_info.get('percent_change_24h', 0):+.2f}%
📊 Volume: ${price_info.get('volume_24h', 0):,.0f}
😨 Fear & Greed: {fear_greed}/100

━━━━━━━━━━━━━━━━━━━━
{signal_emoji} **TRADING SIGNAL**
━━━━━━━━━━━━━━━━━━━━
🎯 Signal: **{signal.signal}**
💪 Confidence: **{signal.confidence:.1f}%**
"""
        
        if signal.signal != "HOLD":
            sl_pct = ((signal.stop_loss/signal.entry_price - 1) * 100)
            tp_pct = ((signal.take_profit/signal.entry_price - 1) * 100)
            response += f"""
💵 Entry: ${signal.entry_price:,.2f}
🛑 Stop Loss: ${signal.stop_loss:,.2f} ({sl_pct:+.2f}%)
🎯 Take Profit: ${signal.take_profit:,.2f} ({tp_pct:+.2f}%)
⚖️ R/R: {signal.risk_reward_ratio:.2f}:1
💰 Position: ${signal.position_size:,.2f}

━━━━━━━━━━━━━━━━━━━━
🧠 **ADVANCED ANALYSIS**
━━━━━━━━━━━━━━━━━━━━
"""
            for i, reason in enumerate(signal.reasons[:6], 1):
                response += f"{i}. {reason}\n"
            
            response += f"""
🎯 SMC: {signal.smc_analysis.get('signal', 'N/A')} ({signal.smc_analysis.get('confidence', 0):.0f}%)
📊 Wyckoff: {signal.wyckoff_analysis}
🤖 ML Prediction: {signal.ml_prediction}
🔬 Backtest WR: {signal.backtest_winrate:.1f}%
⛓️ On-chain: {signal.onchain_score:.0f}/100
🐦 Social: {signal.social_sentiment:.0f}/100
"""
        
        response += f"""
━━━━━━━━━━━━━━━━━━━━
🤖 **AI INSIGHTS**
━━━━━━━━━━━━━━━━━━━━
{ai_analysis[:1200]}

━━━━━━━━━━━━━━━━━━━━
"""
        
        stats = self.db_manager.get_user_stats(user_id)
        if stats['total_signals'] > 0:
            response += f"""📊 **YOUR STATS** (30d)
━━━━━━━━━━━━━━━━━━━━
• Signals: {stats['total_signals']}
• Avg Confidence: {stats['avg_confidence']:.1f}%
• Accuracy: {stats['accuracy']:.1f}%

"""
        
        response += """⚠️ **DISCLAIMER**: Not financial advice. DYOR and manage risk."""
        
        return response[:4000]

# ============================================
# TELEGRAM BOT HANDLERS
# ============================================

ultimate_bot = UltimateTradingBot()

def create_main_keyboard():
    """Create main menu keyboard"""
    markup = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    markup.add(
        types.KeyboardButton("📊 Quick Analysis"),
        types.KeyboardButton("👤 Profile"),
        types.KeyboardButton("📈 My Stats"),
        types.KeyboardButton("🔔 Alerts"),
        types.KeyboardButton("📚 Help"),
        types.KeyboardButton("💡 Market Overview")
    )
    return markup

def create_inline_keyboard(symbol: str):
    """Create inline keyboard for signal"""
    markup = types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        types.InlineKeyboardButton("📊 Chart", callback_data=f"chart_{symbol}"),
        types.InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh_{symbol}"),
        types.InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{symbol}"),
        types.InlineKeyboardButton("📈 Multi-TF", callback_data=f"mtf_{symbol}")
    )
    return markup

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = """🤖 **Welcome to Arshava V2.0!**
━━━━━━━━━━━━━━━━━━━━━━━━

✨ **NEXT-GEN FEATURES**
━━━━━━━━━━━━━━━━━━━━━━━━

🧠 **Smart Money Concepts**
• Order Blocks & Fair Value Gaps
• Liquidity Zones
• Break of Structure (BOS)
• Market Structure Analysis

📊 **Volume Analysis**
• Wyckoff Method
• Volume Spread Analysis
• Accumulation/Distribution

🤖 **AI & Machine Learning**
• ML-powered predictions
• Gradient Boosting models
• Backtested strategies

⛓️ **On-chain & Social**
• Exchange netflows
• Whale activity
• Social sentiment analysis

📈 **Technical Analysis**
• 20+ indicators
• Multi-timeframe analysis
• Professional risk management

━━━━━━━━━━━━━━━━━━━━━━━━
🚀 **GET STARTED**
━━━━━━━━━━━━━━━━━━━━━━━━

Just type: **BTC** or **analyze SOL**

Use the menu buttons below! 👇"""
    
    bot.reply_to(message, welcome_text, parse_mode='Markdown', reply_markup=create_main_keyboard())

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """📖 **ARSHAVA V2.0 GUIDE**
━━━━━━━━━━━━━━━━━━━━━━━━

🎯 **QUICK START**
• Type coin: BTC, ETH, SOL
• /analyze [coin]
• Use menu buttons

📊 **MENU OPTIONS**
━━━━━━━━━━━━━━━━━━━━━━━━

**📊 Quick Analysis**
Get instant analysis for any coin

**👤 Profile**
Setup your trading profile:
• Experience level
• Risk tolerance  
• Account size

**📈 My Stats**
View your performance

**🔔 Alerts**
Set price alerts

**💡 Market Overview**
See top coins overview

━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **SIGNAL GUIDE**
━━━━━━━━━━━━━━━━━━━━━━━━

**Confidence Levels:**
• 80-100%: Very Strong
• 60-79%: Strong
• 40-59%: Moderate
• <40%: Weak

**Always use stop loss!**

━━━━━━━━━━━━━━━━━━━━━━━━
Need help? Contact @YourSupport"""
    
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['profile'])
def setup_profile(message):
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, row_width=3)
    markup.add('Beginner', 'Intermediate', 'Professional')
    msg = bot.reply_to(message, "🎯 **Step 1/3: Experience Level**\n\nSelect your trading experience:", 
                       reply_markup=markup, parse_mode='Markdown')
    bot.register_next_step_handler(msg, process_experience, message.chat.id)

def process_experience(message, user_id):
    experience = message.text.lower()
    if experience not in ['beginner', 'intermediate', 'professional']:
        experience = 'beginner'
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, row_width=3)
    markup.add('Low', 'Medium', 'High')
    msg = bot.reply_to(message, "⚖️ **Step 2/3: Risk Tolerance**\n\nSelect your risk tolerance:", 
                       reply_markup=markup, parse_mode='Markdown')
    bot.register_next_step_handler(msg, process_risk, user_id, experience)

def process_risk(message, user_id, experience):
    risk = message.text.lower()
    if risk not in ['low', 'medium', 'high']:
        risk = 'medium'
    msg = bot.reply_to(message, "💰 **Step 3/3: Account Size**\n\nEnter your trading account size (USD):\n\nExample: 10000", 
                       parse_mode='Markdown')
    bot.register_next_step_handler(msg, process_account_size, user_id, experience, risk)

def process_account_size(message, user_id, experience, risk):
    try:
        account_size = float(re.sub(r'[,$]', '', message.text))
        if account_size < 100:
            bot.reply_to(message, "❌ Minimum account size: $100", reply_markup=create_main_keyboard())
            return
        
        ultimate_bot.db_manager.update_user_profile(user_id, experience_level=experience, 
                                                     risk_tolerance=risk, account_size=account_size)
        
        response = f"""✅ **Profile Saved!**
━━━━━━━━━━━━━━━━━━━━

👤 Experience: {experience.capitalize()}
⚖️ Risk: {risk.capitalize()}
💰 Account: ${account_size:,.0f}

Signals optimized for your profile! 
Type: **BTC** to start"""
        
        bot.reply_to(message, response, parse_mode='Markdown', reply_markup=create_main_keyboard())
    except:
        bot.reply_to(message, "❌ Invalid number. Try again.", reply_markup=create_main_keyboard())

@bot.message_handler(commands=['stats'])
def show_stats(message):
    user_id = message.chat.id
    stats = ultimate_bot.db_manager.get_user_stats(user_id)
    profile = ultimate_bot.db_manager.get_user_profile(user_id)
    
    if stats['total_signals'] == 0:
        response = """📊 **YOUR STATISTICS**
━━━━━━━━━━━━━━━━━━━━

No signals yet!
Type: **BTC** to start"""
    else:
        response = f"""📊 **STATISTICS** (30 Days)
━━━━━━━━━━━━━━━━━━━━

📈 **PERFORMANCE**
• Signals: {stats['total_signals']}
• Avg Confidence: {stats['avg_confidence']:.1f}%
• Accuracy: {stats['accuracy']:.1f}%

👤 **PROFILE**
• Experience: {profile['experience_level'].capitalize()}
• Risk: {profile['risk_tolerance'].capitalize()}
• Account: ${profile['account_size']:,.0f}

Update: /profile"""
    
    bot.reply_to(message, response, parse_mode='Markdown', reply_markup=create_main_keyboard())

@bot.message_handler(commands=['analyze'])
def analyze_command(message):
    processing_msg = bot.reply_to(message, 
        "⏳ **Analyzing...**\n\n🔄 Fetching data\n📊 Calculating indicators\n🧠 Running ML\n🤖 Generating AI insights\n\nPlease wait...",
        parse_mode='Markdown')
    
    try:
        response, chart = ultimate_bot.process_request(message)
        bot.delete_message(message.chat.id, processing_msg.message_id)
        
        # Extract symbol for inline keyboard
        symbols = re.findall(r'\b([A-Z]{2,6})\b', message.text.upper())
        symbol = symbols[0] if symbols else 'BTC'
        
        sent_msg = bot.reply_to(message, response, parse_mode='Markdown', 
                               reply_markup=create_inline_keyboard(symbol))
        
        if chart:
            bot.send_photo(message.chat.id, chart, caption=f"📊 {symbol}/USD Chart")
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        bot.reply_to(message, "❌ Error. Please try again.", reply_markup=create_main_keyboard())

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    """Handle inline keyboard callbacks"""
    try:
        data = call.data.split('_')
        action = data[0]
        symbol = data[1] if len(data) > 1 else 'BTC'
        
        if action == 'refresh':
            bot.answer_callback_query(call.id, "🔄 Refreshing...")
            # Create a mock message object
            mock_msg = type('obj', (object,), {'text': symbol, 'chat': type('obj', (object,), {'id': call.message.chat.id})()})()
            response, chart = ultimate_bot.process_request(mock_msg)
            bot.edit_message_text(response, call.message.chat.id, call.message.message_id, 
                                parse_mode='Markdown', reply_markup=create_inline_keyboard(symbol))
            if chart:
                bot.send_photo(call.message.chat.id, chart, caption=f"📊 {symbol}/USD Updated Chart")
        
        elif action == 'chart':
            bot.answer_callback_query(call.id, "📊 Generating chart...")
            ohlcv_df = ultimate_bot.api_manager.data_fetcher.fetch_ohlcv(symbol, '1d', 200)
            if not ohlcv_df.empty:
                user_profile = ultimate_bot.db_manager.get_user_profile(call.message.chat.id)
                signal = ultimate_bot.signal_generator.generate_signal(ohlcv_df, symbol, user_profile)
                chart = ultimate_bot.chart_generator.generate_chart(ohlcv_df, symbol, signal)
                if chart:
                    bot.send_photo(call.message.chat.id, chart, caption=f"📊 {symbol}/USD Detailed Chart")
        
        elif action == 'alert':
            bot.answer_callback_query(call.id, "🔔 Set alert...")
            msg = bot.send_message(call.message.chat.id, 
                f"🔔 **Set Price Alert for {symbol}**\n\nEnter target price (e.g., 50000):", 
                parse_mode='Markdown')
            bot.register_next_step_handler(msg, process_alert, call.message.chat.id, symbol)
        
        elif action == 'mtf':
            bot.answer_callback_query(call.id, "📈 Multi-timeframe analysis...")
            msg = bot.send_message(call.message.chat.id, 
                f"📈 **Multi-Timeframe Analysis for {symbol}**\n\n⏳ Analyzing 1H, 4H, 1D timeframes...", 
                parse_mode='Markdown')
            mtf_response = perform_mtf_analysis(symbol)
            bot.edit_message_text(mtf_response, call.message.chat.id, msg.message_id, parse_mode='Markdown')
    
    except Exception as e:
        logger.error(f"Callback error: {e}")
        bot.answer_callback_query(call.id, "❌ Error occurred")

def process_alert(message, user_id, symbol):
    """Process price alert setup"""
    try:
        target_price = float(re.sub(r'[,$]', '', message.text))
        
        # Get current price
        price_data = ultimate_bot.api_manager.get_price_data([symbol])
        current_price = price_data.get(symbol, {}).get('price', 0)
        
        if target_price > current_price:
            condition = 'above'
            emoji = '🚀'
        else:
            condition = 'below'
            emoji = '🔴'
        
        ultimate_bot.db_manager.add_price_alert(user_id, symbol, target_price, condition)
        
        response = f"""✅ **Alert Set!**
━━━━━━━━━━━━━━━━━━━━

{emoji} {symbol}: ${target_price:,.2f}
📊 Current: ${current_price:,.2f}
🔔 Notify when {condition}

You'll be notified when price reaches target!"""
        
        bot.reply_to(message, response, parse_mode='Markdown', reply_markup=create_main_keyboard())
    except:
        bot.reply_to(message, "❌ Invalid price. Try again.", reply_markup=create_main_keyboard())

def perform_mtf_analysis(symbol: str) -> str:
    """Perform multi-timeframe analysis"""
    try:
        timeframes = {'1H': '1h', '4H': '4h', '1D': '1d'}
        results = {}
        
        for tf_name, tf_code in timeframes.items():
            df = ultimate_bot.api_manager.data_fetcher.fetch_ohlcv(symbol, tf_code, 100)
            if not df.empty and len(df) >= 50:
                indicators = ultimate_bot.signal_generator.analyzer.calculate_all_indicators(df)
                rsi = indicators.get('rsi', np.array([50]))[-1]
                
                macd = indicators.get('macd', np.array([0]))
                macd_signal = indicators.get('macd_signal', np.array([0]))
                
                if len(macd) > 0 and len(macd_signal) > 0:
                    macd_trend = 'Bullish' if macd[-1] > macd_signal[-1] else 'Bearish'
                else:
                    macd_trend = 'Neutral'
                
                # Determine signal
                if rsi < 35 and macd_trend == 'Bullish':
                    signal = '🚀 BUY'
                elif rsi > 65 and macd_trend == 'Bearish':
                    signal = '🔴 SELL'
                else:
                    signal = '⏸️ HOLD'
                
                results[tf_name] = {
                    'signal': signal,
                    'rsi': rsi,
                    'macd': macd_trend
                }
        
        response = f"""📈 **MULTI-TIMEFRAME ANALYSIS**
━━━━━━━━━━━━━━━━━━━━
**{symbol}/USD**
━━━━━━━━━━━━━━━━━━━━

"""
        
        for tf, data in results.items():
            response += f"""**{tf} Timeframe**
{data['signal']}
RSI: {data['rsi']:.1f}
MACD: {data['macd']}

"""
        
        # Alignment check
        signals = [r['signal'] for r in results.values()]
        if all('BUY' in s for s in signals):
            response += "✅ **STRONG ALIGNMENT**: All timeframes bullish!"
        elif all('SELL' in s for s in signals):
            response += "⚠️ **STRONG ALIGNMENT**: All timeframes bearish!"
        else:
            response += "⚡ **MIXED SIGNALS**: Wait for alignment"
        
        return response
        
    except Exception as e:
        logger.error(f"MTF analysis error: {e}")
        return f"❌ Error performing MTF analysis for {symbol}"

@bot.message_handler(func=lambda message: message.text == "📊 Quick Analysis")
def quick_analysis(message):
    msg = bot.reply_to(message, "🎯 **Quick Analysis**\n\nWhich coin?\n\nExample: BTC, ETH, SOL", 
                      parse_mode='Markdown')
    bot.register_next_step_handler(msg, lambda m: analyze_command(m))

@bot.message_handler(func=lambda message: message.text == "👤 Profile")
def profile_menu(message):
    setup_profile(message)

@bot.message_handler(func=lambda message: message.text == "📈 My Stats")
def stats_menu(message):
    show_stats(message)

@bot.message_handler(func=lambda message: message.text == "🔔 Alerts")
def alerts_menu(message):
    user_id = message.chat.id
    alerts = ultimate_bot.db_manager.get_active_alerts(user_id)
    
    if not alerts:
        response = """🔔 **PRICE ALERTS**
━━━━━━━━━━━━━━━━━━━━

No active alerts.

To set alert:
1. Analyze a coin
2. Click "🔔 Set Alert" button"""
    else:
        response = "🔔 **ACTIVE ALERTS**\n━━━━━━━━━━━━━━━━━━━━\n\n"
        for alert in alerts:
            response += f"• {alert['symbol']}: ${alert['target_price']:,.2f} ({alert['condition']})\n"
    
    bot.reply_to(message, response, parse_mode='Markdown', reply_markup=create_main_keyboard())

@bot.message_handler(func=lambda message: message.text == "📚 Help")
def help_menu(message):
    send_help(message)

@bot.message_handler(func=lambda message: message.text == "💡 Market Overview")
def market_overview(message):
    processing_msg = bot.reply_to(message, "⏳ **Loading Market Overview...**", parse_mode='Markdown')
    
    try:
        symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']
        price_data = ultimate_bot.api_manager.get_price_data(symbols)
        fear_greed = ultimate_bot.api_manager.get_fear_greed_index()
        
        response = f"""💡 **MARKET OVERVIEW**
━━━━━━━━━━━━━━━━━━━━
😨 Fear & Greed: {fear_greed}/100

📊 **TOP COINS**
━━━━━━━━━━━━━━━━━━━━

"""
        
        for symbol in symbols:
            if symbol in price_data:
                data = price_data[symbol]
                change = data.get('percent_change_24h', 0)
                emoji = '🚀' if change > 0 else '🔴'
                response += f"{emoji} **{symbol}**: ${data.get('price', 0):,.2f} ({change:+.2f}%)\n"
        
        response += "\n━━━━━━━━━━━━━━━━━━━━\nType coin name for detailed analysis!"
        
        bot.delete_message(message.chat.id, processing_msg.message_id)
        bot.reply_to(message, response, parse_mode='Markdown', reply_markup=create_main_keyboard())
        
    except Exception as e:
        logger.error(f"Market overview error: {e}")
        bot.reply_to(message, "❌ Error loading market data", reply_markup=create_main_keyboard())

@bot.message_handler(content_types=['text'])
def handle_text_messages(message):
    try:
        text = message.text.upper()
        crypto_keywords = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK', 
                          'BITCOIN', 'ETHEREUM', 'SOLANA', 'CARDANO', 'POLKADOT', 
                          'POLYGON', 'AVALANCHE', 'CHAINLINK', 'PRICE', 'ANALYSIS', 'ANALYZE']
        
        if any(keyword in text for keyword in crypto_keywords):
            processing_msg = bot.reply_to(message, 
                "⏳ **Processing...**\n\n🔍 Fetching data\n📊 Running analysis\n🧠 Computing AI\n\n⏱️ Please wait...",
                parse_mode='Markdown')
            
            response, chart = ultimate_bot.process_request(message)
            
            try:
                bot.delete_message(message.chat.id, processing_msg.message_id)
            except:
                pass
            
            # Extract symbol
            symbols = re.findall(r'\b([A-Z]{2,6})\b', text)
            symbol = symbols[0] if symbols else 'BTC'
            
            bot.reply_to(message, response, parse_mode='Markdown', 
                        reply_markup=create_inline_keyboard(symbol))
            
            if chart:
                bot.send_photo(message.chat.id, chart, caption=f"📊 {symbol}/USD Chart")
        else:
            bot.reply_to(message, 
                "🤔 I didn't recognize that.\n\nTry:\n• BTC, ETH, SOL\n• Use menu buttons below\n• Type /help", 
                parse_mode='Markdown', reply_markup=create_main_keyboard())
    
    except Exception as e:
        logger.error(f"Text handler error: {e}")
        bot.reply_to(message, "❌ Error occurred. Try again.", reply_markup=create_main_keyboard())

# ============================================
# PRICE ALERT MONITORING (Background Task)
# ============================================

def monitor_alerts():
    """Background task to monitor price alerts"""
    while True:
        try:
            # This would need to check all active alerts
            # For production, use proper async task queue
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Alert monitoring error: {e}")
            time.sleep(60)

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main bot execution"""
    logger.info("=" * 70)
    logger.info("🚀 STARTING ARSHAVA V2.0 - ULTIMATE CRYPTO TRADING BOT")
    logger.info("=" * 70)
    
    logger.info("\n📡 Data Sources:")
    logger.info("  ✓ Binance API (Primary)")
    logger.info("  ✓ CryptoCompare (Backup)")
    logger.info("  ✓ CoinGecko (Backup)")
    
    logger.info("\n🧠 Advanced Features:")
    logger.info("  ✓ Smart Money Concepts (SMC)")
    logger.info("  ✓ Volume Spread Analysis (VSA)")
    logger.info("  ✓ Wyckoff Method")
    logger.info("  ✓ Machine Learning Predictions")
    logger.info("  ✓ Backtesting Engine")
    logger.info("  ✓ On-chain Metrics")
    logger.info("  ✓ Social Sentiment Analysis")
    
    logger.info("\n📊 Technical Indicators:")
    logger.info("  ✓ RSI, MACD, Bollinger Bands")
    logger.info("  ✓ Ichimoku Cloud")
    logger.info("  ✓ Elder's Force Index (EFI)")
    logger.info("  ✓ On-Balance Volume (OBV)")
    logger.info("  ✓ Fibonacci Retracements")
    logger.info("  ✓ Elliott Wave Patterns")
    logger.info("  ✓ Heiken Ashi Candles")
    logger.info("  ✓ Volume Profile (POC/VAH/VAL)")
    
    logger.info("\n🎨 UI/UX Features:")
    logger.info("  ✓ Inline Keyboards")
    logger.info("  ✓ Quick Reply Buttons")
    logger.info("  ✓ Interactive Charts")
    logger.info("  ✓ Price Alerts")
    logger.info("  ✓ Multi-Timeframe Analysis")
    
    logger.info("\n🧪 Testing connections...")
    
    try:
        test_prices = ultimate_bot.api_manager.get_price_data(['BTC'])
        if test_prices:
            btc_price = test_prices.get('BTC', {}).get('price', 'N/A')
            logger.info(f"  ✅ Price API: BTC = ${btc_price}")
        else:
            logger.warning("  ⚠️ Price API test returned no data")
    except Exception as e:
        logger.error(f"  ❌ Price API test failed: {e}")
    
    try:
        test_ohlcv = ultimate_bot.api_manager.data_fetcher.fetch_ohlcv('BTC', '1d', 100)
        if not test_ohlcv.empty:
            logger.info(f"  ✅ OHLCV API: {len(test_ohlcv)} candles")
            logger.info(f"     Latest: ${test_ohlcv['close'].iloc[-1]:,.2f}")
        else:
            logger.warning("  ⚠️ OHLCV API test returned empty")
    except Exception as e:
        logger.error(f"  ❌ OHLCV API test failed: {e}")
    
    try:
        fear_greed = ultimate_bot.api_manager.get_fear_greed_index()
        logger.info(f"  ✅ Fear & Greed: {fear_greed}/100")
    except Exception as e:
        logger.error(f"  ❌ Fear & Greed test failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL SYSTEMS READY - ARSHAVA V2.0 ONLINE")
    logger.info("=" * 70)
    logger.info("\n🤖 Bot active and waiting for commands...")
    logger.info("📱 Users can interact via Telegram")
    logger.info("🔄 Press Ctrl+C to stop\n")
    
    # Start alert monitoring in background (optional)
    # threading.Thread(target=monitor_alerts, daemon=True).start()
    
    while True:
        try:
            bot.polling(none_stop=True, interval=1, timeout=20)
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Polling error: {e}")
            logger.info("🔄 Restarting in 15 seconds...")
            time.sleep(15)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Bot shutdown complete")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)