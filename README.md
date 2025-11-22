# Ultimate Crypto Trading Bot

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=flat&logo=python)  
![License](https://img.shields.io/badge/License-Apache_2.0-green?style=flat)  
![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-blue?style=flat&logo=telegram)  
![Stars](https://img.shields.io/github/stars/Arash-Mansourpour/ultimate-crypto-trading-bot?style=flat)  

A sophisticated Telegram bot designed for in-depth cryptocurrency analysis and trading signals. Powered by advanced AI, multi-source data integration, and comprehensive technical tools, this bot empowers traders with actionable insights, risk management, and real-time market intelligence.

Demo 1

🎥 Watch crypto1.mp4

Demo 2

🎥 Watch crypto2.mp4

## Key Features

- **Multi-Source Data Fetching**: Seamlessly retrieves OHLCV data from multiple reliable sources including Binance, Yahoo Finance, CryptoCompare, CoinGecko, Coinbase, and Kraken, with automatic fallback and data imputation for robustness.
- **Advanced Technical Analysis**: Utilizes over 15 indicators such as Heiken Ashi for trend detection, Ichimoku Cloud, Fibonacci levels, Elliott Wave patterns, OBV, EFI, Bollinger Bands, RSI, MACD, and more.
- **On-Chain & Social Metrics**: Integrates real-time on-chain data (e.g., active addresses, transaction counts) from Glassnode and Santiment, alongside social sentiment analysis from Twitter (X) and Reddit using VADER.
- **AI-Powered Signals**: Generates buy/sell/hold signals with confidence scores, entry/exit points, stop-loss, take-profit, position sizing, and risk-reward ratios using machine learning (Random Forest) and backtesting (Monte Carlo simulations).
- **Volume & Order Flow Insights**: Includes Volume Profile (POC, VAH, VAL), Order Flow (Delta Volume, CVD, VWAP), Market Microstructure (order book depth, spread), and volatility models (Historical, GARCH).
- **Pattern Recognition**: Detects harmonic patterns (Gartley, Butterfly), chart patterns (Head & Shoulders), and candlestick formations for enhanced predictive accuracy.
- **Multi-Timeframe & Correlation Analysis**: Supports MTF alignment across 1h/4h/1d/1w and BTC correlation for holistic market views.
- **User Customization**: Profiles for experience level, risk tolerance, and account size to tailor signals and risk management.
- **Secure & Extensible**: Environment variable support for API keys, logging, and modular architecture for easy contributions.

This bot is ideal for traders seeking an edge in volatile crypto markets, blending quantitative analysis with qualitative sentiment for informed decisions.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/Arash-Mansourpour/ultimate-crypto-trading-bot.git
   cd ultimate-crypto-trading-bot
   ```

2. **Install Dependencies**:
   Ensure Python 3.12+ is installed. Create a virtual environment and install required packages:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *Note*: Libraries include `telebot`, `pandas`, `numpy`, `talib`, `groq`, `requests`, `scikit-learn`, `tweepy`, `praw`, and more. See `requirements.txt` for the full list.

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```
   TELEGRAM_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
   GOOGLE_API_KEY=your_google_api_key
   COINMARKETCAP_API_KEY=your_cmc_api_key
   # Add other keys as needed (e.g., Twitter Bearer Token, Reddit Client ID)
   ```
   *Security Tip*: Never commit `.env` to version control—it's already ignored in `.gitignore`.

4. **Run the Bot**:
   ```
   python bot.py
   ```
   The bot will start polling for Telegram messages. Search for your bot in Telegram and use `/start` to begin.

## Usage

- **Commands**:
  - `/start`: Welcome message and bot overview.
  - `/analyze [coin]`: Comprehensive analysis for a cryptocurrency (e.g., `/analyze BTC`).
  - `/profile`: Set your trading profile (experience, risk tolerance, account size).
  - `/stats`: View your signal performance stats.
  - `/help`: Detailed help guide.

- **Interactive Mode**: Simply send a coin name (e.g., "ETH") for instant analysis, including signals, charts insights, and sentiment.

Example Output:
```
🤖 Arshava - Comprehensive BTC Analysis

💰 Price: $65,000.00
📈 24h Change: +2.5%
😨 Fear & Greed: 75/100

🚀 Signal: BUY (Confidence: 85%)
💵 Entry: $65,000.00
🛑 Stop Loss: $63,000.00
🎯 Take Profit: $70,000.00
⚖️ Risk/Reward: 2.5
```

## Requirements

- **Python**: 3.12 or higher.
- **Dependencies**: Listed in `requirements.txt`. Key ones include:
  - Data: `pandas`, `numpy`, `talib`, `scipy`.
  - APIs: `requests`, `groq`, `tweepy`, `praw`.
  - ML: `scikit-learn`, `arch`.
  - Others: `telebot`, `textblob`, `vaderSentiment`.

No internet-dependent installations during runtime—all handled via `pip`.

## Contributing

We welcome contributions! To get started:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit changes: `git commit -m 'Add YourFeature'`.
4. Push: `git push origin feature/YourFeature`.
5. Open a Pull Request.

Please follow code style guidelines (PEP8) and include tests where applicable. For major changes, open an issue first.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by open-source crypto tools and AI advancements.
- Thanks to contributors of libraries like TA-Lib, Groq, and Telebot.
- Built with ❤️ by [Arash Mansourpour](https://github.com/Arash-Mansourpour).

For questions or support, open an issue or reach out via Telegram.

---

*Last Updated: October 14, 2025*
