# Setup

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Set up Alpaca API credentials
Copy `.env.example` to `.env` and fill in your Alpaca API key and secret.
A free account at alpaca.markets is sufficient for historical US equity data.

```bash
cp .env.example .env
# edit .env with your keys
```

## 3. Fetch data
```bash
python scripts/fetch_data.py                              # AAPL 1d, 2023 onwards
python scripts/fetch_data.py TSLA 1h 2023-01-01
python scripts/fetch_data.py NVDA 1d 2020-01-01 2024-01-01

# via CLI
python -m bot fetch --symbol AAPL --timeframe 1d --since 2023-01-01
```

## 4. Run a backtest
```bash
# Compare all strategies
python scripts/run_backtest.py

# Single strategy with HTML report
python scripts/run_backtest.py sma_crossover AAPL 1d 2023-01-01 2024-01-01 --report

# Via CLI
python -m bot backtest --symbol AAPL --timeframe 1d --strategy macd --since 2023-01-01 --report
python -m bot compare  --symbol AAPL --timeframe 1d --since 2023-01-01 --report
```

## Project structure
```
stock-bot/
├── bot/
│   ├── config.py               SimConfig, timeframe constants (252 trading days/yr)
│   ├── data/
│   │   ├── fetcher.py          Alpaca StockHistoricalDataClient, OHLCV fetch
│   │   └── store.py            DuckDB persistence layer
│   ├── strategy/
│   │   ├── base.py             BaseStrategy ABC
│   │   └── technical.py        SMA/EMA crossover, EMA+RSI, Bollinger, MACD
│   └── backtest/
│       ├── engine.py           BacktestEngine (pure Python)
│       └── report.py           Metrics, Rich tables, Plotly HTML charts
└── scripts/
    ├── fetch_data.py
    └── run_backtest.py
```

## Supported timeframes
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1M`

## Notes
- Alpaca free tier uses the IEX feed (~35% of market volume). For full SIP data, a paid subscription is needed.
- Data is split- and dividend-adjusted by default (`adjustment="all"`).
- The backtest assumes long-only, fully invested positions. Commission is 0 (Alpaca default).
