# PUMP Hyperliquid RSI Backtester

## Default strategy

- Market: Hyperliquid perpetual `PUMP`
- Timeframe: 15 minutes
- RSI: Wilder RSI, period 14
- Long: RSI crosses upward through 30
- Short: RSI crosses downward through 70
- Both directions: the opposite signal closes the current position and reverses it
- Initial capital: 1,000 USDT
- Position sizing: 100% of realized equity, 1x notional (no leverage)
- Default taker fee assumption: 0.035% per side
- Default slippage assumption: 0.05% per side
- Signals use completed candles; orders fill at the next candle open

## Install and run

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python pump_rsi_backtest.py
```

## Outputs

The script creates `pump_rsi_results/` containing:

- `candles_with_rsi.csv` — downloaded OHLCV data and RSI/signals
- `trade_history.csv` — every completed long/short trade, prices, fees and P&L
- `equity_curve.csv` — candle-by-candle marked equity
- `performance_report.json` — summary metrics and backtest settings

## Useful options

```bash
# Change the fee assumption to 0.05% per side
python pump_rsi_backtest.py --fee-rate 0.0005

# Test 1-hour candles
python pump_rsi_backtest.py --interval 1h

# Test different RSI thresholds
python pump_rsi_backtest.py --oversold 25 --overbought 75

# Use 50% of equity per position
python pump_rsi_backtest.py --position-pct 0.5

# Change output directory
python pump_rsi_backtest.py --out results_30_70
```

## Important limitations

Hyperliquid's candle snapshot API documents a maximum of the most recent 5,000 candles per market/interval. For 15-minute candles this is about 52 days, not the entire market history. The script downloads the available recent window in smaller requests and cannot recover older candles that the endpoint does not expose.

This is a candle-based approximation. It includes configurable trading fees and slippage but does not include funding payments, liquidation, order-book depth, spread changes, partial fills, latency or market impact. The default fee is an assumption; replace it with the taker fee for your own Hyperliquid account tier. No real orders are placed.
