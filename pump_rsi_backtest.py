#!/usr/bin/env python3
"""Backtest a two-way RSI(14) strategy on Hyperliquid PUMP perpetual candles.

Default rules (signals are calculated on a closed candle and filled at the next
candle's open):
  * RSI crosses UP through 30: enter/flip LONG.
  * RSI crosses DOWN through 70: enter/flip SHORT.

A position is closed/reversed on the opposite signal. Any open position is
closed at the final candle close. Position sizing is 100% of current realized
equity (1x notional; no leverage). Fees and slippage are charged on both sides.

This is a candle-based simulation, not an execution simulator. It excludes
funding payments, liquidation, spread variation, partial fills and market impact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

INFO_URL = "https://api.hyperliquid.xyz/info"
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000,
    "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000,
    "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}


def utc_string(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_candles(coin: str, interval: str, bars: int, pause: float = 0.12) -> pd.DataFrame:
    """Download up to `bars` most recent candles in chunks to avoid oversized calls.

    Hyperliquid documents candleSnapshot as exposing only the most recent 5,000
    candles per market/interval. This function cannot retrieve older candles
    outside that exchange-side retention window.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    if bars < 100:
        raise ValueError("Please request at least 100 candles for RSI warm-up and testing.")
    bars = min(bars, 5000)
    step = INTERVAL_MS[interval]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - bars * step
    chunk_bars = 400
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json", "User-Agent": "PUMP-RSI-Backtester/1.0"})
    rows = []
    cursor = start_ms
    request_count = 0

    while cursor < end_ms:
        chunk_end = min(end_ms, cursor + chunk_bars * step - 1)
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval, "startTime": cursor, "endTime": chunk_end},
        }
        last_error = None
        for attempt in range(4):
            try:
                response = session.post(INFO_URL, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"Unexpected API response: {data}")
                rows.extend(data)
                last_error = None
                break
            except Exception as exc:  # retry transient HTTP/network/API errors
                last_error = exc
                time.sleep(1.0 * (attempt + 1))
        if last_error:
            raise RuntimeError(f"Hyperliquid API request failed: {last_error}")
        request_count += 1
        cursor = chunk_end + 1
        if pause:
            time.sleep(pause)

    if not rows:
        raise RuntimeError(
            f"No candles returned for {coin}. Check that this is the correct perpetual coin symbol "
            "(not necessarily the spot pair label shown in the UI)."
        )

    df = pd.DataFrame(rows)
    required = {"t", "T", "o", "h", "l", "c", "v", "n"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Candle response missing fields: {required - set(df.columns)}")
    df = df.rename(columns={"t": "timestamp_ms", "T": "close_timestamp_ms", "o": "open", "h": "high",
                            "l": "low", "c": "close", "v": "volume", "n": "trade_count"})
    for col in ["timestamp_ms", "close_timestamp_ms", "open", "high", "low", "close", "volume", "trade_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", "open", "high", "low", "close"])
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)

    # Exclude the currently-forming candle to avoid using an unfinished close.
    now_ms = int(time.time() * 1000)
    df = df[df["close_timestamp_ms"] <= now_ms].copy()
    df = df.tail(bars).reset_index(drop=True)
    if len(df) < 100:
        raise RuntimeError(f"Only {len(df)} closed candles received; need at least 100. Verify coin/market availability.")
    df["datetime_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    print(f"Downloaded {len(df):,} closed {interval} candles for {coin} in {request_count} API calls.")
    print(f"Range: {utc_string(int(df.timestamp_ms.iloc[0]))} -> {utc_string(int(df.timestamp_ms.iloc[-1]))}")
    return df


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI using an SMA seed followed by Wilder's recursive smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rsi = pd.Series(index=close.index, dtype=float)
    if len(close) <= period:
        return rsi
    ag = float(avg_gain.iloc[period])
    al = float(avg_loss.iloc[period])
    rsi.iloc[period] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 and ag == 0 else 100 - 100 / (1 + ag / al))
    for i in range(period + 1, len(close)):
        ag = (ag * (period - 1) + float(gain.iloc[i])) / period
        al = (al * (period - 1) + float(loss.iloc[i])) / period
        if al == 0:
            rsi.iloc[i] = 100.0 if ag > 0 else 50.0
        else:
            rsi.iloc[i] = 100 - 100 / (1 + ag / al)
    return rsi


@dataclass
class Position:
    side: str                 # LONG or SHORT
    entry_time: str
    entry_signal_time: str
    entry_price_raw: float
    entry_price_exec: float
    qty: float
    entry_notional: float
    entry_fee: float
    entry_rsi: float
    entry_equity_before_fee: float


def run_backtest(df: pd.DataFrame, initial_capital: float, fee_rate: float,
                 slippage_rate: float, position_pct: float, rsi_period: int,
                 oversold: float, overbought: float):
    df = df.copy()
    df["rsi"] = wilder_rsi(df["close"], rsi_period)
    df["signal"] = ""
    df.loc[(df.rsi.shift(1) <= oversold) & (df.rsi > oversold), "signal"] = "LONG"
    df.loc[(df.rsi.shift(1) >= overbought) & (df.rsi < overbought), "signal"] = "SHORT"

    cash = float(initial_capital)
    position: Optional[Position] = None
    trades = []
    equity_rows = []

    def exec_price(raw: float, side: str, is_entry: bool) -> float:
        # Buy orders pay upward slippage; sell orders receive downward slippage.
        is_buy = (side == "LONG" and is_entry) or (side == "SHORT" and not is_entry)
        return raw * (1 + slippage_rate) if is_buy else raw * (1 - slippage_rate)

    def close_position(raw_price: float, timestamp: str, reason: str, rsi_value: float):
        nonlocal cash, position
        if position is None:
            return
        p = position
        exit_exec = exec_price(raw_price, p.side, is_entry=False)
        direction = 1.0 if p.side == "LONG" else -1.0
        gross = direction * p.qty * (exit_exec - p.entry_price_exec)
        exit_notional = p.qty * exit_exec
        exit_fee = exit_notional * fee_rate
        cash += gross - exit_fee
        net = gross - p.entry_fee - exit_fee
        trades.append({
            "side": p.side,
            "entry_time_utc": p.entry_time,
            "exit_time_utc": timestamp,
            "entry_signal_time_utc": p.entry_signal_time,
            "exit_reason": reason,
            "entry_rsi": p.entry_rsi,
            "exit_rsi": rsi_value,
            "entry_price_raw": p.entry_price_raw,
            "entry_price_exec": p.entry_price_exec,
            "exit_price_raw": raw_price,
            "exit_price_exec": exit_exec,
            "quantity": p.qty,
            "entry_notional_usdt": p.entry_notional,
            "exit_notional_usdt": exit_notional,
            "gross_pnl_usdt": gross,
            "entry_fee_usdt": p.entry_fee,
            "exit_fee_usdt": exit_fee,
            "total_fees_usdt": p.entry_fee + exit_fee,
            "net_pnl_usdt": net,
            "return_on_entry_notional_pct": (net / p.entry_notional * 100) if p.entry_notional else 0.0,
            "equity_after_trade_usdt": cash,
        })
        position = None

    def open_position(side: str, raw_price: float, timestamp: str, signal_time: str, rsi_value: float):
        nonlocal cash, position
        entry_exec = exec_price(raw_price, side, is_entry=True)
        notional = max(0.0, cash) * position_pct
        qty = notional / entry_exec if entry_exec > 0 else 0.0
        fee = notional * fee_rate
        cash -= fee
        position = Position(side, timestamp, signal_time, raw_price, entry_exec, qty,
                            notional, fee, rsi_value, cash + fee)

    # A signal is known only after candle i-1 closes; fill at candle i open.
    # Include a starting equity point so drawdown is measured from the initial capital.
    equity_rows.append({
        "datetime_utc": df.at[1, "datetime_utc"].isoformat(),
        "close": float(df.at[1, "close"]),
        "rsi": float(df.at[1, "rsi"]) if pd.notna(df.at[1, "rsi"]) else math.nan,
        "cash_after_fees_usdt": cash,
        "unrealized_pnl_usdt": 0.0,
        "equity_usdt": cash,
        "position": "FLAT",
    })

    for i in range(2, len(df)):
        signal = df.at[i - 1, "signal"]
        open_raw = float(df.at[i, "open"])
        bar_time = df.at[i, "datetime_utc"].isoformat()
        signal_time = df.at[i - 1, "datetime_utc"].isoformat()
        signal_rsi = float(df.at[i - 1, "rsi"])

        if signal in ("LONG", "SHORT"):
            if position is not None and position.side != signal:
                close_position(open_raw, bar_time, f"REVERSE_TO_{signal}", signal_rsi)
            if position is None:
                open_position(signal, open_raw, bar_time, signal_time, signal_rsi)

        close_raw = float(df.at[i, "close"])
        unrealized = 0.0
        if position is not None:
            direction = 1.0 if position.side == "LONG" else -1.0
            unrealized = direction * position.qty * (close_raw - position.entry_price_exec)
        equity_rows.append({
            "datetime_utc": bar_time,
            "close": close_raw,
            "rsi": float(df.at[i, "rsi"]) if pd.notna(df.at[i, "rsi"]) else math.nan,
            "cash_after_fees_usdt": cash,
            "unrealized_pnl_usdt": unrealized,
            "equity_usdt": cash + unrealized,
            "position": position.side if position else "FLAT",
        })

    # Close any remaining position at the last available close.
    if position is not None:
        last = df.iloc[-1]
        close_position(float(last["close"]), last["datetime_utc"].isoformat(), "END_OF_DATA",
                       float(last["rsi"]) if pd.notna(last["rsi"]) else math.nan)
        if equity_rows:
            equity_rows[-1]["cash_after_fees_usdt"] = cash
            equity_rows[-1]["unrealized_pnl_usdt"] = 0.0
            equity_rows[-1]["equity_usdt"] = cash
            equity_rows[-1]["position"] = "FLAT"

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return df, trades_df, equity_df, cash


def performance_report(trades: pd.DataFrame, equity: pd.DataFrame, initial: float, final: float) -> dict:
    if trades.empty:
        wins = losses = 0
        gross_profit = gross_loss = total_fees = 0.0
        win_rate = profit_factor = avg_trade = 0.0
    else:
        pnl = trades["net_pnl_usdt"]
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        total_fees = float(trades["total_fees_usdt"].sum())
        win_rate = wins / len(trades) * 100
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else ("Infinity" if gross_profit > 0 else 0.0)
        avg_trade = float(pnl.mean())
    if equity.empty:
        max_dd = 0.0
    else:
        curve = equity["equity_usdt"].astype(float)
        peak = curve.cummax()
        dd = (curve / peak - 1.0) * 100
        max_dd = float(dd.min())
    return {
        "initial_capital_usdt": initial,
        "final_equity_usdt": final,
        "net_profit_usdt": final - initial,
        "return_pct": (final / initial - 1) * 100 if initial else 0.0,
        "closed_trades": int(len(trades)),
        "winning_trades": wins,
        "losing_trades": losses,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "gross_winning_trades_usdt": gross_profit,
        "gross_losing_trades_usdt": gross_loss,
        "total_fees_usdt": total_fees,
        "average_net_trade_usdt": avg_trade,
        "max_drawdown_pct": max_dd,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest a two-way RSI strategy using Hyperliquid PUMP candles.")
    parser.add_argument("--coin", default="PUMP", help="Hyperliquid perpetual coin symbol (default: PUMP)")
    parser.add_argument("--interval", default="15m", choices=list(INTERVAL_MS), help="Candle interval (default: 15m)")
    parser.add_argument("--bars", type=int, default=5000, help="Number of recent candles, max 5000 (default: 5000)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital in USDT (default: 1000)")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--oversold", type=float, default=30.0, help="Long trigger: RSI crosses upward through this level")
    parser.add_argument("--overbought", type=float, default=70.0, help="Short trigger: RSI crosses downward through this level")
    parser.add_argument("--fee-rate", type=float, default=0.00035, help="Taker fee per side as decimal; 0.00035 = 0.035%%")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage per side as decimal; 0.0005 = 0.05%%")
    parser.add_argument("--position-pct", type=float, default=1.0, help="Fraction of equity used as notional (default 1.0 = 100%%)")
    parser.add_argument("--out", default="pump_rsi_results", help="Output directory")
    args = parser.parse_args()

    if args.capital <= 0 or args.rsi_period < 2:
        parser.error("capital must be positive and RSI period must be at least 2")
    if not (0 <= args.fee_rate < 0.1 and 0 <= args.slippage < 0.1):
        parser.error("fee-rate and slippage must be decimals between 0 and 0.1")
    if not (0 < args.position_pct <= 1):
        parser.error("position-pct must be greater than 0 and at most 1 (no leverage in this script)")
    if not (0 < args.oversold < args.overbought < 100):
        parser.error("Require 0 < oversold < overbought < 100")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    candles = fetch_candles(args.coin, args.interval, args.bars)
    candles, trades, equity, final_equity = run_backtest(
        candles, args.capital, args.fee_rate, args.slippage, args.position_pct,
        args.rsi_period, args.oversold, args.overbought,
    )
    report = performance_report(trades, equity, args.capital, final_equity)
    report.update({
        "coin": args.coin, "interval": args.interval, "rsi_period": args.rsi_period,
        "long_trigger_cross_up": args.oversold, "short_trigger_cross_down": args.overbought,
        "fee_rate_per_side_pct": args.fee_rate * 100,
        "slippage_per_side_pct": args.slippage * 100,
        "position_pct_of_equity": args.position_pct * 100,
        "candles_tested": int(len(candles)),
        "first_candle_utc": candles.datetime_utc.iloc[0].isoformat(),
        "last_candle_utc": candles.datetime_utc.iloc[-1].isoformat(),
        "funding_included": False,
        "leverage": 1.0,
    })

    candles.to_csv(outdir / "candles_with_rsi.csv", index=False)
    trades.to_csv(outdir / "trade_history.csv", index=False)
    equity.to_csv(outdir / "equity_curve.csv", index=False)
    with (outdir / "performance_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("\n========== PUMP RSI BACKTEST REPORT ==========")
    for key, value in report.items():
        if isinstance(value, float):
            if math.isinf(value):
                shown = "Infinity"
            else:
                shown = f"{value:,.4f}"
        else:
            shown = str(value)
        print(f"{key:34s}: {shown}")
    print("\nFiles saved to:")
    print(f"  {outdir / 'candles_with_rsi.csv'}")
    print(f"  {outdir / 'trade_history.csv'}")
    print(f"  {outdir / 'equity_curve.csv'}")
    print(f"  {outdir / 'performance_report.json'}")
    print("\nNote: Funding payments are NOT included. Verify the coin symbol and your actual fee tier before relying on results.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
