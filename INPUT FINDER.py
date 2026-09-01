"""
Finds the EMA_LENGTH / TP_POINTS / SL_POINTS combination with the best
PROFIT FACTOR (gross profit / gross loss).

TP and SL sweep the full 1-300 range independently. EMA_LENGTH sweeps
whatever list you set in EMA_LENGTHS_OVERRIDE below - put one value in
that list to keep EMA fixed, or several to sweep it too.

Run this from the same folder as this script (BACKTESTER_PATH below
points at the actual backtester file):
    python find_best_profit_factor_ETH.py
"""

import importlib.util
import pathlib
import time
import warnings
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Load backtester ETH-USD.py by path (space in the filename means
# it can't be `import`-ed normally).
# ------------------------------------------------------------
BACKTESTER_PATH = r"C:\Users\User\Desktop\backtester ETH-USD.py"

SCRIPT_PATH = pathlib.Path(BACKTESTER_PATH)
if not SCRIPT_PATH.exists():
    raise FileNotFoundError(
        f"Can't find the backtester at: {SCRIPT_PATH}\n"
        f"Update BACKTESTER_PATH near the top of this file to its full path."
    )
_spec = importlib.util.spec_from_file_location("backtester_eth", SCRIPT_PATH)
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

# ============================================================
# SEARCH RANGES - edit these to taste
#
# TP and SL sweep the FULL 1-300 range. STEP controls the resolution:
# STEP=1 tests every single value. Raise STEP to go faster.
# ============================================================

# EMA lengths to test. Put ONE value here to keep EMA fixed (e.g. [16]),
# or several to sweep it too (e.g. [8, 12, 16, 20, 24, 32]).
# None = use whatever EMA_LENGTH is already set in the backtester (fixed).
EMA_LENGTHS_OVERRIDE = list(range(1, 105))  # [8, 12, 16, 20, 24, 32] or None

TP_STEP = 5
SL_STEP = 5

TP_POINTS_RANGE = list(range(1, 501, TP_STEP))
SL_POINTS_RANGE = list(range(1, 501, SL_STEP))

MIN_TRADES = 200     # ignore combos with too few trades - a "great" profit
                     # factor from 5 trades is noise, not a real edge
TOP_N = 15            # how many rows to print in the leaderboard (kept for CSV / future use)

# Only test combos with TP_POINTS/SL_POINTS >= this. RR < 1 means your
# stop is bigger than your target, which pushes the breakeven win rate
# above 50% - set to 1.0 to require at least a 1:1 reward:risk.
MIN_RR_RATIO = 0

# Safety valve: if your STEP/EMA settings would still produce a huge grid,
# this warns you (rather than silently running for a long time) before starting.
WARN_ABOVE_COMBOS = 200_000

EMA_LENGTHS = list(EMA_LENGTHS_OVERRIDE) if EMA_LENGTHS_OVERRIDE else [bt.EMA_LENGTH]

# ============================================================


def build_signals_for_ema(chart, closes_by_tf, ema_length):
    """Recompute EMAs (cheap - just an ewm on already-resampled closes) and
    the resulting buy/sell signals for one EMA_LENGTH candidate."""
    data = chart.copy()
    for tf_key, col_name in (("1min", "ema_1m"), ("5min", "ema_5m"),
                              ("15min", "ema_15m"), ("30min", "ema_30m")):
        ema = bt.calculate_ema(closes_by_tf[tf_key], ema_length)
        data[col_name] = ema.reindex(chart.index, method="ffill")

    data = data.dropna(subset=["ema_1m", "ema_5m", "ema_15m", "ema_30m"])
    buy, sell, close, open_, high, low = bt.prepare_signals(data)
    return data, buy, sell, open_, high, low


def print_best_row(title, row):
    """Pretty-print a single 'best' combo row (used for all three leaderboards)."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"  EMA_LENGTH:      {int(row['ema_length'])}")
    print(f"  TP_POINTS:       {row['tp_points']}")
    print(f"  SL_POINTS:       {row['sl_points']}")
    print(f"  R:R:             {row['rr_ratio']}")
    print(f"  Profit Factor:   {row['profit_factor']:.3f}")
    print(f"  Total Trades:    {int(row['total_trades'])}")
    print(f"  Win Rate:        {row['win_rate']:.2f}%")
    print(f"  Expectancy:      {row['expectancy']:.2f} pts/trade")
    print(f"  Net Points:      {row['net_points']:.2f}")
    print(f"  Max Drawdown:    {row['max_drawdown']:.2f} pts")
    print(f"  Sharpe Ratio:    {row['sharpe_ratio']:.3f}")


def find_best_profit_factor():
    print("=" * 80)
    print("🔎 FIND BEST PROFIT FACTOR - ETH-USD")
    print("   Sweeping EMA_LENGTH x TP_POINTS x SL_POINTS")
    print("=" * 80)
    print(f"  EMA_LENGTHS:        {EMA_LENGTHS}")
    print(f"  TP_POINTS_RANGE:    {TP_POINTS_RANGE[0]}-{TP_POINTS_RANGE[-1]} step {TP_STEP} ({len(TP_POINTS_RANGE)} values)")
    print(f"  SL_POINTS_RANGE:    {SL_POINTS_RANGE[0]}-{SL_POINTS_RANGE[-1]} step {SL_STEP} ({len(SL_POINTS_RANGE)} values)")
    print(f"  MIN_TRADES:         {MIN_TRADES}")
    print(f"  MIN_RR_RATIO:       {MIN_RR_RATIO}  (TP/SL must be >= this)")

    tp_sl_combos = [(tp, sl) for tp, sl in product(TP_POINTS_RANGE, SL_POINTS_RANGE)
                    if tp / sl >= MIN_RR_RATIO]
    skipped = len(TP_POINTS_RANGE) * len(SL_POINTS_RANGE) - len(tp_sl_combos)
    total_combos = len(EMA_LENGTHS) * len(tp_sl_combos)
    print(f"  TP/SL combos:       {len(tp_sl_combos)}  ({skipped} skipped for RR < {MIN_RR_RATIO})")
    print(f"  Total combos:       {total_combos}  (x {len(EMA_LENGTHS)} EMA lengths)")
    print("=" * 80)

    if total_combos > WARN_ABOVE_COMBOS:
        print(f"\n⚠️  {total_combos:,} combos exceeds WARN_ABOVE_COMBOS ({WARN_ABOVE_COMBOS:,}).")
        print("   This could take a very long time. Raise TP_STEP / SL_STEP, or shrink")
        print("   EMA_LENGTHS_OVERRIDE, to reduce it - or press Enter to continue anyway.")
        input("   Press Enter to proceed, or Ctrl+C to cancel...")

    print("\nLoading dataset...")
    raw_data = bt.load_dataset()

    print("Resampling once per timeframe (reused across every EMA length)...")
    chart = bt.resample_ohlc(raw_data, bt.CHART_TIMEFRAME)
    closes_by_tf = {
        "1min": bt.resample_ohlc(raw_data, "1min")["close"],
        "5min": bt.resample_ohlc(raw_data, "5min")["close"],
        "15min": bt.resample_ohlc(raw_data, "15min")["close"],
        "30min": bt.resample_ohlc(raw_data, "30min")["close"],
    }

    # 1-minute arrays for the intrabar TP/SL tie-break - computed once,
    # reused across every EMA/TP/SL combo (only the bar-range mapping
    # below needs recomputing per EMA length, since dropna shifts rows).
    open_1m = raw_data["open"].to_numpy()
    high_1m = raw_data["high"].to_numpy()
    low_1m = raw_data["low"].to_numpy()
    close_1m = raw_data["close"].to_numpy()

    print("\nRunning sweep...")
    start_time = time.time()
    results = []
    i = 0
    for ema_length in EMA_LENGTHS:
        print(f"\n--- EMA_LENGTH = {ema_length} ---")
        data, buy, sell, open_, high, low = build_signals_for_ema(chart, closes_by_tf, ema_length)
        print(f"  {int(buy.sum())} buy signals, {int(sell.sum())} sell signals over {len(data):,} bars")

        # Bar-range mapping depends on which rows survived this EMA's
        # dropna, so it's recomputed per EMA length (cheap - vectorized).
        bar_1m_start, bar_1m_end = bt.compute_1m_bar_ranges(data.index, raw_data.index, bt.CHART_TIMEFRAME)

        for tp, sl in tp_sl_combos:
            i += 1
            pnl, entry_idx, exit_idx, sides, exit_reasons = bt.simulate(
                open_, high, low, buy, sell, float(tp), float(sl), bt.TOTAL_COST,
                bar_1m_start, bar_1m_end, open_1m, high_1m, low_1m, close_1m,
                bt.MIN_ENTRY_GAP_BARS, bt.USE_1M_TIE_BREAK
            )
            metrics = bt.calculate_metrics(pnl, exit_reasons)
            if metrics["total_trades"] >= MIN_TRADES:
                results.append({
                    "ema_length": ema_length,
                    "tp_points": tp,
                    "sl_points": sl,
                    "rr_ratio": round(tp / sl, 3),
                    **metrics,
                })

            if i == 20 or (i % 500 == 0) or i == total_combos:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_combos - i) / rate if rate > 0 else 0
                print(f"  {i:,}/{total_combos:,} combos "
                      f"({elapsed:,.1f}s elapsed, ~{remaining:,.1f}s remaining, "
                      f"{rate:,.0f} combos/s)", end="\r" if i != total_combos else "\n")

    if not results:
        print("\n⚠️  No combo produced at least MIN_TRADES trades - widen your ranges or lower MIN_TRADES.")
        return None

    results_df = pd.DataFrame(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"eth_best_profit_factor_{ts}.csv"
    results_df.sort_values("profit_factor", ascending=False).to_csv(out_file, index=False)
    print(f"\n✅ Full grid ({len(results_df)} qualifying combos) saved to: {out_file}")

    # At very small TP values, a trade's profit can be smaller than
    # COMMISSION+SLIPPAGE, so every trade nets exactly 0 P&L: 0 wins, 0
    # losses, and profit_factor's gross_profit/gross_loss becomes inf/inf
    # by the fallback rule. That's a breakeven artifact, not a real edge -
    # exclude it from the ranked leaderboard (it's still in the saved CSV).
    ranked_df = results_df[results_df["win_rate"] > 0].copy()
    if ranked_df.empty:
        print("\n⚠️  Every qualifying combo had a 0% win rate (likely TP too small to")
        print("   clear commission+slippage) - nothing meaningful to rank. Check the")
        print("   saved CSV, or raise MIN_TRADES / narrow the TP range.")
        return results_df

    # ------------------------------------------------------------
    # Three separate "top pick" leaderboards - each just the single
    # best combo by that metric, not a top-N table.
    # ------------------------------------------------------------
    best_pf = ranked_df.sort_values("profit_factor", ascending=False).iloc[0]
    best_net_points = ranked_df.sort_values("net_points", ascending=False).iloc[0]
    # max_drawdown is a magnitude of loss - "least drawdown" = smallest value
    best_drawdown = ranked_df.sort_values("max_drawdown", ascending=True).iloc[0]

    print_best_row("🏆 TOP PICK - BEST PROFIT FACTOR", best_pf)
    print_best_row("💰 TOP PICK - BEST NET POINTS", best_net_points)
    print_best_row("🛡️  TOP PICK - LEAST MAX DRAWDOWN", best_drawdown)

    print("\n" + "=" * 80)

    # Return the PF-ranked frame (kept for backward compatibility with any
    # calling code that expects a DataFrame back).
    return ranked_df.sort_values("profit_factor", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    find_best_profit_factor()
