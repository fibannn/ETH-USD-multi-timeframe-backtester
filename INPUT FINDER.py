import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
import itertools
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SETTINGS - ALL FILTERS OFF (FIXED PNL)
# ============================================================


DATASET_PATH = r"C:\Users\User\Desktop\python proj\DATASET ETH (30MONTHS)"
TIMEZONE = "Asia/Kolkata"
CHART_TIMEFRAME = "15min"  # 15-MINUTE CHART


# Fixed parameters
EMA_1H_LENGTH = 100


# ALL FILTERS TURNED OFF
USE_EMA_TOUCH_FILTER = False
EMA_TOUCH_THRESHOLD = 0.2
USE_SESSION_FILTER = False
USE_1H_TREND_FILTER = False
USE_VOLATILITY_FILTER = False  # TURNED OFF
USE_TIMEOUT_FILTER = False
TIMEOUT_BARS = 90

# Commission and slippage (in points) - FIXES PROFIT DISCREPANCY
COMMISSION = 0.5  # points per trade
SLIPPAGE = 0.5    # points per trade
TOTAL_COST = COMMISSION + SLIPPAGE  # 1.0 point total cost per trade


# EMA LENGTHS TO TEST
EMA_LENGTHS = [5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50]


# TP/SL TO TEST (in points)
TP_VALUES = [100, 120, 140, 150, 160, 170, 180, 185, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
SL_VALUES = [15, 20, 25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55, 60]


# Total combinations: 24 EMA * 20 TP * 15 SL = 7,200 combinations


# ============================================================
# DATA LOADING
# ============================================================


BINANCE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def detect_timestamp_unit(series):
    values = pd.to_numeric(series, errors="coerce").dropna().abs()
    median_value = values.median()
    if median_value < 1e11:
        return "s"
    if median_value < 1e14:
        return "ms"
    if median_value < 1e17:
        return "us"
    return "ns"


def load_one_file(file_path):
    print(f"Loading: {file_path.name}", end="\r")
    df = pd.read_csv(file_path, header=None, sep=None, engine="python")
    if df.shape[1] < 6:
        raise ValueError("File must contain horizontal OHLCV data.")
    first_value = str(df.iloc[0, 0]).strip().lower()
    if first_value in {"open_time", "open time", "timestamp", "time"}:
        df = df.iloc[1:].reset_index(drop=True)
    column_count = min(df.shape[1], len(BINANCE_COLUMNS))
    df = df.iloc[:, :column_count].copy()
    df.columns = BINANCE_COLUMNS[:column_count]
    required = ["open_time", "open", "high", "low", "close", "volume"]
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    timestamp_unit = detect_timestamp_unit(df["open_time"])
    df["time"] = pd.to_datetime(df["open_time"], unit=timestamp_unit, errors="coerce", utc=True)
    df["time"] = df["time"].dt.tz_convert(TIMEZONE)
    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    return df


def load_dataset():
    folder = Path(DATASET_PATH)
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = sorted(folder.glob("*.csv"))
    files = [f for f in files if f.name not in {"trades_output.csv", "monthly_profit.csv", "combined_data.csv"}]
    if not files:
        raise FileNotFoundError("No CSV files found.")
    print(f"Found {len(files)} CSV files.")
    frames = []
    for file_path in files:
        try:
            frames.append(load_one_file(file_path))
        except Exception as e:
            print(f"Skipped: {e}")
    if not frames:
        raise ValueError("No valid files were loaded.")
    data = pd.concat(frames).sort_index()
    data = data.loc[~data.index.duplicated(keep="first")]
    print(f"Combined 1-minute data: {len(data):,} candles")
    return data


# ============================================================
# RESAMPLING AND INDICATORS
# ============================================================


def resample_ohlc(data, timeframe):
    return (data[["open", "high", "low", "close", "volume"]]
            .resample(timeframe, label="right", closed="right")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna())


def calculate_ema(series, length):
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def align_ema(source_data, chart_data, timeframe, length):
    timeframe_data = resample_ohlc(source_data, timeframe)
    ema = calculate_ema(timeframe_data["close"], length)
    return ema.reindex(chart_data.index, method="ffill")


def prepare_data(raw_1m, ema_length):
    chart_15m = resample_ohlc(raw_1m, CHART_TIMEFRAME)
    data = chart_15m.copy()
    data["ema_1m"] = align_ema(raw_1m, data, "1min", ema_length)
    data["ema_5m"] = align_ema(raw_1m, data, "5min", ema_length)
    data["ema_15m"] = align_ema(raw_1m, data, "15min", ema_length)
    data["ema_30m"] = align_ema(raw_1m, data, "30min", ema_length)
    data["ema_1h"] = align_ema(raw_1m, data, "60min", EMA_1H_LENGTH)
    data = data.dropna(subset=["ema_1m", "ema_5m", "ema_15m", "ema_30m", "ema_1h"])
    return data


# ============================================================
# SIGNAL PREPARATION (ALL FILTERS OFF)
# ============================================================


def prepare_signals(data, ema_length):
    close = data["close"].to_numpy()
    open_price = data["open"].to_numpy()
    high = data["high"].to_numpy()
    low = data["low"].to_numpy()
    ema_1m = data["ema_1m"].to_numpy()
    ema_5m = data["ema_5m"].to_numpy()
    ema_15m = data["ema_15m"].to_numpy()
    ema_30m = data["ema_30m"].to_numpy()
    ema_1h = data["ema_1h"].to_numpy()
    number = len(data)
    
    bullish_1m = close > ema_1m
    bullish_5m = close > ema_5m
    bullish_15m = close > ema_15m
    bullish_30m = close > ema_30m
    bullish_1h = close > ema_1h
    
    bearish_1m = close < ema_1m
    bearish_5m = close < ema_5m
    bearish_15m = close < ema_15m
    bearish_30m = close < ema_30m
    bearish_1h = close < ema_1h
    
    # ALL BULLISH/BEARISH (no 1h filter)
    all_bullish = bullish_1m & bullish_5m & bullish_15m & bullish_30m
    all_bearish = bearish_1m & bearish_5m & bearish_15m & bearish_30m
    
    previous_bullish = np.roll(all_bullish, 1)
    previous_bearish = np.roll(all_bearish, 1)
    previous_bullish[:2] = False
    previous_bearish[:2] = False
    
    close_1 = np.roll(close, 1)
    close_2 = np.roll(close, 2)
    ema_5m_1 = np.roll(ema_5m, 1)
    ema_5m_2 = np.roll(ema_5m, 2)
    
    bull_confirm = (close_1 > ema_5m_1) & (close_2 > ema_5m_2)
    bear_confirm = (close_1 < ema_5m_1) & (close_2 < ema_5m_2)
    bull_confirm[:2] = False
    bear_confirm[:2] = False
    
    # ALL FILTERS OFF - always True
    candle_touches = np.ones(number, dtype=bool)
    wick_ok = np.ones(number, dtype=bool)
    bounce_bull = np.ones(number, dtype=bool)
    bounce_bear = np.ones(number, dtype=bool)
    session_ok = np.ones(number, dtype=bool)
    volatility_ok = np.ones(number, dtype=bool)
    
    buy_signal = all_bullish & bull_confirm & candle_touches & bounce_bull & wick_ok & ~previous_bullish & session_ok & volatility_ok
    sell_signal = all_bearish & bear_confirm & bounce_bear & wick_ok & ~previous_bearish & session_ok & volatility_ok
    
    return buy_signal, sell_signal, close, open_price, high, low


# ============================================================
# TRADE SIMULATOR (FIXED - WITH COMMISSION/SLIPPAGE)
# ============================================================


@njit(cache=True)
def simulate(open_price, high, low, buy_signal, sell_signal, tp_points, sl_points, timeout_bars, use_timeout_filter, total_cost):
    n = len(open_price)
    trade_pnl = np.empty(n)
    entry_indices = np.empty(n, dtype=np.int64)
    exit_indices = np.empty(n, dtype=np.int64)
    sides = np.empty(n, dtype=np.int64)
    trade_count = 0
    position = 0
    entry_price = 0.0
    entry_index = -1
    timeout_counter = 0
    for i in range(2, n - 1):
        if position != 0:
            if position == 1:
                tp_price = entry_price + tp_points
                sl_price = entry_price - sl_points
                sl_hit = low[i] <= sl_price
                tp_hit = high[i] >= tp_price
                if sl_hit:
                    trade_pnl[trade_count] = -sl_points - total_cost  # FIXED: Subtract cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = 1
                    trade_count += 1
                    position = 0
                elif tp_hit:
                    trade_pnl[trade_count] = tp_points - total_cost  # FIXED: Subtract cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = 1
                    trade_count += 1
                    position = 0
            else:
                tp_price = entry_price - tp_points
                sl_price = entry_price + sl_points
                sl_hit = high[i] >= sl_price
                tp_hit = low[i] <= tp_price
                if sl_hit:
                    trade_pnl[trade_count] = -sl_points - total_cost  # FIXED: Subtract cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = -1
                    trade_count += 1
                    position = 0
                elif tp_hit:
                    trade_pnl[trade_count] = tp_points - total_cost  # FIXED: Subtract cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = -1
                    trade_count += 1
                    position = 0
        if timeout_counter > 0:
            timeout_counter -= 1
        timeout_active = use_timeout_filter and timeout_counter > 0
        if position == 0 and not timeout_active:
            if buy_signal[i]:
                position = 1
                entry_index = i + 1
                entry_price = open_price[i + 1]
            elif sell_signal[i]:
                position = -1
                entry_index = i + 1
                entry_price = open_price[i + 1]
        if position == 0 and use_timeout_filter:
            if trade_count > 0:
                timeout_counter = timeout_bars
    return trade_pnl[:trade_count], entry_indices[:trade_count], exit_indices[:trade_count], sides[:trade_count]


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(pnl):
    if len(pnl) == 0:
        return {"total_trades": 0, "win_rate": 0.0, "net_points": 0.0, "expectancy": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "largest_win": 0.0, "largest_loss": 0.0, "sharpe_ratio": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    equity = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(equity) - equity
    returns = np.diff(equity)
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0.0
    return {
        "total_trades": len(pnl),
        "win_rate": len(wins) / len(pnl) * 100 if len(pnl) > 0 else 0.0,
        "net_points": pnl.sum(),
        "expectancy": pnl.mean() if len(pnl) > 0 else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "max_drawdown": drawdown.max() if len(drawdown) > 0 else 0.0,
        "avg_win": wins.mean() if len(wins) > 0 else 0.0,
        "avg_loss": abs(losses.mean()) if len(losses) > 0 else 0.0,
        "largest_win": wins.max() if len(wins) > 0 else 0.0,
        "largest_loss": losses.min() if len(losses) > 0 else 0.0,
        "sharpe_ratio": sharpe,
    }


# ============================================================
# OPTIMIZATION
# ============================================================


def run_optimization():
    print("=" * 80)
    print("🏆 CHAMPION PARAMETER OPTIMIZATION (ALL FILTERS OFF)")
    print("=" * 80)
    print(f"\n⚙️  Fixed Settings:")
    print(f"  CHART_TIMEFRAME: {CHART_TIMEFRAME}")
    print(f"  EMA_1H_LENGTH: {EMA_1H_LENGTH}")
    print(f"  COMMISSION: {COMMISSION} points")
    print(f"  SLIPPAGE: {SLIPPAGE} points")
    print(f"  TOTAL COST: {TOTAL_COST} points per trade")
    print(f"\n🔍 Parameter Ranges:")
    print(f"  EMA_LENGTH: {len(EMA_LENGTHS)} values [{min(EMA_LENGTHS)}-{max(EMA_LENGTHS)}]")
    print(f"  TP_POINTS: {len(TP_VALUES)} values [{min(TP_VALUES)}-{max(TP_VALUES)}]")
    print(f"  SL_POINTS: {len(SL_VALUES)} values [{min(SL_VALUES)}-{max(SL_VALUES)}]")
    total_combos = len(EMA_LENGTHS) * len(TP_VALUES) * len(SL_VALUES)
    print(f"\n💥 Total Combinations: {total_combos:,}")
    print("=" * 80)
    
    print("\nLoading dataset...")
    raw_data = load_dataset()
    
    combinations = list(itertools.product(EMA_LENGTHS, TP_VALUES, SL_VALUES))
    print(f"\nTesting {len(combinations):,} combinations...\n")
    
    results = []
    
    for idx, (ema_len, tp, sl) in enumerate(combinations, 1):
        print(f"[{idx:5d}/{total_combos:,}] EMA={ema_len:2d}, TP={tp:3d}, SL={sl:2d}", end="\r")
        
        try:
            data = prepare_data(raw_data, ema_len)
            buy_signal, sell_signal, close, open_price, high, low = prepare_signals(data, ema_len)
            pnl, entry_indices, exit_indices, sides = simulate(
                open_price, high, low, buy_signal, sell_signal,
                tp, sl, TIMEOUT_BARS, USE_TIMEOUT_FILTER, TOTAL_COST
            )
            metrics = calculate_metrics(pnl)
            results.append({
                "ema_length": ema_len,
                "tp_points": tp,
                "sl_points": sl,
                **metrics
            })
        except Exception as e:
            continue
    
    print("\n\n")
    
    if not results:
        print("No valid results found!")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Composite score
    results_df["score"] = (
        results_df["net_points"].rank(pct=True) * 0.5 +
        (1 - results_df["max_drawdown"].rank(pct=True)) * 0.3 +
        results_df["expectancy"].rank(pct=True) * 0.2
    )
    
    results_df = results_df.sort_values(
        by=["net_points", "max_drawdown", "expectancy"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"champion_optimization_{len(combinations):,}_combinations_{timestamp}.csv".replace(",", "")
    results_df.to_csv(output_file, index=False)
    
    # Display results
    print("=" * 80)
    print("🏆 TOP 20 CHAMPION COMBINATIONS")
    print("=" * 80)
    display_cols = ["ema_length", "tp_points", "sl_points", "net_points", "max_drawdown", "total_trades", "win_rate", "expectancy", "profit_factor", "sharpe_ratio"]
    print(results_df[display_cols].head(20).to_string(index=False))
    
    best = results_df.iloc[0]
    print("\n" + "=" * 80)
    print("🥇 #1 CHAMPION COMBINATION")
    print("=" * 80)
    print(f"EMA_LENGTH:  {int(best['ema_length'])}")
    print(f"TP_POINTS:   {int(best['tp_points'])}")
    print(f"SL_POINTS:   {int(best['sl_points'])}")
    print(f"\n📊 Performance:")
    print(f"  Net Points:    {best['net_points']:.2f}")
    print(f"  Max Drawdown:  {best['max_drawdown']:.2f}")
    print(f"  Total Trades:  {int(best['total_trades'])}")
    print(f"  Win Rate:      {best['win_rate']:.2f}%")
    print(f"  Expectancy:    {best['expectancy']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    print(f"  Sharpe Ratio:  {best['sharpe_ratio']:.3f}")
    print(f"  Avg Win:       {best['avg_win']:.2f}")
    print(f"  Avg Loss:      {best['avg_loss']:.2f}")
    
    print("\n" + "=" * 80)
    print("🥈 TOP 5 DETAILED RESULTS")
    print("=" * 80)
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n#{i+1}: EMA={int(row['ema_length']):2d}, TP={int(row['tp_points']):3d}, SL={int(row['sl_points']):2d}")
        print(f"    Net: {row['net_points']:8.2f} | DD: {row['max_drawdown']:6.2f} | Trades: {int(row['total_trades']):4d} | Win%: {row['win_rate']:5.2f} | Exp: {row['expectancy']:6.2f} | PF: {row['profit_factor']:5.2f} | Sharpe: {row['sharpe_ratio']:.3f}")
    
    print(f"\n✅ All {len(combinations):,} results saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    results = run_optimization()