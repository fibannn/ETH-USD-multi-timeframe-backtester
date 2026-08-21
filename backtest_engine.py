import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# SETTINGS - MATCHES (ETH) CHAMPION - 1m/15m/30m Stack
# ============================================================

DATASET_PATH = r"C:\Users\User\Desktop\python proj\DATASET ETH (30MONTHS)"
TIMEZONE = "Asia/Kolkata"
CHART_TIMEFRAME = "15min"  # Your strategy chart timeframe

EMA_LENGTH = 16
TP_POINTS = 166.0
SL_POINTS = 34.0

COMMISSION = 0.0
SLIPPAGE = 0.5
TOTAL_COST = COMMISSION + SLIPPAGE

# New tuned parameters
MAX_WICK_RATIO = 0.94       # was 0.7
BULL_BOUNCE_MULT = 1.0018   # was 1.003
BEAR_BOUNCE_MULT = 0.9982   # was 0.997

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
    print(f"Timestamp unit detected: {timestamp_unit}")
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


def prepare_data(raw_1m):
    chart = resample_ohlc(raw_1m, CHART_TIMEFRAME)
    data = chart.copy()
    data["ema_1m"] = align_ema(raw_1m, data, "1min", EMA_LENGTH)
    data["ema_5m"] = align_ema(raw_1m, data, "5min", EMA_LENGTH)
    data["ema_15m"] = align_ema(raw_1m, data, "15min", EMA_LENGTH)
    data["ema_30m"] = align_ema(raw_1m, data, "30min", EMA_LENGTH)
    data = data.dropna(subset=["ema_1m", "ema_5m", "ema_15m", "ema_30m"])
    return data

# ============================================================
# SIGNAL LOGIC (UPDATED TO MATCH NEW STRATEGY)
# trend stack + 5m confirm + bounce (1.0018 / 0.9982) + wick < 0.94
# ============================================================


def prepare_signals(data):
    close = data["close"].to_numpy()
    open_price = data["open"].to_numpy()
    high = data["high"].to_numpy()
    low = data["low"].to_numpy()
    ema_1m = data["ema_1m"].to_numpy()
    ema_5m = data["ema_5m"].to_numpy()
    ema_15m = data["ema_15m"].to_numpy()
    ema_30m = data["ema_30m"].to_numpy()

    # Trend stack 1m/15m/30m
    trend_1min_bull = close > ema_1m
    trend_15min_bull = close > ema_15m
    trend_30min_bull = close > ema_30m
    trend_1min_bear = close < ema_1m
    trend_15min_bear = close < ema_15m
    trend_30min_bear = close < ema_30m

    all_bullish = trend_1min_bull & trend_15min_bull & trend_30min_bull
    all_bearish = trend_1min_bear & trend_15min_bear & trend_30min_bear

    # 5m confirm
    close_1 = np.roll(close, 1)
    close_2 = np.roll(close, 2)
    ema_5m_1 = np.roll(ema_5m, 1)
    ema_5m_2 = np.roll(ema_5m, 2)
    bull_confirm = (close_1 > ema_5m_1) & (close_2 > ema_5m_2)
    bear_confirm = (close_1 < ema_5m_1) & (close_2 < ema_5m_2)
    bull_confirm[:2] = False
    bear_confirm[:2] = False

    # Wick filter (updated threshold)
    body = np.abs(close - open_price)
    candle_range = high - low
    wick_ratio = np.where(candle_range > 0, (candle_range - body) / candle_range, 0.0)
    no_excessive_wick = wick_ratio < MAX_WICK_RATIO   # 0.94

    # Bounce filter (updated multipliers)
    ema_bounce_bull = (low <= ema_5m * BULL_BOUNCE_MULT) & (close > ema_5m)
    ema_bounce_bear = (high >= ema_5m * BEAR_BOUNCE_MULT) & (close < ema_5m)

    # Fresh signal (not all_bullish[1])
    all_bullish_prev = np.roll(all_bullish, 1)
    all_bearish_prev = np.roll(all_bearish, 1)
    all_bullish_prev[0] = False
    all_bearish_prev[0] = False

    buy_condition = all_bullish & bull_confirm & ema_bounce_bull & no_excessive_wick & ~all_bullish_prev
    sell_condition = all_bearish & bear_confirm & ema_bounce_bear & no_excessive_wick & ~all_bearish_prev

    buy_condition[:3] = False
    sell_condition[:3] = False

    return buy_condition, sell_condition, close, open_price, high, low

# ============================================================
# TRADE SIMULATOR (TP/SL ONLY)
# Same entry/exit logic as before
# ============================================================


@njit(cache=True)
def simulate(open_price, high, low, buy_signal, sell_signal, tp_points, sl_points, total_cost):
    n = len(open_price)
    trade_pnl = np.empty(n)
    entry_indices = np.empty(n, dtype=np.int64)
    exit_indices = np.empty(n, dtype=np.int64)
    sides = np.empty(n, dtype=np.int64)
    exit_reasons = np.empty(n, dtype=np.int64)  # 1=TP, 2=SL
    trade_count = 0
    position = 0
    entry_price = 0.0
    entry_index = -1

    for i in range(2, n - 1):
        if position != 0:
            if position == 1:
                tp_price = entry_price + tp_points
                sl_price = entry_price - sl_points
                sl_hit = low[i] <= sl_price
                tp_hit = high[i] >= tp_price
                if sl_hit:
                    trade_pnl[trade_count] = -sl_points - total_cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = 1
                    exit_reasons[trade_count] = 2
                    trade_count += 1
                    position = 0
                elif tp_hit:
                    trade_pnl[trade_count] = tp_points - total_cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = 1
                    exit_reasons[trade_count] = 1
                    trade_count += 1
                    position = 0
            else:
                tp_price = entry_price - tp_points
                sl_price = entry_price + sl_points
                sl_hit = high[i] >= sl_price
                tp_hit = low[i] <= tp_price
                if sl_hit:
                    trade_pnl[trade_count] = -sl_points - total_cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = -1
                    exit_reasons[trade_count] = 2
                    trade_count += 1
                    position = 0
                elif tp_hit:
                    trade_pnl[trade_count] = tp_points - total_cost
                    entry_indices[trade_count] = entry_index
                    exit_indices[trade_count] = i
                    sides[trade_count] = -1
                    exit_reasons[trade_count] = 1
                    trade_count += 1
                    position = 0

        # Entry at next bar open (like Pine strategy)
        if position == 0:
            if buy_signal[i]:
                position = 1
                entry_index = i + 1
                entry_price = open_price[i + 1]
            elif sell_signal[i]:
                position = -1
                entry_index = i + 1
                entry_price = open_price[i + 1]

    return trade_pnl[:trade_count], entry_indices[:trade_count], exit_indices[:trade_count], sides[:trade_count], exit_reasons[:trade_count]

# ============================================================
# METRICS
# ============================================================


def calculate_metrics(pnl, exit_reasons):
    if len(pnl) == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "net_points": 0.0, "expectancy": 0.0,
            "profit_factor": 0.0, "max_drawdown": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "tp_hits": 0, "sl_hits": 0, "sharpe_ratio": 0.0
        }
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    equity = np.cumsum(pnl)
    drawdown = np.maximum.accumulate(equity) - equity
    returns = np.diff(equity)
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0

    tp_count = np.sum(exit_reasons == 1)
    sl_count = np.sum(exit_reasons == 2)

    return {
        "total_trades": len(pnl),
        "win_rate": len(wins) / len(pnl) * 100,
        "net_points": pnl.sum(),
        "expectancy": pnl.mean(),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "max_drawdown": drawdown.max(),
        "avg_win": wins.mean(),
        "avg_loss": abs(losses.mean()),
        "tp_hits": tp_count,
        "sl_hits": sl_count,
        "sharpe_ratio": sharpe,
    }

# ============================================================
# MAIN BACKTEST FUNCTION
# ============================================================


def run_backtest():
    print("=" * 80)
    print("🧪 BACKTESTER - (ETH) CHAMPION - Updated Wick/Bounce")
    print("=" * 80)
    print(f"\n📌 Parameters:")
    print(f"  EMA_LENGTH:      {EMA_LENGTH}")
    print(f"  TP_POINTS:       {TP_POINTS}")
    print(f"  SL_POINTS:       {SL_POINTS}")
    print(f"  CHART_TIMEFRAME: {CHART_TIMEFRAME}")
    print(f"  MAX_WICK_RATIO:  {MAX_WICK_RATIO}")
    print(f"  BULL_BOUNCE:     {BULL_BOUNCE_MULT}")
    print(f"  BEAR_BOUNCE:     {BEAR_BOUNCE_MULT}")
    print("=" * 80)

    print("\nLoading dataset...")
    raw_data = load_dataset()

    print("\nPreparing data & signals...")
    data = prepare_data(raw_data)
    buy_signal, sell_signal, close, open_price, high, low = prepare_signals(data)

    print("\nRunning simulation...")
    pnl, entry_indices, exit_indices, sides, exit_reasons = simulate(
        open_price, high, low, buy_signal, sell_signal,
        TP_POINTS, SL_POINTS, TOTAL_COST
    )

    print("\nCalculating metrics...")
    metrics = calculate_metrics(pnl, exit_reasons)

    trades_df = pd.DataFrame({
        "entry_time": data.index[entry_indices],
        "exit_time": data.index[exit_indices],
        "side": np.where(sides == 1, "LONG", "SHORT"),
        "entry_price": open_price[entry_indices],
        "exit_price": open_price[exit_indices],
        "points": pnl,
        "exit_reason": np.where(exit_reasons == 1, "TP", "SL"),
    })

    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"  Net Points:     {metrics['net_points']:.2f}")
    print(f"  Total Trades:   {metrics['total_trades']}")
    print(f"  Win Rate:       {metrics['win_rate']:.2f}%")
    print(f"  Expectancy:     {metrics['expectancy']:.2f} pts")
    print(f"  Profit Factor:  {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.2f} pts")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.3f}")
    print(f"  TP Hits:        {metrics['tp_hits']}")
    print(f"  SL Hits:        {metrics['sl_hits']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_file = f"eth_champion_updated_backtest_EMA{EMA_LENGTH}_TP{TP_POINTS}_SL{SL_POINTS}_{ts}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"\n✅ Trades saved to: {trades_file}")

    print("\n" + "=" * 80)
    print("📋 LAST 10 TRADES")
    print("=" * 80)
    print(trades_df.tail(10).to_string(index=False))

    return trades_df, metrics


if __name__ == "__main__":
    trades_df, metrics = run_backtest()
