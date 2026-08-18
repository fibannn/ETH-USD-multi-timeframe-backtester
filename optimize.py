import argparse
import json
import warnings
import pandas as pd
import numpy as np
import optuna
from backtest_engine import prepare_base, attach_emas, run_backtest

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def load_1m_csv(path: str) -> pd.DataFrame:
    # peek at first line to detect whether this is a labeled CSV or a raw
    # headerless Binance kline export
    with open(path) as f:
        first_line = f.readline().strip()
    first_token = first_line.split(",")[0].strip().strip('"')

    if first_token.replace(".", "", 1).lstrip("-").isdigit():
        # raw Binance klines: open_time,open,high,low,close,volume,close_time,
        # quote_volume,trades,taker_buy_base,taker_buy_quote,ignore
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
        df = pd.read_csv(path, header=None, names=cols)
        ts = df["open_time"].iloc[0]
        if ts > 10**17:
            unit = "ns"
        elif ts > 10**14:
            unit = "us"
        elif ts > 10**11:
            unit = "ms"
        else:
            unit = "s"
        df["time"] = pd.to_datetime(df["open_time"], unit=unit)
        df = df.set_index("time").sort_index()
        return df[["open", "high", "low", "close"]]

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next(c for c in df.columns if c in ("time", "timestamp", "date", "datetime"))
    df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None)
    df = df.rename(columns={time_col: "time"}).set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")
    return df[["open", "high", "low", "close"]]


def build_objective(base: dict, min_trades: int, ema_cache: dict, metric: str = "win_rate"):

    def objective(trial):
        ema_length = trial.suggest_int("ema_length", 10, 60, step=2)
        ema_1h_length = trial.suggest_int("ema_1h_length", 10, 200, step=5)

        cache_key = (ema_length, ema_1h_length)
        if cache_key not in ema_cache:
            ema_cache[cache_key] = attach_emas(base, ema_length, ema_1h_length)
        data = ema_cache[cache_key]

        stop_loss_points = trial.suggest_float("stop_loss_points", 15, 100, step=5)
        min_points = trial.suggest_float("min_points", 30, 400, step=10)

        params = dict(
            min_points=min_points,
            stop_loss_points=stop_loss_points,
            use_ema_touch_filter=trial.suggest_categorical("use_ema_touch_filter", [True, False]),
            ema_touch_threshold=trial.suggest_float("ema_touch_threshold", 0.1, 2.0, step=0.1),
            use_session_filter=trial.suggest_categorical("use_session_filter", [True, False]),
            timeout_bars=trial.suggest_int("timeout_bars", 10, 200, step=10),
            use_timeout_filter=trial.suggest_categorical("use_timeout_filter", [True, False]),
            use_1h_trend_filter=trial.suggest_categorical("use_1h_trend_filter", [True, False]),
        )

        result = run_backtest(data, params)
        trial.set_user_attr("metrics", result)

        if result["total_trades"] < min_trades:
            return -1e9 if metric == "net_points" else -1.0
        return result[metric]

    return objective


def robustness_check(base, best_params, ema_cache, min_trades):
    """Perturb each numeric param by one step and see if win rate holds up
    nearby -- flags results that only work at one exact, fragile value."""
    numeric = ["ema_length", "ema_1h_length", "min_points", "stop_loss_points",
               "timeout_bars", "ema_touch_threshold"]
    steps = {"ema_length": 2, "ema_1h_length": 5, "min_points": 10,
             "stop_loss_points": 5, "timeout_bars": 10, "ema_touch_threshold": 0.1}
    neighbor_rates = []
    for p in numeric:
        for direction in (-1, 1):
            variant = dict(best_params)
            variant[p] = variant[p] + direction * steps[p]
            if variant[p] <= 0:
                continue
            ema_length = variant["ema_length"]
            ema_1h_length = variant["ema_1h_length"]
            key = (ema_length, ema_1h_length)
            if key not in ema_cache:
                ema_cache[key] = attach_emas(base, ema_length, ema_1h_length)
            data = ema_cache[key]
            bt_params = {k: v for k, v in variant.items() if k not in ("ema_length", "ema_1h_length")}
            r = run_backtest(data, bt_params)
            if r["total_trades"] >= min_trades:
                neighbor_rates.append(r["win_rate"])
    if not neighbor_rates:
        return None
    return {"neighbor_avg_win_rate": round(float(np.mean(neighbor_rates)), 2),
            "neighbor_min_win_rate": round(float(np.min(neighbor_rates)), 2),
            "n_neighbors_tested": len(neighbor_rates)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--chart-tf", default="15min",
                     help="Chart timeframe the strategy trades on, e.g. 15min, 5min, 1h")
    ap.add_argument("--trials", type=int, default=1500)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--metric", default="win_rate", choices=["win_rate", "net_points", "expectancy", "profit_factor"])
    args = ap.parse_args()

    df_1m = load_1m_csv(args.csv)
    print(f"Loaded {len(df_1m):,} 1-min bars from {df_1m.index[0]} to {df_1m.index[-1]}")
    print(f"Trading on {args.chart_tf} chart bars, ranking by {args.metric}")

    base = prepare_base(df_1m, args.chart_tf)
    ema_cache = {}
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({
        "ema_length": 42, "ema_1h_length": 25, "stop_loss_points": 60.0,
        "min_points": 190.0, "use_ema_touch_filter": True, "ema_touch_threshold": 1.5,
        "use_session_filter": False, "timeout_bars": 40, "use_timeout_filter": False,
        "use_1h_trend_filter": True,
    })
    study.optimize(build_objective(base, args.min_trades, ema_cache, args.metric),
                    n_trials=args.trials, show_progress_bar=False)

    trials = [t for t in study.trials if t.value is not None and t.value > -1e8]
    trials.sort(key=lambda t: t.user_attrs["metrics"][args.metric], reverse=True)

    print(f"\n{len(trials)}/{args.trials} trials met the min-trades={args.min_trades} threshold\n")
    print("=" * 100)
    results = []
    for rank, t in enumerate(trials[:args.top_n], 1):
        m = t.user_attrs["metrics"]
        row = {"rank": rank, **t.params, **m}
        results.append(row)
        rr = t.params["min_points"] / t.params["stop_loss_points"]
        print(f"#{rank}  win_rate={m['win_rate']}%  trades={m['total_trades']}  "
              f"expectancy={m['expectancy']}pts  net={m['net_points']}pts  "
              f"pf={m['profit_factor']}  maxDD={m['max_drawdown']}  RR=1:{rr:.2f}")
        print(f"    params: { {k: v for k, v in t.params.items()} }")

    if trials:
        print("\nRobustness check on #1 (does the win rate hold at nearby parameter values?):")
        rc = robustness_check(base, trials[0].params, ema_cache, args.min_trades)
        print(rc if rc else "  Not enough neighboring combos cleared the trade-count threshold.")

    pd.DataFrame(results).to_csv("optimization_results.csv", index=False)
    print("\nFull top results saved to optimization_results.csv")


if __name__ == "__main__":
    main()
