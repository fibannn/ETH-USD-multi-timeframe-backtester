# Case Study: Building and Validating a Systematic Trading Strategy Backtester

## Table of Contents
1. [The Problem](#1-the-problem)
2. [Building the Engine](#2-building-the-engine)
3. [First Results — and Why I Didn't Trust Them](#3-first-results--and-why-i-didnt-trust-them)
4. [Catching Overfitting: 6 Months vs 18 Months](#4-catching-overfitting-6-months-vs-18-months)
5. [Cross-Validating Against Real Execution Data](#5-cross-validating-against-real-execution-data)
6. [The Whipsaw Problem, and Nine Failed Fixes](#6-the-whipsaw-problem-and-nine-failed-fixes)
7. [The Bug: An 87% Overstatement](#7-the-bug-an-87-overstatement)
8. [Where the Project Stands](#8-where-the-project-stands)
9. [What I'd Do Differently](#9-what-id-do-differently)

---

## 1. The Problem

The starting point was a discretionary trading strategy written in Pine Script for TradingView — a multi-timeframe EMA trend-alignment system (checking agreement across 1-minute, 5-minute, 15-minute, and 30-minute EMAs before entering a trade, with a fixed take-profit/stop-loss exit). Testing parameter changes meant manually adjusting inputs in TradingView's Strategy Tester one at a time — no way to search a parameter space of any real size, and no way to be confident a "good" result wasn't just a lucky combination.

**Goal**: build a Python engine that could search this parameter space systematically, verify results weren't overfit, and — critically — actually validate the backtest against reality rather than just trusting it.

## 2. Building the Engine

The Pine Script logic was ported to Python as closely as possible, replicating:
- The 5-timeframe EMA trend-alignment check
- An EMA "touch and bounce" entry filter
- A candle wick-ratio filter (rejecting entries on candles that are mostly wick)
- Fixed-point take-profit and stop-loss exits

**Performance was a real constraint.** A naive Python simulation loop over months of 1-minute data, run thousands of times during a parameter search, would be far too slow. The core simulation loop was rewritten using [Numba](https://numba.pydata.org/)'s JIT compiler, bringing a full-history backtest down to roughly 2 milliseconds — fast enough to run thousands of trials in a parameter search that would otherwise take hours.

Parameter search itself used [Optuna](https://optuna.org/) with a TPE (Tree-structured Parzen Estimator) sampler — smarter than grid search for a space too large to brute-force, since it learns which regions of the parameter space look promising and concentrates trials there.

## 3. First Results — and Why I Didn't Trust Them

An early search (win-rate-ranked, on a relatively short window of data) turned up a configuration with an ~88% win rate. That result was **immediately suspicious** rather than exciting — a win rate that high for a trend-following strategy with a modest reward:risk ratio doesn't have an obvious economic explanation. Testing it against a longer window confirmed the suspicion: it was a fluke of one unusually strong trending period, not a real edge.

This set the tone for the rest of the project: **any result that looked "too good" got tested harder, not trusted more.**

## 4. Catching Overfitting: 6 Months vs 18 Months

A later configuration, tuned on 6 months of data, showed genuinely clean performance — zero losing months, a reasonable win rate, solid profit factor. Before accepting it, the same configuration was re-tested against a full 18 months of data (the original 6 months plus a full preceding year).

**The result changed materially.** Net profit dropped, and 8 of the 18 months were now losing — a completely different risk profile than the "zero losing months" picture the 6-month test had shown. Plotting the monthly results made the cause obvious: the config's positive months were concentrated almost exactly in the window it had originally been tuned on. **This is overfitting made visible**, not a subtle statistical inference — the strategy had learned the specific noise of its tuning window, not a durable pattern.

A fresh parameter search across the full 18-month window (later extended to 30 months, spanning three different calendar years and market regimes) found a materially different, more robust configuration. The lesson: a backtest's apparent quality should always be checked against how much data — and how much *out-of-sample* data — it was actually validated on.

## 5. Cross-Validating Against Real Execution Data

Throughout the project, results were periodically checked against real trade-level data from a live (paper) trading account running the equivalent strategy on TradingView. This surfaced two separate, genuine data-quality issues that a backtest-only workflow would never have caught:

**Timezone misalignment (gold/XAUUSD).** The historical data source used a fixed timestamp convention (EST, no daylight-saving adjustment) while the live chart displayed in a different timezone. This meant every multi-timeframe EMA calculation was being computed on subtly misaligned bars. Comparing specific real executed trades against the backtest's simulated trades for the same period — matching them up by approximate price level and direction — revealed a consistent multi-hour offset, which was then corrected and re-validated.

**Broker/exchange price divergence (ETH).** Comparing real trades against backtest trades for the same nominal strategy showed the *pattern* of wins and losses matched reasonably well, but *exact* entry/exit prices and timing sometimes diverged meaningfully. This was traced to a genuine difference between the exchange used for historical data and the broker used for live execution (different underlying liquidity, different price feeds) — a real, unavoidable source of backtest-to-live divergence worth knowing about rather than being surprised by later.

## 6. The Whipsaw Problem, and Nine Failed Fixes

Certain historical months consistently underperformed across many different tested configurations — investigation showed these weren't crashes, but periods of high volatility with low *net* directional progress ("whipsaw": sharp moves that fully reverse within days, repeatedly stopping out a trend-following entry before the trend can develop).

Nine distinct approaches were tested to reduce this vulnerability. All nine were rejected:

| # | Approach | Why it failed |
|---|---|---|
| 1 | ADX trend-strength filter | Reduced bad-month losses, but by damaging good months equally |
| 2 | Volume confirmation | Reduced profit at every threshold tested (5 thresholds tried) |
| 3 | Max trade-hold-duration cutoff | Disproportionately cut the strategy's largest winning trades |
| 4 | Consecutive-loss circuit breaker | Made both drawdown *and* the underlying losing streak worse |
| 5 | Blanket cooldown after every trade | Turned the strategy net-negative at longer cooldown settings |
| 6 | Trend-persistence requirement (N-bar confirmation) | No win-rate improvement — just fewer trades |
| 7 | Dynamic (volatility-scaled) stop-loss | No clean improvement at any tested scaling factor |
| 8 | Minimum EMA separation ("conviction" threshold) | Win rate stayed flat while trade count and profit fell |
| 9 | Cross-asset (BTC) trend confirmation | Slightly worse win rate, ~26% profit reduction |

**The pattern across all nine** is more useful than any individual result: approaches evaluating *present market conditions* (e.g., current volatility relative to its own recent average — which *did* eventually work, once properly tuned) tended to be viable; approaches reacting to *recent trade history* (a loss just happened, a streak is underway) consistently were not. A plausible explanation: trade outcomes are a lagging, noisy signal of market conditions, while a direct measurement of current volatility is a leading one.

Two approaches evaluating present conditions were eventually accepted:
- An **ATR-based volatility filter**, blocking new entries when current volatility is abnormally elevated relative to its own recent average
- A **close-based stop-loss confirmation** rule, requiring a bar's close (not just an intrabar price wick) to breach the stop level before exiting — filtering out brief noise spikes that reverse immediately

## 7. The Bug: An 87% Overstatement

This is the most important part of the project, and not because of a success.

After adopting close-based stop-loss confirmation, backtest results looked very strong — a configuration with 30 out of 30 profitable months across the full dataset. Before trusting this, real trade-level data from live paper trading was compared against it, trade by trade, in chronological order.

**Every winning trade matched the expected take-profit value exactly. Not a single losing trade matched the expected stop-loss value.** Realized losses clustered somewhat above the nominal stop distance, with a smaller but real tail of much larger losses during fast moves.

The cause, once found, was simple: the backtest's close-based stop-loss logic recorded every such loss as exactly the nominal stop distance, regardless of where the confirming bar's close actually landed. In reality — and as the live data now clearly showed — a fast move can carry price well past the nominal stop level before a bar closes to confirm it. The backtest was silently assuming a best-case exit price on every single stop-loss trade using this mechanism.

**Correcting this and re-running the full 30-month backtest**: net profit dropped from the previously reported figure by roughly 87%. Average realized loss size, measured honestly, was nearly double the nominal stop distance the backtest had assumed.

This single bug had been sitting underneath every verification step performed up to that point — exhaustive parameter grids, monthly consistency checks, out-of-sample testing across three years of data. None of those checks could catch it, because they were all built using the same flawed core assumption. **Only comparison against real, ground-truth execution data surfaced it.**

## 8. Where the Project Stands
### Final configuration and results — updated (EMA 9 / TP 351 / SL 141)

A subsequent grid sweep across EMA length, take-profit, and stop-loss — ranked by net points rather than profit factor alone, and cross-checked against the profit-factor-ranked and drawdown-ranked leaderboards from the same sweep — surfaced a materially different configuration as the best-balanced result:

**Parameters**

| Parameter       | Value |
| --------------- | ----- |
| EMA Length      | 9     |
| Take-Profit     | 351.0 pts |
| Stop-Loss       | 141.0 pts |
| Chart Timeframe | 15min |
| Reward:Risk     | 2.489 |

**Results**

| Metric        | Value           |
| ------------- | --------------- |
| Net Points    | 9,570.00        |
| Total Trades  | 189             |
| Win Rate      | 39.15%          |
| Expectancy    | 50.63 pts/trade |
| Profit Factor | 1.586           |
| Max Drawdown  | 1,714.00 pts    |
| Sharpe Ratio  | 0.215           |




AND ==================================================================================================================================================================================================================================



Following the bug correction, the parameter search was re-run from scratch under the corrected model. A hybrid exit approach — close-based confirmation for most stop-loss exits, combined with a wider "emergency" hard stop (a genuine wick-triggered order) to cap the tail-risk of the fastest, worst-case moves — produced the best verified result under honest accounting: a meaningfully more modest, but far more trustworthy, profit figure than the original (incorrect) number, with real drawdown properly accounted for.

A subsequent search specifically targeting "zero losing months across 30 months, reward:risk ratio ≥ 2:1, and max drawdown under a fixed threshold" — a natural next question once realistic risk figures were available — found that **no tested configuration satisfies all three simultaneously**. The closest approaches trade off cleanly against each other (tighter stops reduce drawdown but increase losing months; wider stops do the reverse), which is itself a useful, honest finding: it suggests these three targets may be structurally in tension for this strategy family, not just a matter of more search.

### Final configuration and results — wick-based stop-loss

With the close-based confirmation bug fixed and the exit mechanics now modeled honestly, the search was also re-run using a straightforward **wick-based (intrabar) stop-loss** as a baseline comparison against the hybrid close-based/emergency-stop approach above — i.e., the SL fires as soon as price touches the level intrabar, with no confirmation delay. This removes the confirmation-bar assumption entirely, at the cost of being more exposed to brief noise spikes. The best configuration found under this exit model was:

**Parameters**

| Parameter | Value |
|---|---|
| EMA Length | 16 |
| Take-Profit | 166.0 pts |
| Stop-Loss | 34.0 pts |
| Chart Timeframe | 15min |
| Max Wick Ratio | 0.94 |
| Bull Bounce | 1.0018 |
| Bear Bounce | 0.9982 |

**Results**

| Metric | Value |
|---|---|
| Net Points | 10,359.50 |
| Total Trades | 1,149 |
| Win Rate | 21.76% |
| Expectancy | 9.02 pts/trade |
| Profit Factor | 1.33 |
| Max Drawdown | 1,235.50 pts |
| Sharpe Ratio | 0.110 |
| TP Hits | 250 |
| SL Hits | 899 |

This configuration reflects a materially different risk/reward shape than earlier configurations in this project: a low win rate (~22%) offset by a large reward:risk ratio (166 vs. 34, ~4.9:1), rather than a high win rate with a tight ratio. It is profitable on a Net Points and Profit Factor basis, but the Sharpe Ratio (0.110) is low, indicating substantial volatility in the equity curve relative to its return — consistent with a strategy that loses far more often than it wins and depends on a minority of large winners to stay net-positive. This result is presented as a genuine, honestly-measured backtest output under the corrected exit model, not as the project's recommended configuration; per the "zero losing months / ≥2:1 reward:risk / bounded drawdown" search described above, no single configuration tested — including this one — satisfies all target criteria simultaneously.

## 9. What I'd Do Differently

- **Validate against ground-truth execution data earlier and more often**, not just as a final sanity check. The overshoot bug existed for a significant portion of the project before being caught; earlier, more frequent live-vs-backtest comparison would have surfaced it sooner.
- **Model exit mechanics as carefully as entry logic from the start.** Entry conditions received far more scrutiny (exhaustive grids, robustness checks) than the exact mechanics of how a simulated stop-loss actually resolves — which turned out to matter enormously.
- **Treat "too good" results as a prompt to look harder, not a reason to move faster** — this instinct caught the original overfit 6-month result and the initial 88% win-rate fluke, and is the single habit most responsible for this project's results being trustworthy rather than merely impressive-looking.
