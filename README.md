# Optimal Trade Entry (OTE) Options Strategy — Nifty 50

A directional Nifty 50 **options** backtesting system that combines Smart-Money-Concepts
structure (Fair Value Gaps + Fibonacci confluence) on a higher timeframe with a lower-timeframe
trigger, then executes the signal through a Black-Scholes-based strike/spread selection engine
with realistic transaction costs.

## Strategy Logic

1. **HTF Swing Mapping** (`POI_mapping.py`)
   Detects the most recent confirmed swing high/low on the higher timeframe (default `1hr`)
   using a fractal high/low filter (`left_n`/`right_n` bars) with an ATR displacement check,
   and validates that no later bar has invalidated the swing.

2. **Golden Zone + FVG (the POI)** (`POI_mapping.py`)
   From the swing, computes the 50% and 78.6% Fibonacci retracement levels. Any 3-candle Fair
   Value Gap inside the swing that overlaps this zone is a candidate POI. If multiple candidates
   exist, the largest gap is kept as the `merged_poi` (a `[low, high]` price band), and it's
   discarded if price has since traded through the swing extremes.

3. **LTF Trigger** (`entry_trigger.py`)
   On the lower timeframe (default `5min`), once price taps into the HTF POI, the engine watches
   for a local 3-candle FVG in the direction of the expected reversal, occurring inside the POI
   band. This confirmation fires the entry signal (entry price, direction, take-profit at the
   HTF swing extreme, stop-loss at the POI boundary).

4. **Strike & Route Selection** (`options_execution.py`)
   Given the signal, this module:
   - Builds an implied-vol smile from the option chain snapshot (Black-Scholes inversion) and
     picks the strike `K*` whose delta is closest to `delta_target` (a relative, non-hardcoded
     selection — ~0.6 delta" rather than a fixed strike).
   - Estimates expected reward/risk to TP/SL via a delta-gamma (Taylor) approximation and derives
     an Option Reward:Risk Ratio (ORRR).
   - Routes to a **naked long** or a **vertical spread** (selling a further OTM strike) based on
     two filters: IV rich vs 20-day HV, and ORRR below threshold — spreads are used to cut cost
     when premium is expensive or edge is thin.
   - Sizes the trade strictly in **lots** (`risk_budget / risk_per_contract`, capped at
     `max_lots`), never raw contract counts.

5. **Execution & Cost Model** (`backtest.py`)
   Opens the position at the next available chain snapshot, applies slippage (as a % of quoted
   premium) and per-leg brokerage + exchange transaction fees on both entry and a modeled exit,
   and skips trades where the estimated round-trip friction would eat >99% of the distance to
   target. Positions are closed on stop-loss, take-profit, or a forced square-off before expiry.

## Backtest Results

Sample run over the supplied dataset (`tradesheet.csv`), starting capital ₹1,00,00,000:

| Metric | Value |
|---|---|
| Trades | 15 |
| Win rate | 53.3% |
| Total gross P&L | ₹37,625.00 |
| Total fees | ₹1,374.41 |
| Total slippage cost | ₹7,064.00 |
| Total net P&L | ₹29,186.59 |
| Ending capital | ₹1,00,29,186.59 |
| Total return | 0.29% |
| Average net P&L / trade | ₹1,945.77 |
| Profit factor | 1.50 |
| Max drawdown | −₹42,427.97 (−0.42%) |
| Naked long trades | 8 |
| Vertical spread trades | 7 |
| Average win | ₹10,876.88 |
| Average loss | −₹8,261.21 |
| Win/loss ratio | 1.32 |
| Best trade | ₹42,088.38 |
| Worst trade | −₹18,529.56 |
| Average holding period | ~1 day 6 hours |
