# Global Swing Desk — v3 (Real Data Edition)

A professional swing trading dashboard for Forex and global equities (US, Europe, Asia).
Pulls **real data from Yahoo Finance**, computes 8+ technical indicators, identifies
high-probability setups using proven strategies (SEPA/VCP, CANSLIM, Stage Analysis,
Connors RSI, Trend Pullback), and **ranks every setup 0–100** so you know exactly
which trades to prioritize.

## Architecture

This is a two-piece system, same pattern used on real trading desks:

```
+---------------------+         +----------------------+
|  fetch_signals.py   |  data   |   dashboard.html     |
|  (data engine)      | ------> |   (visualization)    |
|                     |  .json  |                      |
|  - Yahoo Finance    |         |  - Candlestick chart |
|  - Indicators       |         |  - Ranked signals    |
|  - Setup detection  |         |  - Risk calculator   |
|  - Priority scoring |         |  - Checklist         |
+---------------------+         +----------------------+
```

The data engine runs on your machine where Yahoo Finance is accessible.
The dashboard reads the resulting JSON file in your browser.

## Layout

The dashboard uses a focused three-column workspace:

- **Left column — Signals & live intel**: top-ranked trades, currency/sector
  strength heatmap, and a scrollable table of every active setup. This is your
  scanning surface — what's hot, what's primed, what to look at next.
- **Middle column — Charts & context**: the selected instrument's price,
  candle chart with EMAs and entry/stop/target lines, priority-score
  breakdown, trade levels, and strategy description. This is where the
  decision gets made.
- **Right column — Execution tools**: position-size calculator, pre-trade
  checklist, and a local journal. Once you've decided what to take, this is
  what you act through.

## Quick Start

### 1. Install dependencies

```bash
pip install yfinance pandas numpy
```

### 2. Run the data engine

```bash
python3 fetch_signals.py
```

This will:
- Fetch 1 year of daily OHLCV for ~66 instruments (FX, US large/small cap, EU, Asia, EM)
- Compute EMA(8, 21, 50), SMA(50, 150, 200), ATR(14), RSI(14), RSI(2), ADX(14)
- Run Stage Analysis (Weinstein), compute RS Rating (O'Neil)
- Detect setups: VCP, CANSLIM breakouts, pullbacks, Connors mean reversion
- Score each setup 0-100 across 6 dimensions
- Write `data.json`

### 3. Open the dashboard

Open `dashboard.html` in your browser. **Note:** Browsers block `file://` from
reading local JSON files for security. Use a simple local server:

```bash
python3 -m http.server 8000
# Then open http://localhost:8000/dashboard.html
```

### 4. Refresh data periodically

Re-run `python3 fetch_signals.py` whenever you want fresh data.
For automatic updates, set a cron job:

```bash
# Every 30 minutes during market hours, weekdays only
*/30 9-16 * * 1-5 cd /path/to/dashboard && python3 fetch_signals.py
```

## The Trade Priority Score (the ranking system)

Every setup gets scored on six factors, totaling 0-100:

| Component         | Max | What it measures                                           |
|-------------------|-----|------------------------------------------------------------|
| Trend Quality     | 25  | Stage 2/4, MA alignment (50>150>200), ADX strength         |
| Relative Strength | 20  | RS rating vs benchmark, multi-period momentum              |
| Setup Quality     | 20  | Pattern type (VCP, base, pullback), volume confirmation    |
| Risk/Reward       | 15  | R:R ratio (3.0+=15, 2.5+=13, 2.0+=10, 1.5+=5, less=0)      |
| Momentum          | 10  | ROC, distance from 52w high, RSI position, EMA8 alignment  |
| Catalyst/Timing   | 10  | Earnings proximity (-5d to +10d is post-earnings drift)    |

**How to use it:**
- **80–100**: Top-tier. Size aggressively (up to your 1% risk cap).
- **70–79**: High conviction. Size normally.
- **50–69**: Marginal. Take only the best in this range, half-size.
- **Below 50**: Skip. Watch list only.

## Strategies Implemented

### Equity (Stocks)

- **SEPA / VCP** (Mark Minervini) — Stage 2 trend + Volatility Contraction Pattern.
- **CANSLIM Breakout** (William O'Neil) — Breakout from flat base / cup-with-handle on volume.
- **Pullback to 21 EMA** — In a Stage 2 uptrend, buying the dip to the institutional zone.
- **Connors RSI(2)** (Larry Connors) — Quality stock above 200 SMA with RSI(2) < 10.

### FX (Forex)

- **Trend Pullback to 21 EMA** — Buying pullbacks in an established trend.
- **Structure Break** — Breakout of 20-day range with volume/ATR expansion.

## Stage Analysis (Weinstein)

Every stock is classified into a stage. **Trade Stage 2 longs and Stage 4 shorts only.**

- **Stage 1 (Basing)** — Sideways. Sit out.
- **Stage 2 (Advancing)** — Above rising 30-week MA. **BUY ZONE.**
- **Stage 3 (Topping)** — Above MA but flattening. **WATCH.**
- **Stage 4 (Declining)** — Below falling 30-week MA. **SHORT ZONE.**

## Universe (default)

- **FX**: 9 majors and crosses
- **US**: 16 large caps
- **Europe**: 12 leaders
- **Asia**: 9 leaders
- **US Small Cap**: 10 Russell 2000 leaders (RS-ranked vs IWM)
- **Emerging Markets**: 10 leaders across Brazil, India, China, LatAm, SEA (RS-ranked vs EEM)

Edit `UNIVERSE` at the top of `fetch_signals.py` to add or remove instruments.

## Risk Management Rules (non-negotiable)

1. **Never risk more than 1% per trade**.
2. **Never risk more than 3% across all open positions**.
3. **Correlated trades count as one position**.
4. **Minimum 2:1 R:R**.
5. **Stop behind structure**, not arbitrary % levels.
6. **No earnings within 5 days** for stocks.
7. **No average down. Ever.**

## Validating the priority score (backtest)

The ranking is only useful if higher scores really do win more. Run the backtest
to check that on your own data:

```bash
python3 backtest.py
```

This walks every instrument forward bar by bar, computes the priority score
that would have been issued at each historical date (using only data available
up to that point — no lookahead), simulates the trade (exit on stop, target,
or 40-bar timeout) and writes `backtest.json`. The dashboard's **Score
Validation** panel then shows, per bucket (50-59 / 60-69 / 70-79 / 80+):

- N (sample size — small N means don't read too much into the number)
- Win rate (% of trades that hit target before stop, plus timeout P&L)
- Expectancy (average R-multiple per trade — this is the one that matters)
- Outcomes breakdown (target hits / stop hits / timeouts) and avg days held

If 80+ doesn't beat 50-59 on expectancy, the score isn't earning its keep and
the weights in `compute_signal` need tuning. The backtest is slow (~15–30 min
for the full universe) — run it occasionally, not every session.

## Files

- `fetch_signals.py` — The data engine. Run on your machine.
- `backtest.py` — Validates the priority score against historical outcomes.
- `dashboard.html` — The visualization. Open in browser.
- `data.json` — Generated by the engine. Read by the dashboard.
- `backtest.json` — Generated by the backtest. Read by the dashboard (optional).
- `README.md` — This file.

## Disclaimer

This is a decision-support tool, not investment advice. Past performance does not
guarantee future results. **Always do your own analysis and never trade money you
can't afford to lose.** Stay disciplined.
