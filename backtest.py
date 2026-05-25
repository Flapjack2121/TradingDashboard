"""
=============================================================================
GLOBAL SWING DESK - Priority Score Backtest
=============================================================================
Walks each instrument forward bar by bar, computes the priority score that
would have been issued at that historical date (using only data available
up to that point), simulates the trade (exit on stop, target, or 40-bar
timeout), and writes backtest.json with stats grouped by score bucket.

Use this to answer: "do 80+ scores actually win more than 50-60s?"

Run:    python3 backtest.py
Output: backtest.json (loaded by dashboard.html)
=============================================================================
"""

import json
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

from fetch_signals import (
    UNIVERSE,
    BENCHMARK, BENCH_EU, BENCH_ASIA, BENCH_SMALL, BENCH_EM,
    fetch_one, compute_signal,
)

# --- Backtest parameters ---------------------------------------------------
HOLD_BARS   = 40     # max trading days to hold before timeout exit
WARMUP_BARS = 250    # need a year of history before we issue signals
MIN_SCORE   = 50     # ignore any signal under this — matches the README's "skip" floor
HISTORY     = '3y'   # pulled from Yahoo; gives ~2y of usable backtest after warmup
STEP        = 1      # evaluate every Nth bar (1 = every day; 5 = weekly)

# Bucket boundaries (inclusive on both sides)
BUCKETS = [
    (50, 59, '50-59'),
    (60, 69, '60-69'),
    (70, 79, '70-79'),
    (80, 100, '80+'),
]


def simulate_trade(future_df, bias, entry, stop, target, rr):
    """
    Walk forward through future bars. Exit at first stop or target touch.
    If neither hits within HOLD_BARS, mark to the final close.
    Returns (outcome, r_multiple, bars_held).
    """
    risk = abs(entry - stop)
    if risk <= 0 or len(future_df) == 0:
        return 'invalid', 0.0, 0

    for j in range(len(future_df)):
        row = future_df.iloc[j]
        hi, lo = float(row['High']), float(row['Low'])
        if bias == 'long':
            # Conservative tiebreak: if a bar straddles both, assume stop hit first.
            if lo <= stop:
                return 'stop', -1.0, j + 1
            if hi >= target:
                return 'target', float(rr), j + 1
        else:
            if hi >= stop:
                return 'stop', -1.0, j + 1
            if lo <= target:
                return 'target', float(rr), j + 1

    final_close = float(future_df['Close'].iloc[-1])
    if bias == 'long':
        r = (final_close - entry) / risk
    else:
        r = (entry - final_close) / risk
    return 'timeout', round(r, 2), len(future_df)


def backtest_instrument(symbol_info, df, bench_df, asset_class):
    """Walk df bar by bar; collect every triggered signal and its outcome."""
    trades = []
    if df is None or len(df) < WARMUP_BARS + HOLD_BARS + 1:
        return trades

    # Strip any cached earnings date — we can't reliably know past earnings dates,
    # and using a current one would create lookahead bias.
    item = {k: v for k, v in symbol_info.items() if k != 'earnings_days'}

    for i in range(WARMUP_BARS, len(df) - HOLD_BARS, STEP):
        sub = df.iloc[:i + 1]
        if bench_df is not None:
            bench_slice = bench_df[bench_df.index <= df.index[i]]
            if len(bench_slice) < 60:
                bench_slice = None
        else:
            bench_slice = None

        try:
            sig = compute_signal(item, sub, bench_slice, asset_class)
        except Exception:
            continue

        if not sig or sig['bias'] == 'flat':
            continue
        if not sig.get('rr') or sig['rr'] < 1.5:
            continue
        if sig['priority_score'] < MIN_SCORE:
            continue

        future = df.iloc[i + 1:i + 1 + HOLD_BARS]
        outcome, r_mult, days = simulate_trade(
            future, sig['bias'], sig['entry'], sig['stop'], sig['target'], sig['rr']
        )
        if outcome == 'invalid':
            continue

        trades.append({
            'date': df.index[i].strftime('%Y-%m-%d'),
            'symbol': sig['symbol'],
            'asset_class': asset_class,
            'score': int(sig['priority_score']),
            'bias': sig['bias'],
            'strategy': sig['strategy'],
            'rr': float(sig['rr']),
            'r_multiple': float(r_mult),
            'days_held': int(days),
            'outcome': outcome,
        })
    return trades


def aggregate(trades):
    """Group trades into score buckets and compute summary stats."""
    out = []
    for lo, hi, label in BUCKETS:
        bt = [t for t in trades if lo <= t['score'] <= hi]
        n = len(bt)
        if n == 0:
            out.append({
                'label': label, 'n': 0,
                'win_rate': None, 'expectancy': None,
                'avg_days': None, 'wins': 0, 'losses': 0,
                'target_hits': 0, 'stop_hits': 0, 'timeouts': 0,
            })
            continue
        wins   = sum(1 for t in bt if t['r_multiple'] > 0)
        losses = sum(1 for t in bt if t['r_multiple'] <= 0)
        target_hits = sum(1 for t in bt if t['outcome'] == 'target')
        stop_hits   = sum(1 for t in bt if t['outcome'] == 'stop')
        timeouts    = sum(1 for t in bt if t['outcome'] == 'timeout')
        out.append({
            'label': label,
            'n': n,
            'win_rate': round(wins / n * 100, 1),
            'expectancy': round(sum(t['r_multiple'] for t in bt) / n, 2),
            'avg_days': round(sum(t['days_held'] for t in bt) / n, 1),
            'wins': wins,
            'losses': losses,
            'target_hits': target_hits,
            'stop_hits': stop_hits,
            'timeouts': timeouts,
        })
    return out


def main():
    print("=" * 70)
    print("PRIORITY SCORE BACKTEST")
    print("=" * 70)
    print(f"  Hold: {HOLD_BARS} bars   Warmup: {WARMUP_BARS} bars   "
          f"Min score: {MIN_SCORE}   History: {HISTORY}   Step: {STEP}")
    print("  Note: this is slow (~15-30 min for the full universe).")

    print("\n[1/N] Fetching benchmarks...")
    bench = {
        'us':       fetch_one(BENCHMARK,   period=HISTORY),
        'eu':       fetch_one(BENCH_EU,    period=HISTORY),
        'asia':     fetch_one(BENCH_ASIA,  period=HISTORY),
        'smallcap': fetch_one(BENCH_SMALL, period=HISTORY),
        'em':       fetch_one(BENCH_EM,    period=HISTORY),
    }

    all_trades = []
    asset_classes = list(UNIVERSE.keys())
    total_steps = len(asset_classes) + 1
    for idx, asset_class in enumerate(asset_classes, start=2):
        items = UNIVERSE[asset_class]
        bench_df = bench.get(asset_class)  # None for fx, that's expected
        print(f"\n[{idx}/{total_steps}] {asset_class.upper()} ({len(items)} instruments)")
        for item in items:
            sym = item['symbol']
            print(f"  -> {sym:12s}", end=' ', flush=True)
            df = fetch_one(sym, period=HISTORY)
            if df is None:
                print("(skipped - no data)")
                continue
            trades = backtest_instrument(item, df, bench_df, asset_class)
            wins = sum(1 for t in trades if t['r_multiple'] > 0)
            print(f"{len(trades):4d} signals, {wins} wins")
            all_trades.extend(trades)

    buckets = aggregate(all_trades)

    print("\n" + "=" * 70)
    print(f"RESULTS BY SCORE BUCKET ({len(all_trades)} total trades)")
    print("=" * 70)
    print(f"  {'Bucket':<10s} {'N':>5s} {'Win%':>7s} {'Exp R':>7s} {'Days':>6s}  Outcome (T/S/X)")
    for b in buckets:
        if b['n'] == 0:
            print(f"  {b['label']:<10s} {'-':>5s} {'-':>7s} {'-':>7s} {'-':>6s}")
            continue
        outcome_str = f"{b['target_hits']}/{b['stop_hits']}/{b['timeouts']}"
        print(f"  {b['label']:<10s} {b['n']:>5d} {b['win_rate']:>6.1f}% "
              f"{b['expectancy']:>+6.2f}R {b['avg_days']:>5.1f}d  {outcome_str}")
    print("=" * 70)

    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'parameters': {
            'hold_bars': HOLD_BARS,
            'warmup_bars': WARMUP_BARS,
            'min_score': MIN_SCORE,
            'history_period': HISTORY,
            'step': STEP,
        },
        'total_trades': len(all_trades),
        'buckets': buckets,
    }

    with open('backtest.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nOK backtest.json written ({len(all_trades)} trades)")
    print("   Refresh the dashboard to see the Score Validation panel.\n")


if __name__ == '__main__':
    main()
