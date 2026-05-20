"""
run_pipeline.py
===============
End-to-end driver. Run `python run_pipeline.py` to execute the whole flow on
synthetic data (works with no network); pass --live to use yfinance instead.

    python run_pipeline.py                      # synthetic, offline
    python run_pipeline.py --live SPY IVV        # real pair via yfinance
"""

from __future__ import annotations
import argparse
import pandas as pd

from data_loader import make_synthetic_pair, load_prices
from models import KalmanHedgeRatio, screen_pairs
from backtester import backtest
from analytics import print_summary, plot_results


def run(prices, y_col, x_col, delta=1e-4, cost_bps=5.0):
    # Screening: confirm the pair is actually cointegrated before trading it.
    res = screen_pairs(prices[[y_col, x_col]], candidates=[(y_col, x_col)])
    print("Cointegration screen:")
    print(res.to_string(index=False))
    print()

    kf = KalmanHedgeRatio(delta=delta)
    kf_out = kf.run_df(prices[y_col], prices[x_col])

    bt = backtest(prices, kf_out, y_col=y_col, x_col=x_col, cost_bps=cost_bps)
    print_summary(bt)
    path = plot_results(bt, kf_out, path="results.png")
    print(f"\nSaved plot to {path}")
    return bt, kf_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--live", nargs=2, metavar=("Y", "X"), default=None,
                   help="Two tickers to pull via yfinance, e.g. --live SPY IVV")
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--cost-bps", type=float, default=5.0)
    args = p.parse_args()

    if args.live:
        y_t, x_t = args.live
        prices = load_prices([y_t, x_t])
        prices = prices.rename(columns={y_t: "Y", x_t: "X"})
        run(prices, "Y", "X", delta=args.delta, cost_bps=args.cost_bps)
    else:
        prices, _ = make_synthetic_pair()
        run(prices, "Y", "X", delta=args.delta, cost_bps=args.cost_bps)


if __name__ == "__main__":
    main()
