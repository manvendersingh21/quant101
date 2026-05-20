"""
data_loader.py
==============
Two ways to get price data into the engine:

1. `load_prices`   -- pull real daily bars from Yahoo via yfinance (your machine).
2. `make_synthetic_pair` -- generate a pair with a KNOWN, time-varying hedge ratio.
   This is not filler. It is the ground truth you test the Kalman filter against:
   if the filter can't recover a beta you generated yourself, the bug is in your
   code, not the market.

The rest of the engine never cares which one produced the DataFrame, so you can
develop and unit-test offline, then flip to live data with a one-line change.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def load_prices(tickers, start="2018-01-01", end=None, column="Close"):
    """Download daily bars for `tickers` and return a wide DataFrame of one
    price column, indexed by date, one column per ticker.

    Requires `yfinance` (pip install yfinance) and network access to Yahoo.
    Kept deliberately thin -- all the interesting logic lives downstream.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Run `pip install yfinance`, or use "
            "make_synthetic_pair() to develop offline."
        ) from e

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False
    )

    # yfinance returns a column MultiIndex (field, ticker) for multiple tickers.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw[column].copy()
    else:  # single ticker -> flat columns
        prices = raw[[column]].copy()
        prices.columns = [tickers if isinstance(tickers, str) else tickers[0]]

    prices = prices.dropna(how="all").ffill().dropna()
    return prices


def make_synthetic_pair(
    n=1500,
    beta_start=1.2,
    beta_end=1.8,
    alpha=2.0,
    spread_vol=0.5,
    spread_ar=0.92,
    base_price=50.0,
    base_vol=0.8,
    seed=7,
):
    """Generate two price series y, x related by a SLOWLY DRIFTING hedge ratio.

        y_t = beta_t * x_t + alpha + spread_t

    where beta_t walks linearly from `beta_start` to `beta_end` (this is exactly
    the regime drift a static OLS hedge ratio gets wrong and a Kalman filter is
    supposed to track) and spread_t is a mean-reverting AR(1) process -- i.e. the
    pair is genuinely cointegrated by construction.

    Returns
    -------
    prices : DataFrame with columns ['Y', 'X']
    truth  : DataFrame with the true beta_t and spread_t, for validation.
    """
    rng = np.random.default_rng(seed)

    # x is a random-walk price (a non-stationary I(1) series, like a real stock).
    x_returns = rng.normal(0, base_vol, n)
    x = base_price + np.cumsum(x_returns)
    x = np.maximum(x, 1.0)  # keep prices positive

    # beta drifts linearly; this is the "regime shift" the doc warns about.
    beta = np.linspace(beta_start, beta_end, n)

    # spread is stationary AR(1) -> mean-reverting, which is what makes y,x cointegrated.
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread_ar * spread[t - 1] + rng.normal(0, spread_vol)

    y = beta * x + alpha + spread

    idx = pd.bdate_range("2018-01-01", periods=n)
    prices = pd.DataFrame({"Y": y, "X": x}, index=idx)
    truth = pd.DataFrame({"beta": beta, "spread": spread}, index=idx)
    return prices, truth


if __name__ == "__main__":
    prices, truth = make_synthetic_pair()
    print(prices.tail())
    print("\nTrue beta drifts from %.2f to %.2f"
          % (truth["beta"].iloc[0], truth["beta"].iloc[-1]))
