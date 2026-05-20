"""
backtester.py
=============
Turn the Kalman spread into trades, then into an equity curve -- with the two
frictions that separate an honest backtest from a fantasy: transaction costs and
a de-cointegration stop-loss (Days 3-4).

Signal logic (z = spread / sqrt(innovation_variance)):
    enter long  spread  when z < -entry_z   (buy Y, sell beta*X)
    enter short spread  when z > +entry_z   (sell Y, buy beta*X)
    exit                when z crosses back through `exit_z` (default 0)
    stop-out            when |z| > stop_z    (pair has likely broken; bail)

LOOK-AHEAD NOTE: we trade the signal on the NEXT bar's return. The z-score at
time t is computed from the Kalman innovation at t, but the position it implies
only earns the return from t -> t+1. Acting on the same bar that generated the
signal is the most common silent backtest cheat; we shift to avoid it.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def generate_signals(z: pd.Series, entry_z=2.0, exit_z=0.0, stop_z=4.0) -> pd.Series:
    """Stateful position generator. Returns a Series in {-1, 0, +1}:
    +1 = long the spread, -1 = short the spread, 0 = flat.

    Implemented as an explicit state machine (not vectorized) because entry/exit
    depend on the PATH of z, not just its current value -- you stay in a trade
    until z reverts, you don't re-evaluate from scratch each bar.
    """
    pos = np.zeros(len(z))
    state = 0
    zv = z.values
    for t in range(len(zv)):
        zt = zv[t]
        if state == 0:  # flat: look for entry
            if zt < -entry_z:
                state = 1
            elif zt > entry_z:
                state = -1
        elif state == 1:  # long spread: exit on revert-up through exit_z, or stop
            if zt >= exit_z or zt < -stop_z:
                state = 0
        elif state == -1:  # short spread: exit on revert-down, or stop
            if zt <= exit_z or zt > stop_z:
                state = 0
        pos[t] = state
    return pd.Series(pos, index=z.index, name="position")


def backtest(
    prices: pd.DataFrame,
    kf_output: pd.DataFrame,
    y_col="Y",
    x_col="X",
    entry_z=2.0,
    exit_z=0.0,
    stop_z=4.0,
    cost_bps=5.0,
    burn_in=60,
):
    """Run the spread backtest.

    Parameters
    ----------
    prices    : DataFrame with the two price columns.
    kf_output : output of KalmanHedgeRatio.run_df (needs 'beta' and 'spread_z').
    cost_bps  : round-trip-agnostic per-trade cost in basis points, charged on
                the notional traded each time the position changes.
    burn_in   : bars to skip while the filter converges (no trading before this).

    Returns
    -------
    DataFrame with positions, spread P&L, costs, and a cumulative equity curve
    (in return space, starting at 1.0).
    """
    df = prices.copy()
    df["beta"] = kf_output["beta"]
    df["z"] = kf_output["spread_z"]

    # Don't trade during burn-in: force z into the dead zone so no signal fires.
    z = df["z"].copy()
    z.iloc[:burn_in] = 0.0

    df["position"] = generate_signals(z, entry_z, exit_z, stop_z)

    # ---- P&L accounting in DOLLAR terms on a fixed capital base ----
    # Holding +1 "spread unit" = long 1 share Y, short beta shares X.
    # Dollar P&L over t-1 -> t is position_{t-1} * (dY_t - beta_{t-1} * dX_t).
    dy = df[y_col].diff()
    dx = df[x_col].diff()
    df["pnl_gross"] = (df["position"].shift(1) * (dy - df["beta"].shift(1) * dx)).fillna(0.0)

    # Transaction cost in DOLLARS: cost_bps charged on the dollar notional traded
    # when the position changes. Notional of one spread unit = |Y| + |beta|*|X|.
    notional = (df[y_col].abs() + df["beta"].abs() * df[x_col].abs())
    turnover_units = df["position"].diff().abs().fillna(0.0)
    df["cost"] = turnover_units * notional.shift(1).fillna(notional.iloc[0]) * (cost_bps / 1e4)

    df["pnl_net"] = df["pnl_gross"] - df["cost"]

    # Dollar-neutral spread strategies are conventionally tracked with an ADDITIVE
    # equity curve (cumulative sum of per-period returns on fixed capital), not a
    # compounded one -- there's no reinvestment of a growing base here, so
    # compounding would massively overstate results.
    capital = float(notional.median())
    df["ret_net"] = df["pnl_net"] / capital
    df["ret_gross"] = df["pnl_gross"] / capital

    df["equity"] = 1.0 + df["ret_net"].cumsum()
    df["n_trades"] = (turnover_units > 0).cumsum()
    return df


if __name__ == "__main__":
    from data_loader import make_synthetic_pair
    from models import KalmanHedgeRatio

    prices, _ = make_synthetic_pair()
    kf = KalmanHedgeRatio()
    out = kf.run_df(prices["Y"], prices["X"])
    bt = backtest(prices, out)
    print("Final equity: %.3f" % bt["equity"].iloc[-1])
    print("Total position changes: %d" % int((bt["position"].diff().abs() > 0).sum()))
