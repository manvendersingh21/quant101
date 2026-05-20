"""
analytics.py
============
Performance metrics and plots (Day 5). Everything here takes the backtest
DataFrame (which has a 'ret_net' column and an 'equity' curve) and summarizes it.

The metrics are deliberately the standard ones a reviewer expects, computed the
standard way, with the annualization factor exposed so you don't have to guess
whether a number is daily or annual.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods=TRADING_DAYS) -> float:
    """Geometric annualized return from a series of per-period returns."""
    growth = (1.0 + returns).prod()
    n = len(returns)
    if n == 0 or growth <= 0:
        return np.nan
    return growth ** (periods / n) - 1.0


def annualized_vol(returns: pd.Series, periods=TRADING_DAYS) -> float:
    return returns.std(ddof=1) * np.sqrt(periods)


def sharpe_ratio(returns: pd.Series, rf=0.0, periods=TRADING_DAYS) -> float:
    """Annualized Sharpe. rf is an annual risk-free rate, converted to per-period."""
    excess = returns - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.sqrt(periods) * excess.mean() / sd


def sortino_ratio(returns: pd.Series, rf=0.0, periods=TRADING_DAYS) -> float:
    """Like Sharpe but only penalizes DOWNSIDE volatility (returns below 0)."""
    excess = returns - rf / periods
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).mean()) if len(downside) else np.nan
    if not dd or np.isnan(dd):
        return np.nan
    return np.sqrt(periods) * excess.mean() / dd


def max_drawdown(equity: pd.Series):
    """Return (max_drawdown_fraction, peak_date, trough_date, duration_in_bars).

    Drawdown is measured from the running peak of the equity curve.
    """
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    trough = drawdown.idxmin()
    max_dd = drawdown.min()
    # Peak is the last time we were at the running max before the trough.
    peak = equity.loc[:trough].idxmax()
    duration = equity.index.get_loc(trough) - equity.index.get_loc(peak)
    return max_dd, peak, trough, duration


def summarize(bt: pd.DataFrame, ret_col="ret_net", equity_col="equity") -> dict:
    """One-stop performance summary dict, ready to print or log.

    Note: equity here is ADDITIVE (1 + cumulative sum of returns), so total
    return is read straight off the curve and 'annualized return' is the mean
    per-period return scaled by TRADING_DAYS (arithmetic), consistent with that.
    """
    r = bt[ret_col].dropna()
    eq = bt[equity_col].dropna()
    max_dd, peak, trough, dur = max_drawdown(eq)
    return {
        "Total return": eq.iloc[-1] - 1.0,
        "Annualized return": r.mean() * TRADING_DAYS,
        "Annualized vol": annualized_vol(r),
        "Sharpe": sharpe_ratio(r),
        "Sortino": sortino_ratio(r),
        "Max drawdown": max_dd,
        "Max DD duration (bars)": int(dur),
        "Num trades": int((bt["position"].diff().abs() > 0).sum())
        if "position" in bt else np.nan,
    }


def print_summary(bt: pd.DataFrame):
    s = summarize(bt)
    print("Performance summary")
    print("-" * 40)
    for k, v in s.items():
        if isinstance(v, float):
            if "return" in k.lower() or "vol" in k.lower() or "drawdown" in k.lower():
                print(f"{k:<28} {v:>10.2%}")
            else:
                print(f"{k:<28} {v:>10.2f}")
        else:
            print(f"{k:<28} {v:>10}")


def plot_results(bt: pd.DataFrame, kf_output=None, path="results.png"):
    """Save a 3-panel figure: equity curve, z-score with bands, hedge ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 3 if kf_output is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 3.2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(bt.index, bt["equity"], color="#1b3a5c", lw=1.4)
    axes[0].set_ylabel("Equity")
    axes[0].set_title("Strategy equity curve")
    axes[0].grid(alpha=0.3)

    if kf_output is not None:
        z = kf_output["spread_z"]
        axes[1].plot(z.index, z, color="#555", lw=0.8)
        for lvl, c in [(2, "#c0392b"), (-2, "#c0392b"), (0, "#888")]:
            axes[1].axhline(lvl, color=c, ls="--", lw=0.8)
        # Clip to the tradeable range; the early convergence transient can be huge
        # and would otherwise flatten the part of the series we actually trade.
        zclip = z.iloc[60:]
        lim = max(5.0, float(zclip.abs().quantile(0.999)) * 1.1)
        axes[1].set_ylim(-lim, lim)
        axes[1].set_ylabel("Spread z-score")
        axes[1].set_title("Kalman spread z-score with +/-2 entry bands")
        axes[1].grid(alpha=0.3)

        axes[2].plot(kf_output.index, kf_output["beta"], color="#196f3d", lw=1.2)
        axes[2].set_ylabel("Hedge ratio beta_t")
        axes[2].set_title("Kalman-filtered hedge ratio (time-varying)")
        axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
