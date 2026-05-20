"""
models.py
=========
The statistical core: (1) honest cointegration screening, and (2) a Kalman
filter that tracks a time-varying hedge ratio.

Two design decisions worth reading before you trust this code:

* SCREENING IS CONFIRMATORY, NOT EXPLORATORY.  `screen_pairs` will test whatever
  candidate pairs you hand it, but it also reports how many pairs you tested so
  you can reason about multiple comparisons. Testing all 105 pairs in a basket
  of 15 and keeping whatever clears p<0.05 is how you "discover" cointegration
  that is really just noise -- at p<0.05 you expect ~5 false positives from 105
  independent tests. Feed it economically-motivated candidates (same sector /
  same underlying) and treat the p-value as confirmation.

* THE KALMAN FILTER IS A REGRESSION THAT UPDATES EVERY DAY.  Instead of one OLS
  hedge ratio for a whole window, the state [beta_t, alpha_t] evolves over time
  and is re-estimated each step from the latest price. No lookback window, so no
  lookback-window bias.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller


# ---------------------------------------------------------------------------
# 1. Cointegration screening (Day 1)
# ---------------------------------------------------------------------------

def adf_pvalue(series) -> float:
    """Augmented Dickey-Fuller p-value. Low p => series is stationary
    (mean-reverting), which is what we want for a spread."""
    series = np.asarray(series, dtype=float)
    series = series[~np.isnan(series)]
    return adfuller(series, autolag="AIC")[1]


def screen_pairs(prices: pd.DataFrame, candidates=None, pvalue_threshold=0.05):
    """Run the Engle-Granger cointegration test on candidate pairs.

    Parameters
    ----------
    prices : wide DataFrame, one column per asset.
    candidates : list of (a, b) column-name tuples to test. If None, ALL
        combinations are tested -- convenient, but see the multiple-comparisons
        warning in the module docstring.
    pvalue_threshold : significance cutoff for flagging a pair.

    Returns
    -------
    DataFrame sorted by p-value, with a `passes` flag and a note on how many
    tests were run (so you can apply a Bonferroni-style mental discount).
    """
    cols = list(prices.columns)
    if candidates is None:
        candidates = list(combinations(cols, 2))

    rows = []
    for a, b in candidates:
        s1, s2 = prices[a].dropna(), prices[b].dropna()
        joined = pd.concat([s1, s2], axis=1).dropna()
        if len(joined) < 50:
            continue
        # Engle-Granger: regress one on the other, ADF-test the residual.
        _, pval, _ = coint(joined[a], joined[b])
        rows.append({"asset_a": a, "asset_b": b, "coint_pvalue": pval})

    result = pd.DataFrame(rows).sort_values("coint_pvalue").reset_index(drop=True)
    if not result.empty:
        result["passes"] = result["coint_pvalue"] < pvalue_threshold
        n_tests = len(result)
        # Bonferroni-adjusted threshold, reported as context not gospel.
        result.attrs["n_tests"] = n_tests
        result.attrs["bonferroni_threshold"] = pvalue_threshold / max(n_tests, 1)
    return result


# ---------------------------------------------------------------------------
# 2. Kalman filter hedge-ratio tracker (Day 2)
# ---------------------------------------------------------------------------

class KalmanHedgeRatio:
    """Track a time-varying hedge ratio with a Kalman filter.

    State-space model
    -----------------
        observation:  y_t = [x_t, 1] . [beta_t, alpha_t]^T + e_t,   e_t ~ N(0, R)
        state evo:    [beta_t, alpha_t] = [beta_{t-1}, alpha_{t-1}] + w_t,
                                                            w_t ~ N(0, Q)

    So the "regression coefficients" (hedge ratio beta and intercept alpha) are a
    hidden 2-D random walk, and each day's observation of y given x nudges our
    estimate. `delta` controls how fast beta is allowed to move (process noise);
    keep it SMALL (1e-5..1e-4) so the hedge ratio doesn't chase daily noise --
    exactly the tip in the project plan.

    Attributes after .run()
    ------------------------
    beta, alpha : Series of filtered state estimates.
    spread      : the prediction error e_t (a.k.a. innovation) -- the clean,
                  no-look-ahead signal we trade on.
    innov_var   : the filter's own variance of that prediction error, used to
                  standardize the spread without a rolling window.
    """

    def __init__(self, delta=1e-4, obs_cov=1.0):
        self.delta = float(delta)
        self.obs_cov = float(obs_cov)

    def run(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        n = len(y)

        # Process-noise covariance Q. The delta/(1-delta) parameterization is the
        # standard trick (see Chan, "Algorithmic Trading") to express "how much
        # state drift per step" as a single scalar.
        Q = self.delta / (1.0 - self.delta) * np.eye(2)
        R = self.obs_cov  # scalar observation noise

        beta = np.zeros((n, 2))      # state mean [beta, alpha]
        P = np.zeros((2, 2))         # state covariance, start diffuse-ish at 0
        theta = np.zeros(2)          # running state estimate

        spread = np.zeros(n)         # prediction error e_t (the innovation)
        innov_var = np.zeros(n)      # variance of e_t

        for t in range(n):
            # --- Predict ---  state is a random walk, so mean unchanged, cov grows.
            P = P + Q
            obs = np.array([x[t], 1.0])           # observation matrix H_t

            # --- Innovation --- prediction error of y_t BEFORE seeing it.
            yhat = obs @ theta
            e = y[t] - yhat                        # this is the spread, look-ahead-free
            S = obs @ P @ obs + R                  # innovation variance

            # --- Update --- Kalman gain, then correct the state.
            K = (P @ obs) / S
            theta = theta + K * e
            P = P - np.outer(K, obs) @ P

            beta[t] = theta
            spread[t] = e
            innov_var[t] = S

        return beta, spread, innov_var

    def run_df(self, y: pd.Series, x: pd.Series) -> pd.DataFrame:
        """Same as run() but returns a tidy DataFrame aligned to y's index."""
        beta, spread, innov_var = self.run(y, x)
        return pd.DataFrame(
            {
                "beta": beta[:, 0],
                "alpha": beta[:, 1],
                "spread": spread,
                "innov_var": innov_var,
                "spread_z": spread / np.sqrt(innov_var),
            },
            index=y.index,
        )


if __name__ == "__main__":
    from data_loader import make_synthetic_pair

    prices, truth = make_synthetic_pair()
    kf = KalmanHedgeRatio(delta=1e-4)
    out = kf.run_df(prices["Y"], prices["X"])
    # Compare recovered beta to the truth on the back half (after filter converges).
    half = len(out) // 2
    err = (out["beta"].iloc[half:] - truth["beta"].iloc[half:]).abs().mean()
    print("Mean |beta_hat - beta_true| (2nd half): %.4f" % err)
    print(out.tail())
