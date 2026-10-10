"""
Volatility Anomaly Score (0-100).

GOAL
----
Flag options whose implied volatility looks out of line with comparable
options on the same underlying. A flag means "worth a closer look". It does
NOT mean the option is mispriced or that a trade would make money.

THE THREE COMPARISONS
---------------------
(A) Nearby strikes (weight 55%)
    Take up to 3 strikes on each side in the same expiration (excluding the
    option itself), draw a straight trend line of IV against log-moneyness
    through them, and read off the IV the line predicts at this strike.
    A line (rather than a plain average) respects the downward skew, so the
    first and last strikes in the chain are not unfairly flagged.
(B) Neighboring expirations (weight 30%)
    The previous and next expirations may sit at a different overall IV
    level, so we compare SHAPE: how far this option sits above its own
    expiry's at-the-money IV, versus how far the same moneyness sits above
    ATM in the neighboring expirations' smoothed smiles.
(C) Rest of the chain (weight 15%)
    Robust z-score of log(IV / 30-day realized vol) across every usable
    option on the underlying. Note: realized vol is the same number for every
    option on a stock, so it cannot by itself tell options apart. That is why
    the IV - RV gap is shown as context but carries little weight in the score.

TURNING DIFFERENCES INTO z-SCORES
---------------------------------
A raw difference of 2 vol points is huge on a calm, smooth smile and tiny on
a noisy one. So each difference is divided by the "typical" difference in
that expiration, measured robustly as 1.4826 x median absolute deviation
(MAD). For normally distributed data 1.4826 x MAD equals the standard
deviation, but unlike the standard deviation, the MAD is not inflated by the
very outliers we are trying to find.

    z = difference / max(1.4826 x MAD, 0.5 vol points)

The floor stops a perfectly smooth smile from turning tiny wiggles into huge z.

COMBINING
---------
    composite = weighted average of |z_A|, |z_B|, |z_C| (missing ones skipped)
    score     = 100 x min(composite / 4, 1)

A combined 4-standard-deviation outlier scores 100. Direction ("rich" =
IV above neighbors, "cheap" = below) comes from the sign of z_A (or z_B).

UNIVERSE
--------
Only liquid quotes with a solved IV, only out-of-the-money options (their
prices are mostly time value, so their IVs are reliable), at least a few days
to expiry, and strikes within ~±35% of spot.

WHERE IT FAILS
--------------
Earnings dates, dividends, hard-to-borrow stocks and American early exercise
create legitimate kinks; stale quotes create fake ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.analytics.surface import interpolate_iv_at

MAD_TO_STD = 1.4826


def robust_scale(x: pd.Series, floor: float) -> float:
    x = x.dropna()
    if len(x) < 3:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    return max(MAD_TO_STD * mad, floor)


def scoring_universe(chains: pd.DataFrame, s: config.AnomalySettings = config.SETTINGS.anomaly,
                     min_dte: int = config.SETTINGS.liquidity.min_dte) -> pd.DataFrame:
    if chains.empty:
        return chains.copy()
    u = chains[(chains["iv_status"] == "ok") & chains["is_liquid"].astype(bool) & chains["is_otm"].astype(bool)
               & (chains["dte"] >= min_dte) & (chains["log_moneyness"].abs() <= s.max_abs_log_moneyness)]
    return u.sort_values(["expiration", "strike"]).reset_index(drop=True)


def _strike_neighbor_component(u: pd.DataFrame, k: int, floor: float) -> pd.DataFrame:
    u = u.copy()
    u["neighbor_iv"] = np.nan
    for _, idx in u.groupby("expiration").groups.items():
        g = u.loc[idx].sort_values("strike")
        ivs, xs = g["computed_iv"].to_numpy(), g["log_moneyness"].to_numpy()
        est = []
        for i in range(len(ivs)):
            sel = np.r_[max(0, i - k):i, i + 1:min(len(ivs), i + 1 + k)]
            if len(sel) < 3:
                est.append(np.nan)
                continue
            # Straight line through the neighbors (self excluded), read off at
            # this strike. Unlike a plain average, a line respects the skew's
            # slope, so the lowest/highest strikes are not biased.
            slope, intercept = np.polyfit(xs[sel], ivs[sel], 1)
            est.append(intercept + slope * xs[i])
        u.loc[g.index, "neighbor_iv"] = est
    u["diff_strike"] = u["computed_iv"] - u["neighbor_iv"]
    scale = u.groupby("expiration")["diff_strike"].transform(lambda x: robust_scale(x, floor))
    u["z_strike"] = u["diff_strike"] / scale
    return u


def _atm_iv_by_expiry(u: pd.DataFrame) -> pd.Series:
    """IV interpolated at log-moneyness 0 (the money) for each expiration."""
    return u.groupby("expiration").apply(lambda g: interpolate_iv_at(g, 0.0), include_groups=False)


def _expiry_neighbor_component(u: pd.DataFrame, floor: float) -> pd.DataFrame:
    u = u.copy()
    exps = sorted(u["expiration"].unique())
    # Compare against the neighbors' SMOOTHED smile (the strike-trend values
    # from step A) rather than their raw IVs. Otherwise one outlier in the
    # 30-day expiry would make the same strike in the 45-day expiry look like
    # a mirror-image outlier.
    smooth = u.assign(computed_iv=u["neighbor_iv"].fillna(u["computed_iv"]))
    atm = _atm_iv_by_expiry(smooth)
    atm_own = _atm_iv_by_expiry(u)
    by_exp = {e: smooth[smooth["expiration"] == e] for e in exps}
    expected = []
    for row in u.itertuples():
        i = exps.index(row.expiration)
        own_atm = atm_own.get(row.expiration, np.nan)
        prem = []
        for j in (i - 1, i + 1):
            if 0 <= j < len(exps):
                other = exps[j]
                iv_other = interpolate_iv_at(by_exp[other], row.log_moneyness)
                if np.isfinite(iv_other) and np.isfinite(atm.get(other, np.nan)):
                    prem.append(iv_other - atm[other])
        expected.append(own_atm + np.mean(prem) if prem and np.isfinite(own_atm) else np.nan)
    u["expiry_expected_iv"] = expected
    u["diff_expiry"] = u["computed_iv"] - u["expiry_expected_iv"]
    scale = u.groupby("expiration")["diff_expiry"].transform(lambda x: robust_scale(x, floor))
    u["z_expiry"] = u["diff_expiry"] / scale
    return u


def score_anomalies(chains: pd.DataFrame, rv30: float, ticker: str = "",
                    s: config.AnomalySettings = config.SETTINGS.anomaly) -> pd.DataFrame:
    """Return the scoring universe with component z-scores, the 0-100 score and a direction."""
    u = scoring_universe(chains, s)
    if u.empty:
        return u
    u = _strike_neighbor_component(u, s.neighbor_count, s.min_scale_vol_points)
    u = _expiry_neighbor_component(u, s.min_scale_vol_points)

    rv_ok = rv30 is not None and np.isfinite(rv30) and rv30 > 0
    x = np.log(u["computed_iv"] / rv30) if rv_ok else np.log(u["computed_iv"])
    scale_c = robust_scale(x, 0.02)  # 0.02 in log units ≈ 2% relative
    u["z_chain"] = (x - np.median(x)) / scale_c if np.isfinite(scale_c) else np.nan

    w = np.array([s.weight_strike_neighbors, s.weight_expiry_neighbors, s.weight_rest_of_chain])
    z = np.abs(u[["z_strike", "z_expiry", "z_chain"]].to_numpy(dtype=float))
    avail = np.isfinite(z)
    weights = np.where(avail, w, 0.0)
    wsum = weights.sum(axis=1)
    composite = np.where(wsum > 0, np.nansum(np.where(avail, z, 0.0) * weights, axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    u["composite_z"] = composite
    u["score"] = 100.0 * np.clip(composite / s.z_cap, 0.0, 1.0)

    sign = np.where(np.isfinite(u["z_strike"]), np.sign(u["z_strike"]), np.sign(u["z_expiry"].fillna(0)))
    u["direction"] = np.where(sign >= 0, "rich", "cheap")
    u["rv30"] = rv30 if rv_ok else np.nan
    u["iv_minus_rv"] = u["computed_iv"] - u["rv30"]
    u["ticker"] = ticker.upper()
    u["flagged"] = u["score"] >= s.flag_threshold
    return u.sort_values("score", ascending=False).reset_index(drop=True)
