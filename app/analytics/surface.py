"""
Volatility surface and term structure.

The surface is a grid of implied volatilities:
    rows    = expiration dates
    columns = strikes
    value   = implied volatility

Convention: we use OUT-OF-THE-MONEY options only (puts below the stock price,
calls above it). OTM options are the most traded, and their prices are mostly
time value, so their IVs are the most reliable. In-the-money options are
mostly intrinsic value; a one-cent quote error moves their IV a lot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NICE_STEPS = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]


def surface_points(chains: pd.DataFrame, side: str = "otm", max_abs_log_moneyness: float = 0.25,
                   liquid_only: bool = True) -> pd.DataFrame:
    """Long table (one row per usable option) for the chosen side: 'otm', 'call' or 'put'."""
    if chains.empty:
        return chains
    df = chains[chains["iv_status"] == "ok"].copy()
    if liquid_only:
        df = df[df["is_liquid"]]
    if side == "otm":
        df = df[df["is_otm"]]
    else:
        df = df[df["option_type"] == side]
    df = df[df["log_moneyness"].abs() <= max_abs_log_moneyness]
    return df[["expiration", "dte", "strike", "moneyness", "log_moneyness", "option_type", "computed_iv",
               "open_interest", "mid"]].reset_index(drop=True)


def choose_strike_step(strikes: pd.Series, max_columns: int = 40) -> float:
    """Smallest 'nice' strike spacing that keeps the heatmap readable."""
    lo, hi = float(strikes.min()), float(strikes.max())
    for step in NICE_STEPS:
        if (hi - lo) / step <= max_columns:
            return step
    return NICE_STEPS[-1]


def surface_grid(points: pd.DataFrame, max_columns: int = 40) -> tuple[pd.DataFrame, float]:
    """
    Pivot to rows=expiration, columns=strike bucket, values=mean IV.
    Strikes are bucketed to a 'nice' spacing so a $1-strike chain like SPY
    still fits on screen; the step used is returned so the UI can say so.
    """
    if points.empty:
        return pd.DataFrame(), float("nan")
    step = choose_strike_step(points["strike"], max_columns)
    p = points.assign(strike_bucket=(points["strike"] / step).round() * step)
    grid = p.pivot_table(index="expiration", columns="strike_bucket", values="computed_iv", aggfunc="mean")
    return grid.sort_index(), step


def term_structure(chains: pd.DataFrame, spot: float) -> pd.DataFrame:
    """At-the-money IV for each expiration (average of call and put at the strike nearest spot)."""
    rows = []
    for exp, g in chains.groupby("expiration"):
        g = g[(g["iv_status"] == "ok") & g["is_liquid"]]
        if g.empty:
            continue
        k = g.loc[(g["strike"] - spot).abs().idxmin(), "strike"]
        rows.append({"expiration": exp, "dte": int(g["dte"].iloc[0]),
                     "atm_iv": float(g.loc[g["strike"] == k, "computed_iv"].mean()), "atm_strike": float(k)})
    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True) if rows else pd.DataFrame()


def atm_iv_history(snapshots: pd.DataFrame, target_dte: int = 30) -> pd.DataFrame:
    """
    From COLLECTED snapshots: for each snapshot day, the ATM IV of the
    expiration closest to `target_dte`. Uses only data stored that day.
    """
    out = []
    for d, day in snapshots.groupby("snapshot_date"):
        day = day[(day["iv_status"] == "ok") & day["is_liquid"].astype(bool)]
        if day.empty:
            continue
        exp_dte = day.groupby("expiration")["dte"].first()
        exp = (exp_dte - target_dte).abs().idxmin()
        g = day[day["expiration"] == exp]
        spot = float(g["spot"].iloc[0])
        k = g.loc[(g["strike"] - spot).abs().idxmin(), "strike"]
        out.append({"date": pd.Timestamp(d), "atm_iv": float(g.loc[g["strike"] == k, "computed_iv"].mean()),
                    "expiration": exp, "dte": int(exp_dte[exp]), "rv30": float(g["rv30"].iloc[0])})
    return pd.DataFrame(out)


def interpolate_iv_at(points_one_expiry: pd.DataFrame, log_m: float) -> float:
    """Linear interpolation of IV at a given log-moneyness within one expiration (NaN outside the range)."""
    g = points_one_expiry.sort_values("log_moneyness")
    x, y = g["log_moneyness"].to_numpy(), g["computed_iv"].to_numpy()
    if len(x) < 2 or log_m < x[0] or log_m > x[-1]:
        return float("nan")
    return float(np.interp(log_m, x, y))
