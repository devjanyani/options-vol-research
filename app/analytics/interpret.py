"""
Rule-based (non-AI) plain-English interpretation of implied vs realized vol.

These sentences are deterministic: same numbers in, same words out. They
deliberately avoid the word "mispriced".
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def iv_vs_rv_sentences(iv: float, rv: dict[int, float], expiration: str,
                       earnings_date: date | None) -> list[str]:
    out: list[str] = []
    rv30 = rv.get(30, float("nan"))
    if not (pd.notna(iv) and pd.notna(rv30)) or rv30 <= 0:
        return ["Not enough data to compare implied and realized volatility for this contract."]

    diff = iv - rv30
    if diff > 0.02:
        out.append(f"Implied volatility is {iv:.1%} versus {rv30:.1%} realized over the last 30 trading days "
                   f"(+{diff * 100:.1f} vol points). The options market is pricing more future movement than "
                   f"the stock has recently experienced.")
    elif diff < -0.02:
        out.append(f"Implied volatility is {iv:.1%} versus {rv30:.1%} realized over the last 30 trading days "
                   f"({diff * 100:.1f} vol points). The options market is pricing less movement than the stock "
                   f"has recently delivered, which often happens after a volatility spike as traders expect calm.")
    else:
        out.append(f"Implied volatility ({iv:.1%}) is close to 30-day realized volatility ({rv30:.1%}).")

    rv10, rv60 = rv.get(10, float("nan")), rv.get(60, float("nan"))
    if pd.notna(rv10) and pd.notna(rv60) and rv60 > 0:
        if rv10 > rv60 * 1.25:
            out.append(f"Recent movement is accelerating: 10-day realized ({rv10:.1%}) is well above 60-day ({rv60:.1%}).")
        elif rv10 < rv60 * 0.8:
            out.append(f"Recent movement has calmed: 10-day realized ({rv10:.1%}) is well below 60-day ({rv60:.1%}).")

    if earnings_date is not None:
        if earnings_date <= pd.Timestamp(expiration).date():
            out.append(f"An earnings announcement is scheduled for {earnings_date:%b %d, %Y}, before this option "
                       f"expires. Earnings moves are not in past realized volatility but are priced into IV, so "
                       f"a higher IV here is expected.")
        else:
            out.append(f"The next earnings date ({earnings_date:%b %d, %Y}) falls after this expiration.")

    if diff > 0.02:
        out.append("A higher IV is not automatically 'expensive': on average implied volatility sits above the "
                   "volatility that follows, because option sellers demand payment for crash risk.")
    return out
