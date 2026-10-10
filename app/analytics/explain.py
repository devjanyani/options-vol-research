"""
Plain-English explanations of anomaly signals (Phase 7).

Two generators, same input:
1. `template_explanation`: deterministic, rule-based, no AI. Always available.
2. `llm_explanation`: an LLM (Claude) rewrites the SAME facts in natural
   language. It is optional and clearly labeled in the UI.

GUARDRAIL AGAINST INVENTED NUMBERS
----------------------------------
The LLM only ever sees a "fact sheet" of numbers this app computed. After it
answers, `find_unverified_numbers` extracts every number from its text and
checks that each one appears in the fact sheet. If the LLM introduces any
number we did not give it, the AI text is rejected and the rule-based
explanation is shown instead, with a note saying why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app import config

NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d[\d,]*(?:\.\d+)?")


def build_fact_sheet(row: pd.Series, spot: float, earnings_date: date | None) -> dict[str, str]:
    """Every number the explanation may use, already formatted."""
    facts = {
        "ticker": str(row["ticker"]),
        "option": f"{row['strike']:g} {row['option_type']}",
        "strike": f"{row['strike']:g}",
        "option_type": str(row["option_type"]),
        "expiration": str(row["expiration"]),
        "days_to_expiry": str(int(row["dte"])),
        "spot": f"{spot:,.2f}",
        "strike_vs_spot_pct": f"{row['moneyness'] * 100:.1f}%",
        "implied_vol": f"{row['computed_iv'] * 100:.1f}%",
        "nearby_strikes_trend_iv": f"{row['neighbor_iv'] * 100:.1f}%" if pd.notna(row["neighbor_iv"]) else "unavailable",
        "difference_vs_nearby_vol_points": f"{(row['computed_iv'] - row['neighbor_iv']) * 100:+.1f}" if pd.notna(row["neighbor_iv"]) else "unavailable",
        "neighbor_expirations_expected_iv": f"{row['expiry_expected_iv'] * 100:.1f}%" if pd.notna(row.get("expiry_expected_iv")) else "unavailable",
        "realized_vol_30d": f"{row['rv30'] * 100:.1f}%" if pd.notna(row["rv30"]) else "unavailable",
        "iv_minus_realized_vol_points": f"{row['iv_minus_rv'] * 100:+.1f}" if pd.notna(row["iv_minus_rv"]) else "unavailable",
        "anomaly_score": f"{row['score']:.0f}",
        "direction": "above" if row["direction"] == "rich" else "below",
        "open_interest": f"{int(row['open_interest']):,}",
        "bid_ask_spread_pct_of_mid": f"{row['rel_spread'] * 100:.1f}%",
    }
    if earnings_date is not None:
        facts["next_earnings_date"] = earnings_date.isoformat()
        facts["earnings_before_expiry"] = "yes" if earnings_date <= pd.Timestamp(row["expiration"]).date() else "no"
    else:
        facts["next_earnings_date"] = "none listed by the data source"
    return facts


def template_explanation(f: dict[str, str]) -> str:
    rel = "expensive" if f["direction"] == "above" else "cheap"
    parts = [
        f"{f['ticker']}'s {f['expiration']} ${f['strike']} {f['option_type']} ({f['days_to_expiry']} days to expiry) "
        f"has an implied volatility of {f['implied_vol']}.",
    ]
    if f["nearby_strikes_trend_iv"] != "unavailable":
        parts.append(f"The trend through nearby strikes suggests about {f['nearby_strikes_trend_iv']} "
                     f"({f['difference_vs_nearby_vol_points']} vol points), so this option looks unusually {rel} "
                     f"relative to its neighbors.")
    if f["realized_vol_30d"] != "unavailable":
        parts.append(f"For context, the stock's 30-day realized volatility is {f['realized_vol_30d']}.")
    if f.get("earnings_before_expiry") == "yes":
        parts.append(f"An earnings announcement on {f['next_earnings_date']} falls before expiry, which can "
                     f"legitimately raise implied volatility.")
    parts.append(f"Anomaly score: {f['anomaly_score']}/100. This is a potential volatility anomaly, not a "
                 f"guaranteed mispricing: stale quotes, events, dividends or early-exercise effects can explain it. "
                 f"The bid/ask spread is {f['bid_ask_spread_pct_of_mid']} of the mid price.")
    return " ".join(parts)


def _normalize(tok: str) -> str:
    try:
        return f"{float(tok.replace(',', '')):.4f}"
    except ValueError:
        return tok


def find_unverified_numbers(text: str, facts: dict[str, str]) -> list[str]:
    """Numbers in `text` that do not appear anywhere in the fact sheet."""
    texts = list(facts.values()) + [k.replace("_", " ") for k in facts]  # keys too, e.g. "realized vol 30d"
    allowed = {_normalize(t) for v in texts for t in NUMBER_RE.findall(v)}
    allowed |= {_normalize(t.lstrip("+-")) for v in texts for t in NUMBER_RE.findall(v)}
    allowed |= {_normalize(x) for x in ("100", "0")}  # "out of 100", "0-100" are part of the score's definition
    found = NUMBER_RE.findall(text)
    return [t for t in found if _normalize(t) not in allowed and _normalize(t.lstrip("-")) not in allowed]


SYSTEM_PROMPT = (
    "You explain options-volatility research signals to a finance student. "
    "Use ONLY the facts provided. Never introduce any number, date, price or statistic that is not in the facts, "
    "and do not compute new numbers (no subtraction, ratios or rounding). Copy numbers exactly as written. "
    "Never say the option is mispriced or that a trade will be profitable; call it a 'potential volatility anomaly'. "
    "Mention plausible innocent explanations (events, stale quotes, dividends, early exercise). "
    "If an earnings date before expiry is listed, mention it. Write 3-5 plain sentences, no lists, no headings."
)


@dataclass
class Explanation:
    text: str
    source: str            # "ai" or "rule-based"
    note: str = ""         # why AI was not used, if applicable


def llm_explanation(facts: dict[str, str], api_key: str = "", model: str = "") -> Explanation:
    fallback = template_explanation(facts)
    api_key = api_key or config.ANTHROPIC_API_KEY
    if not api_key:
        return Explanation(fallback, "rule-based", "No ANTHROPIC_API_KEY set; showing the rule-based explanation.")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        fact_lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
        msg = client.messages.create(
            model=model or config.LLM_MODEL, max_tokens=400, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Facts:\n{fact_lines}\n\nExplain this signal."}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception as exc:  # noqa: BLE001
        return Explanation(fallback, "rule-based", f"AI request failed ({type(exc).__name__}); showing the rule-based explanation.")
    bad = find_unverified_numbers(text, facts)
    if bad:
        return Explanation(fallback, "rule-based",
                           f"AI text was rejected because it contained numbers not computed by the app: {', '.join(bad[:5])}.")
    return Explanation(text, "ai")
