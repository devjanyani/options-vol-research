"""Reusable UI building blocks."""

from __future__ import annotations

import html
import math

import streamlit as st

from app.ui import learn, theme
from app.utils.provenance import DESCRIPTIONS, DataKind


def badge(kind: DataKind | str) -> str:
    kind = DataKind(kind) if not isinstance(kind, DataKind) else kind
    color = theme.PROVENANCE_COLORS[kind.value]
    tip = html.escape(DESCRIPTIONS[kind])
    return (
        f'<span class="ovrp-badge" title="{tip}" style="color:{color};border-color:{color};">'
        f"{kind.value}</span>"
    )


def page_header(title: str, subtitle: str, kinds: list[DataKind] = ()) -> None:
    badges = "".join(badge(k) for k in kinds)
    st.markdown(
        f'<div class="ovrp-page-title"><h1>{html.escape(title)}</h1>{badges}</div>'
        f'<p class="ovrp-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "") -> None:
    """kind: '' (info), 'warn', 'crit', 'ai'. `text` may contain simple HTML."""
    st.markdown(f'<div class="ovrp-callout {kind}">{text}</div>', unsafe_allow_html=True)


def synthetic_banner() -> None:
    st.markdown(
        '<div class="ovrp-synthetic">SYNTHETIC TEST DATA: developer test mode. '
        "Nothing on this screen is market data.</div>",
        unsafe_allow_html=True,
    )


def explain_card(label: str, value: str, explanation: str) -> None:
    st.markdown(
        f'<div class="ovrp-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="explain">{explanation}</div></div>',
        unsafe_allow_html=True,
    )


def learn_panel(key: str, expanded: bool = False) -> None:
    """Standard 'Learn' expander: problem, why traders care, formula, assumptions, failure modes."""
    c = learn.CONCEPTS[key]
    with st.expander(f"Learn: {c['title']}", expanded=expanded):
        st.markdown(
            f"**What problem it solves.** {c['problem']}\n\n"
            f"**Why traders care.** {c['why']}\n\n"
            f"**What the formula means.** {c['formula']}\n\n"
            f"**What we assume.** {c['assumptions']}\n\n"
            f"**Where it can fail.** {c['failure']}"
        )


def pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x * 100:.{digits}f}%"


def money(x: float, digits: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{digits}f}"


def num(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:,.{digits}f}"


def pnl_stack(gross: float, cost_lines: list[tuple[str, float]], net: float, label: str = "Simulated P&L") -> None:
    rows = [f'<div class="row"><span>Gross P&amp;L at mid prices</span><span>{money(gross)}</span></div>']
    for name, v in cost_lines:
        rows.append(f'<div class="row"><span>− {html.escape(name)}</span><span>{money(-abs(v))}</span></div>')
    rows.append(f'<div class="row total"><span>Net P&amp;L</span><span>{money(net)}</span></div>')
    st.markdown(f'<div class="ovrp-pnl-stack">{"".join(rows)}</div>', unsafe_allow_html=True)


def topbar(ctx) -> None:
    """Ticker, price, daily change, session status. Shown on every page."""
    from app.utils.dates import is_market_open

    s = ctx.spot
    chg_html = ""
    if s.previous_close:
        chg = s.price - s.previous_close
        cls, arrow = ("chg-up", "▲") if chg >= 0 else ("chg-dn", "▼")
        chg_html = f'<span class="{cls}">{arrow} {chg:+.2f} ({chg / s.previous_close:+.2%})</span>'
    session = "Market open" if is_market_open() else "Market closed: quotes may be stale"
    earn = f" · Next earnings {ctx.earnings_date:%b %d, %Y}" if ctx.earnings_date else ""
    st.markdown(
        f'<div class="ovrp-topbar"><div class="ovrp-quote"><span class="tk">{html.escape(ctx.ticker)}</span>'
        f'<span class="px">{s.price:,.2f}</span>{chg_html}'
        f'<span class="meta">{session}{earn}</span></div>{badge(DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE)}</div>',
        unsafe_allow_html=True,
    )
