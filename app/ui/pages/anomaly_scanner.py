"""Section 5: Anomaly Scanner, with plain-English (rule-based and optional AI) explanations."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import config
from app.analytics.anomaly import score_anomalies
from app.analytics.explain import build_fact_sheet, llm_explanation, template_explanation
from app.ui import components as c
from app.ui import learn, state, theme
from app.utils.dates import market_now
from app.utils.provenance import DataKind

TABLE_COLS = {
    "ticker": st.column_config.TextColumn("Ticker"),
    "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
    "expiration": st.column_config.TextColumn("Expiration"),
    "dte": st.column_config.NumberColumn("DTE", format="%d"),
    "option_type": st.column_config.TextColumn("Call/Put"),
    "iv_pct": st.column_config.NumberColumn("Implied vol", format="%.1f%%", help=learn.GLOSSARY["iv"]),
    "rv_pct": st.column_config.NumberColumn("Hist. vol 30d", format="%.1f%%", help=learn.GLOSSARY["rv"]),
    "iv_rv_pts": st.column_config.NumberColumn("IV − HV (pts)", format="%+.1f",
                                               help="Implied minus 30-day realized volatility, in vol points."),
    "neighbor_pct": st.column_config.NumberColumn("Nearby-strike IV", format="%.1f%%",
                                                  help="IV predicted by a trend line through up to 3 strikes on each side."),
    "vs_nb_pts": st.column_config.NumberColumn("vs nearby (pts)", format="%+.1f"),
    "score": st.column_config.ProgressColumn("Anomaly score", min_value=0, max_value=100, format="%.0f",
                                             help=learn.GLOSSARY["anomaly_score"]),
    "direction": st.column_config.TextColumn("IV vs neighbors", help="rich = IV above comparable options; cheap = below."),
}


def _table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["iv_pct"] = out["computed_iv"] * 100
    out["rv_pct"] = out["rv30"] * 100
    out["iv_rv_pts"] = out["iv_minus_rv"] * 100
    out["neighbor_pct"] = out["neighbor_iv"] * 100
    out["vs_nb_pts"] = (out["computed_iv"] - out["neighbor_iv"]) * 100
    return out[list(TABLE_COLS)]


def _smile_chart(scored: pd.DataFrame, expiration: str, threshold: float) -> go.Figure:
    g = scored[scored["expiration"] == expiration].sort_values("strike")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["strike"], y=g["computed_iv"] * 100, mode="lines+markers", name="Implied vol",
                             line=dict(color=theme.SERIES[0], width=2), marker=dict(size=7),
                             customdata=g["score"], hovertemplate="Strike %{x:,.2f}<br>IV %{y:.2f}%<br>"
                                                                  "Score %{customdata:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["strike"], y=g["neighbor_iv"] * 100, mode="lines", name="Nearby-strike trend",
                             line=dict(color=theme.SERIES[1], width=2, dash="dash"),
                             hovertemplate="Trend %{y:.2f}%<extra></extra>"))
    f = g[g["score"] >= threshold]
    if not f.empty:
        fig.add_trace(go.Scatter(x=f["strike"], y=f["computed_iv"] * 100, mode="markers", name="Flagged",
                                 marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=theme.WARN, width=2.5)),
                                 hovertemplate="Flagged · score %{customdata:.0f}<extra></extra>", customdata=f["score"]))
    fig.update_layout(title=f"Smile for {expiration}: each option vs the trend of its neighbors", height=360,
                      xaxis_title="Strike", yaxis_title="Implied volatility (%)", yaxis_ticksuffix="%",
                      hovermode="closest")
    return fig


def render() -> None:
    ctx = state.require_context()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Anomaly Scanner",
                  "Finds options whose implied volatility is unusually far from comparable options. Results are "
                  "<b>potential volatility anomalies</b>: reasons to look closer, never guaranteed mispricings.", kinds)

    f1, f2, f3 = st.columns(3)
    max_days = f1.select_slider("Expirations up to", options=[30, 60, 90, 120, 180], value=90,
                                format_func=lambda d: f"{d} days")
    threshold = f2.slider("Flag threshold (score)", 20, 90, int(config.SETTINGS.anomaly.flag_threshold), step=5)
    show_all = f3.toggle("Show all scored options", value=False)

    exps = state.default_surface_expirations(ctx, max_days=max_days, max_count=10)
    multi = state.load_multi_chain(ctx.ticker, exps, ctx.spot.price, ctx.rate.value, ctx.dividend.value, ctx.rv30)
    if multi.empty:
        st.warning("No option data available.")
        return
    scored = score_anomalies(multi, ctx.rv30, ctx.ticker)
    if scored.empty:
        st.warning("No options passed the liquidity filters. Outside market hours Yahoo often returns zero bids; "
                   "try again while the market is open.")
        return
    flagged = scored[scored["score"] >= threshold]

    m = st.columns(4)
    m[0].metric("Contracts downloaded", f"{len(multi):,}")
    m[1].metric("Scored (liquid, OTM)", f"{len(scored):,}", help="Only liquid out-of-the-money options with a solvable IV are scored.")
    m[2].metric("Flagged", f"{len(flagged):,}")
    m[3].metric("Highest score", f"{scored['score'].max():.0f}")

    st.dataframe(_table(scored if show_all else flagged), column_config=TABLE_COLS, hide_index=True,
                 use_container_width=True, height=min(420, 38 + 35 * max(1, len(scored if show_all else flagged))))
    if flagged.empty:
        c.callout(f"No option scored at or above {threshold}. On a liquid underlying that is the normal outcome: the "
                  "surface is usually smooth.")

    focus_exp = (flagged if not flagged.empty else scored)["expiration"].iloc[0]
    exp_choice = st.selectbox("Smile chart expiration", sorted(scored["expiration"].unique()),
                              index=sorted(scored["expiration"].unique()).index(focus_exp))
    st.plotly_chart(_smile_chart(scored, exp_choice, threshold), use_container_width=True)

    if not flagged.empty:
        st.markdown("#### Explanations")
        st.caption("Rule-based explanations are generated from the numbers above with fixed sentences. "
                   "AI explanations are optional, clearly labeled, and rejected if they contain any number the app "
                   "did not compute.")
        cache = st.session_state.setdefault("ai_explanations", {})
        for _, row in flagged.head(10).iterrows():
            facts = build_fact_sheet(row, ctx.spot.price, ctx.earnings_date)
            key = f"{row['contract_symbol']}|{ctx.fetched_at}"
            title = (f"{row['ticker']} {row['expiration']} {row['strike']:g} {row['option_type']} · "
                     f"score {row['score']:.0f} · IV {row['direction']}")
            with st.expander(title):
                c.callout(f"<b>Rule-based explanation.</b> {html.escape(template_explanation(facts))}")
                if key in cache:
                    ex = cache[key]
                    if ex.source == "ai":
                        c.callout(f"<b>AI-generated explanation ({html.escape(config.LLM_MODEL)}).</b> "
                                  f"Every number was checked against the app's calculations.<br>{html.escape(ex.text)}", "ai")
                    else:
                        c.callout(html.escape(ex.note), "warn")
                elif st.button("Generate AI explanation", key=f"ai_{key}"):
                    with st.spinner("Asking the model…"):
                        cache[key] = llm_explanation(facts)
                    st.rerun()
                with st.popover("Facts sent to the AI"):
                    st.json(facts)

        if st.button("Log flagged signals to the database"):
            sig = flagged.assign(signal_date=market_now().strftime("%Y-%m-%d"), iv=flagged["computed_iv"])
            n = state.get_repo().save_signals(sig, DataKind.SYNTHETIC.value if ctx.synthetic else DataKind.LIVE.value)
            st.success(f"Saved {n} signals.")

    c.learn_panel("anomaly")
