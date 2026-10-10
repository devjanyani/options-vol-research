"""Section 4: Volatility Surface. Heatmap first, smile curves, optional 3D."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.analytics.surface import surface_grid, surface_points
from app.ui import components as c
from app.ui import state, theme
from app.utils.provenance import DataKind

SEQ_SCALE = [[i / (len(theme.SEQ_BLUE) - 1), col] for i, col in enumerate(theme.SEQ_BLUE)]


def render() -> None:
    ctx = state.require_context()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Volatility Surface",
                  "The volatility surface shows how much volatility the options market is pricing for different "
                  "strikes and expiration dates. If Black-Scholes were exactly right, every cell would be the same.",
                  kinds)

    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.3, 1])
    side_label = f1.selectbox("Options used", ["OTM (standard)", "Calls only", "Puts only"],
                              help="OTM = puts below spot, calls above. These are the most traded and most reliable.")
    side = {"OTM (standard)": "otm", "Calls only": "call", "Puts only": "put"}[side_label]
    max_days = f2.select_slider("Expirations up to", options=[30, 60, 90, 120, 180, 270, 370], value=120,
                                format_func=lambda d: f"{d} days")
    width = f3.select_slider("Strike range", options=[0.05, 0.10, 0.15, 0.20, 0.25], value=0.15,
                             format_func=lambda w: f"±{w:.0%} of spot")
    show_3d = f4.toggle("3D view", value=False)

    exps = state.default_surface_expirations(ctx, max_days=max_days, max_count=12)
    if not exps:
        st.warning("No expirations in that window.")
        return
    multi = state.load_multi_chain(ctx.ticker, exps, ctx.spot.price, ctx.rate.value, ctx.dividend.value, ctx.rv30)
    failed = multi.attrs.get("failed_expirations", []) if not multi.empty else list(exps)
    if failed:
        st.caption(f"Could not download: {', '.join(failed)}")
    pts = surface_points(multi, side=side, max_abs_log_moneyness=float(np.log(1 + width)))
    if pts.empty:
        st.warning("No liquid quotes with a solvable IV in this range. Outside market hours Yahoo often returns "
                   "zero bids, which our filters reject. Try again during market hours.")
        return

    grid, step = surface_grid(pts)
    dte_by_exp = pts.groupby("expiration")["dte"].first()
    row_labels = [f"{e} · {dte_by_exp[e]}d" for e in grid.index]
    z = grid.to_numpy() * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=grid.columns.to_numpy(), y=row_labels, colorscale=SEQ_SCALE, hoverongaps=False,
        colorbar=dict(title=dict(text="IV %", font=dict(color=theme.TEXT_2)), ticksuffix="%",
                      tickfont=dict(family=theme.FONT_MONO, color=theme.MUTED)),
        xgap=2, ygap=2,
        customdata=np.broadcast_to((grid.columns.to_numpy() / ctx.spot.price)[None, :], z.shape),
        hovertemplate="Expiry %{y}<br>Strike ≈ %{x:,.2f} (%{customdata:.1%} of spot)<br>IV %{z:.2f}%<extra></extra>",
    ))
    fig.add_vline(x=ctx.spot.price, line=dict(color=theme.TEXT, width=1, dash="dot"),
                  annotation_text=f"spot {ctx.spot.price:,.2f}", annotation_position="top",
                  annotation_font=dict(color=theme.TEXT_2, size=11))
    fig.update_layout(title="Implied volatility heatmap (rows = expirations, columns = strikes)",
                      height=max(320, 46 * len(grid) + 120), xaxis_title=f"Strike (bucketed to ${step:g})",
                      yaxis=dict(autorange="reversed", tickfont=dict(family=theme.FONT_MONO, size=11)),
                      hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(pts)} liquid options across {len(grid)} expirations. Blank cells = no usable quote. "
               f"Darker = lower IV, brighter = higher IV.")

    # Smile curves for a few expirations
    st.markdown("#### Volatility smile: IV across strikes for selected expirations")
    default = list(grid.index[: min(3, len(grid))])
    chosen = st.multiselect("Expirations", list(grid.index), default=default, max_selections=4,
                            format_func=lambda e: f"{e} · {dte_by_exp[e]}d")
    fig2 = go.Figure()
    for i, e in enumerate(chosen):
        g = pts[pts["expiration"] == e].sort_values("strike")
        fig2.add_trace(go.Scatter(x=g["strike"], y=g["computed_iv"] * 100, mode="lines+markers",
                                  name=f"{e} ({dte_by_exp[e]}d)", line=dict(color=theme.SERIES[i], width=2),
                                  marker=dict(size=6), hovertemplate="Strike %{x:,.2f}<br>IV %{y:.2f}%"))
    fig2.add_vline(x=ctx.spot.price, line=dict(color=theme.MUTED, width=1, dash="dot"))
    fig2.update_layout(height=360, xaxis_title="Strike", yaxis_title="Implied volatility (%)", yaxis_ticksuffix="%",
                       hovermode="closest")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Equity smiles usually slope down from left to right ('skew'): downside puts carry higher IV because "
               "investors pay up for crash protection.")

    if show_3d:
        mgrid = pts.assign(m_bucket=(pts["moneyness"] / 0.01).round() * 0.01).pivot_table(
            index="dte", columns="m_bucket", values="computed_iv", aggfunc="mean").sort_index()
        fig3 = go.Figure(go.Surface(z=mgrid.to_numpy() * 100, x=mgrid.columns.to_numpy() * 100, y=mgrid.index.to_numpy(),
                                    colorscale=SEQ_SCALE, connectgaps=False,
                                    colorbar=dict(title=dict(text="IV %"), ticksuffix="%"),
                                    hovertemplate="Moneyness %{x:.0f}%<br>DTE %{y}<br>IV %{z:.2f}%<extra></extra>"))
        fig3.update_layout(height=560, scene=dict(
            xaxis_title="Strike / spot (%)", yaxis_title="Days to expiry", zaxis_title="IV (%)",
            xaxis=dict(backgroundcolor=theme.PANEL, gridcolor="#2a2b2f"),
            yaxis=dict(backgroundcolor=theme.PANEL, gridcolor="#2a2b2f"),
            zaxis=dict(backgroundcolor=theme.PANEL, gridcolor="#2a2b2f")), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("3D view uses moneyness (strike ÷ spot) so different expirations line up. Gaps are left empty "
                   "rather than interpolated.")

    c.learn_panel("surface")
