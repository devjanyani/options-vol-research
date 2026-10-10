"""
All beginner-facing explanations live here, separate from code and layout.

GLOSSARY: one or two sentences, used for column tooltips and "?" icons.
CONCEPTS: longer "Learn" panels. Each one answers the same five questions:
    What problem does it solve? Why do traders care? What does the formula
    mean? What are we assuming? Where can it fail?
"""

GLOSSARY = {
    "spot": "The current price of the underlying stock or ETF.",
    "strike": "The price at which the option lets you buy (call) or sell (put) the stock.",
    "expiration": "The last day the option exists. US equity options stop trading at 4:00 pm New York time.",
    "dte": "Days to expiration: calendar days left until the option expires.",
    "option_type": "A call profits when the stock rises above the strike; a put profits when it falls below the strike.",
    "bid": "The highest price someone is currently willing to pay. If you sell right now, you get roughly this.",
    "ask": "The lowest price someone is currently willing to sell at. If you buy right now, you pay roughly this.",
    "mid": "Halfway between bid and ask. Our best single estimate of the option's market value.",
    "spread": "Ask minus bid. The cost of trading immediately. Wide spreads mean expensive, illiquid options.",
    "rel_spread": "Spread divided by the mid price. 10% means you lose about 5% just from buying at the ask.",
    "last_price": "Price of the most recent trade. It can be hours or days old, so we prefer the mid price.",
    "volume": "Number of contracts traded today.",
    "open_interest": "Number of contracts currently open (not yet closed or expired). A measure of how established a market is.",
    "yahoo_iv": "Yahoo's own implied-volatility estimate. Often stale; shown only for comparison.",
    "computed_iv": "Our implied volatility, solved from the mid price with Black-Scholes. The amount of future price movement the options market appears to be expecting, as an annual percentage.",
    "iv": "Implied volatility: the amount of future price movement the options market appears to be expecting, expressed as an annualized percentage.",
    "rv": "Realized (historical) volatility: how much the stock actually moved over a recent window, annualized.",
    "delta": "How much the option price changes when the stock moves $1. Calls 0 to 1, puts -1 to 0.",
    "gamma": "How much delta changes when the stock moves $1. High near the strike and close to expiry.",
    "vega": "How much the option price changes when implied volatility moves 1 percentage point.",
    "theta": "How much value the option loses per calendar day from the passage of time.",
    "moneyness": "Strike divided by spot. Below 1 = strike below the stock price; above 1 = strike above it.",
    "model_price": "Black-Scholes price if future volatility equals the stock's 30-day realized volatility.",
    "market_price": "The option's bid/ask midpoint, i.e. what the market currently charges.",
    "is_liquid": "Whether the quote passes our filters (real bid, reasonable spread, enough open interest). Illiquid quotes are excluded from analysis.",
    "anomaly_score": "0-100. How unusual this option's IV is versus nearby strikes, nearby expirations and the rest of the chain. High does NOT mean profitable.",
    "win_rate": "Share of simulated trades that made money after the stated costs.",
    "sharpe": "Average return divided by the variability of returns, annualized. Higher = smoother returns. Very sensitive to sample size.",
    "max_drawdown": "Largest peak-to-trough fall in cumulative P&L. 'How bad did it get before it got better?'",
}


def delta_sentence(delta: float, option_type: str) -> str:
    direction = "rises" if delta >= 0 else "falls"
    return (
        f"Delta ≈ {delta:.2f} means that, all else equal, this {option_type}'s price {direction} by about "
        f"${abs(delta):.2f} when the stock rises $1. It is also a rough, risk-neutral gauge of the chance "
        f"the option finishes in the money (about {abs(delta):.0%})."
    )


def gamma_sentence(gamma: float, delta: float) -> str:
    return (
        f"Gamma ≈ {gamma:.4f} means that if the stock rises $1, delta moves from {delta:.2f} to about "
        f"{delta + gamma:.2f}. Gamma is what makes option positions speed up or slow down as the stock moves."
    )


def vega_sentence(vega: float) -> str:
    return (
        f"Vega ≈ {vega:.3f} means that if implied volatility rises by 1 percentage point (say 30% → 31%), "
        f"the option gains about ${vega:.3f} per share (${vega * 100:.2f} per 100-share contract)."
    )


def theta_sentence(theta: float) -> str:
    return (
        f"Theta ≈ {theta:.3f} means the option loses about ${abs(theta):.3f} per share "
        f"(${abs(theta) * 100:.2f} per contract) each calendar day if nothing else changes."
    )


CONCEPTS: dict[str, dict[str, str]] = {
    "options_basics": {
        "title": "Options in five minutes",
        "problem": "Options let you bet on, or protect against, a stock's move with limited upfront cost.",
        "why": "Their price depends mostly on ONE unknown: how much the stock will move. That makes options the market's way of trading volatility.",
        "formula": (
            "A call's payoff at expiry is max(Stock − Strike, 0). A put's is max(Strike − Stock, 0). "
            "Before expiry an option is worth more than this 'intrinsic value' because the stock still has time to move. "
            "That extra is 'time value', and it is bigger when the stock is expected to be more volatile."
        ),
        "assumptions": "One US equity option controls 100 shares. Prices are quoted per share.",
        "failure": "Quoted prices are only meaningful if someone will actually trade there. A $0 bid means no buyer.",
    },
    "realized_vol": {
        "title": "Realized (historical) volatility",
        "problem": "We need a measure of how much the stock has ACTUALLY been moving, to compare with what options imply.",
        "why": "If options price in much more movement than the stock delivers, option sellers tend to profit, and vice versa. That gap is the heart of volatility trading.",
        "formula": (
            "Take daily log returns ln(P_today / P_yesterday), compute their standard deviation over N days, "
            "then multiply by √252 to annualize. Variance grows linearly with time, so standard deviation grows with its square root. "
            "A 1% daily standard deviation ≈ 15.9% annualized."
        ),
        "assumptions": "Daily moves are independent and come from a stable distribution over the window.",
        "failure": "It is backward-looking. A calm stock with earnings next week will show low realized vol but legitimately high implied vol. Short windows are noisy; long windows react slowly.",
    },
    "black_scholes": {
        "title": "Black-Scholes",
        "problem": "Gives a fair price for an option from five observable inputs plus one unobservable one: volatility.",
        "why": "It is the common language of options markets. Traders quote options in implied volatility (from Black-Scholes) rather than dollars because IV is comparable across strikes, expiries and stocks.",
        "formula": (
            "Call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2), Put = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1), "
            "with d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T) and d2 = d1 − σ√T. "
            "Read it as: (what you expect to receive) − (what you expect to pay), each weighted by a probability and discounted to today."
        ),
        "assumptions": "Constant volatility, log-normal prices with no jumps, constant interest rate and dividend yield, frictionless trading, European exercise.",
        "failure": "Volatility is not constant (the smile/skew proves it), stocks jump on news, and US equity options are American (early exercise), which matters most for deep in-the-money puts and calls just before dividends.",
    },
    "implied_vol": {
        "title": "Implied volatility",
        "problem": "Converts an option's dollar price into the volatility the market is 'charging' for.",
        "why": "A $5 option on a $50 stock and a $5 option on a $500 stock are not comparable. Their IVs are. IV is how traders say an option is 'expensive' or 'cheap'.",
        "formula": "Find σ such that BlackScholes(σ) = market mid price. There is no closed-form answer, so we search numerically with Brent's method between 0.01% and 500%.",
        "assumptions": "That Black-Scholes is the right translation between price and volatility, and that the mid price is a fair market price.",
        "failure": "Stale or one-sided quotes produce garbage IVs; some prices are impossible under any volatility (below intrinsic value). We flag those instead of forcing a number.",
    },
    "greeks": {
        "title": "The Greeks",
        "problem": "Tell you how your option's value will react to each risk factor.",
        "why": "Traders manage a book of options by its Greeks, not by individual positions: 'I am long $50k of vega and short $2k of theta per day.'",
        "formula": "Each Greek is a derivative (rate of change) of the Black-Scholes price: Delta = ∂V/∂S, Gamma = ∂²V/∂S², Vega = ∂V/∂σ, Theta = ∂V/∂t.",
        "assumptions": "They describe SMALL moves with everything else held fixed.",
        "failure": "For big moves the Greeks change (that is what gamma measures). Real markets move several factors at once, e.g. stocks falling while IV rises.",
    },
    "variance_risk_premium": {
        "title": "Why implied vol is usually above realized vol",
        "problem": "Explains why 'IV > historical vol' is NOT automatically a mispricing.",
        "why": "Option sellers take on crash risk. Investors willingly overpay for protection (like insurance), so on average IV exceeds the volatility that actually follows. This average gap is called the variance risk premium.",
        "formula": "Variance risk premium ≈ IV² − (future realized vol)². We look at the simpler IV − realized vol.",
        "assumptions": "That the premium is roughly stable over time.",
        "failure": "It is compensation for real risk. Strategies that sell volatility earn small gains most of the time and occasionally suffer large losses (e.g. February 2018, March 2020).",
    },
    "surface": {
        "title": "The volatility surface",
        "problem": "Shows how IV varies by strike and by expiration in one picture.",
        "why": "If Black-Scholes were literally true the surface would be flat. Its shape reveals what the market fears: downside 'skew' (crash protection is expensive) and term structure (near-term vs long-term uncertainty, earnings bumps).",
        "formula": "Each cell = implied volatility of the option at that strike (column) and expiration (row). We use out-of-the-money options (puts below spot, calls above), which are the most actively traded.",
        "assumptions": "Quotes across the surface are roughly simultaneous and liquid.",
        "failure": "Gaps where quotes are missing or illiquid; very short expirations are noisy.",
    },
    "anomaly": {
        "title": "The Volatility Anomaly Score",
        "problem": "Flags options whose IV looks out of line with their neighbors, which is worth a closer look.",
        "why": "The surface is usually smooth. A kink can be a real opportunity, a hint of informed trading, a known event, or (very often) a bad quote.",
        "formula": (
            "Three robust z-scores, each = (difference) / (typical difference in that expiry): "
            "(A) IV vs median IV of up to 3 strikes on each side, weight 50%; "
            "(B) IV vs the same moneyness at adjacent expirations, weight 25%; "
            "(C) log(IV / 30-day realized vol) vs the rest of the chain, weight 25%. "
            "Score = 100 × min(weighted |z| / 4, 1). A combined 4-standard-deviation outlier scores 100."
        ),
        "assumptions": "Neighboring options should have similar IVs; the median absolute deviation is a fair 'typical difference'.",
        "failure": "Earnings, dividends, hard-to-borrow stocks and American early exercise create legitimate kinks. A high score is a reason to investigate, not a trade.",
    },
    "backtest": {
        "title": "Backtesting honestly",
        "problem": "Checks whether a rule would have made money in the past, before risking real money.",
        "why": "Almost every idea looks good until it is tested with chronological data and real costs.",
        "formula": "On each historical date, use ONLY data available that day to make a decision, then measure the outcome over the next N trading days. Repeat, step forward, never peek.",
        "assumptions": "Fills at the stated prices with the stated costs; one contract per signal; a static delta hedge.",
        "failure": "Look-ahead bias, overfitting thresholds, too few trades, regime changes, and ignoring costs. Past simulated results do not predict future returns.",
    },
    "costs": {
        "title": "Transaction costs",
        "problem": "Turns theoretical P&L into what you could actually have earned.",
        "why": "Options spreads are wide relative to option prices. Many 'profitable' strategies lose money once you buy at the ask and sell at the bid.",
        "formula": "Net P&L = Gross P&L (mid to mid) − half-spread on entry − half-spread on exit − slippage − commissions − hedge costs.",
        "assumptions": "$0.65 per contract per side, 10% of the spread as slippage, $0.01/share for the stock hedge. All editable.",
        "failure": "Real costs depend on size, time of day, and market conditions. Large orders move prices more.",
    },
    "sharpe_drawdown": {
        "title": "Sharpe ratio and maximum drawdown",
        "problem": "Summarize risk-adjusted performance and worst-case pain.",
        "why": "A strategy that makes 10% with wild swings is worse than one making 8% smoothly. Firms look at both numbers.",
        "formula": "Sharpe = mean(period returns) / std(period returns) × √(periods per year). Max drawdown = largest fall from a running peak of cumulative P&L.",
        "assumptions": "Returns are roughly independent and normally distributed (they are not, for short-volatility strategies).",
        "failure": "Short-volatility strategies often show high Sharpe ratios right up until a crash. Few trades make both numbers unreliable.",
    },
}
