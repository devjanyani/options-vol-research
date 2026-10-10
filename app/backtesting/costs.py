"""
Transaction costs (Phase 9).

Gross P&L assumes we trade at the mid price. Real trading does not work like
that. Each option trade pays:

1. Half the bid/ask spread: buying at the ask (or selling at the bid) instead
   of the mid costs (ask - bid) / 2 per share.
2. Slippage: the price often moves against you a little more while you trade.
   We assume a fixed fraction of the full spread.
3. Fees: commission + exchange/regulatory fees per contract.

A delta-hedged trade also buys or sells stock, which costs a little per share.

All costs are returned in DOLLARS for the whole position.
"""

from __future__ import annotations

from dataclasses import dataclass

from app import config


@dataclass(frozen=True)
class LegCost:
    half_spread: float
    slippage: float
    fees: float

    @property
    def total(self) -> float:
        return self.half_spread + self.slippage + self.fees


def option_leg_cost(bid: float, ask: float, contracts: int,
                    cm: config.CostModel = config.SETTINGS.costs) -> LegCost:
    """Cost of ONE fill (entry or exit) of `contracts` contracts with a real quote."""
    spread = max(ask - bid, 0.0)
    m = config.CONTRACT_MULTIPLIER * abs(contracts)
    return LegCost(half_spread=0.5 * spread * m,
                   slippage=cm.slippage_fraction_of_spread * spread * m,
                   fees=cm.option_fee_per_contract * abs(contracts))


def modeled_option_leg_cost(model_price: float, contracts: int,
                            cm: config.CostModel = config.SETTINGS.costs) -> LegCost:
    """Same as above when there is no real quote: the spread is ASSUMED to be a fraction of the price."""
    spread = cm.modeled_spread_fraction * model_price
    m = config.CONTRACT_MULTIPLIER * abs(contracts)
    return LegCost(half_spread=0.5 * spread * m,
                   slippage=cm.slippage_fraction_of_spread * spread * m,
                   fees=cm.option_fee_per_contract * abs(contracts))


def stock_hedge_cost(shares: float, cm: config.CostModel = config.SETTINGS.costs) -> float:
    """Cost of buying or selling `shares` of stock once."""
    return abs(shares) * cm.stock_cost_per_share
