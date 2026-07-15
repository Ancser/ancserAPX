# Production backtest execution-cost semantics

The production backtest now reports trading turnover and deducts three
separate execution-cost components. Rates are one-way basis points of the
notional actually traded:

- `commission_bps`: broker commission on buys and sells. Default `0`, matching
  the normal Alpaca US-stock broker commission assumption.
- `slippage_bps`: spread plus execution slippage on buys and sells. Default
  `5`. This is not presented as an Alpaca commission.
- `regulatory_sell_bps`: a caller-supplied blended regulatory rate on sells
  only. Default `0`; SEC/FINRA rates change and TAF contains per-share/cap rules
  that cannot be reconstructed exactly from daily target weights.

For pre-trade weights `w` and new targets `w*`:

```text
gross turnover   = sum(abs(w* - w))
one-way turnover = gross turnover / 2        # reporting only
commission       = equity * gross turnover * commission_bps / 10,000
slippage         = equity * gross turnover * slippage_bps / 10,000
regulatory fee   = equity * sell turnover  * regulatory_sell_bps / 10,000
```

The cost base is gross turnover. One-way turnover is not charged again, so a
buy or sale is never double-counted. Margin/borrow expense remains a separate
daily financing cost.

## Turnover penalty

A turnover penalty is an *ex-ante portfolio construction rule*, not another
broker fee. A typical optimizer maximizes:

```text
expected score - lambda * sum(abs(target weight - current weight))
```

It creates a no-trade region: a replacement must offer enough expected signal
improvement to justify trading. This can materially improve net returns when
the ranking changes near the Top-N boundary, but it also changes the strategy's
holdings. Claude #1 currently has a deterministic Top-20 rule rather than an
optimizer, so a penalty must be implemented as a shared backtest/live portfolio
rule and validated out of sample before deployment. The newly reported
turnover and costs provide the evidence needed to calibrate it.

## ADV participation cap

Participation is `order notional / average daily dollar volume`. A cap clips or
stages an order when that ratio is too high. It prevents a backtest from
assuming that an illiquid position can be entered immediately at the observed
close with negligible market impact.

It matters most as account capital grows, in small/illiquid names, and around
market shocks. It is less likely to bind for a small account trading the most
liquid SPY/QQQ constituents, but remains an essential capacity and failure-mode
guard. A fixed-bps production cost model does not replace it: the richer
Research v2 engine already supports lagged ADV, square-root impact, and an ADV
cap. Promoting that rule to live requires order slicing, partial-fill state, and
the same shared target logic in both live and backtest.

## Current limits

- Daily bars cannot identify quoted spread, open-auction slippage, partial
  fills, or per-share regulatory caps.
- Fixed slippage is a sensitivity assumption, not an observed Alpaca fee.
- A market order can exceed the assumed 5 bps during volatile or illiquid
  sessions.
- The static current constituent universe introduces survivorship bias.

## Official fee references checked 2026-07-14

- Alpaca commission/clearing disclosure:
  https://alpaca.markets/support/commission-clearing-fees
- Alpaca regulatory fee explanation:
  https://alpaca.markets/support/regulatory-fees
- Alpaca brokerage fee schedule:
  https://files.alpaca.markets/disclosures/BrokFeeSched.pdf
