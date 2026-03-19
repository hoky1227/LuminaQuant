# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_liquidity_shock_reversion_5m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 14.56%
- market_neutral: 30.00%
- mean_reversion: 30.00%
- trend: 25.44%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | liquidity_shock_reversion_5m_thin_lo_72_0.010 | LiquidityShockReversionStrategy | mean_reversion | 5m | 30.00% | -1.714 | -0.95% | -0.245 | -0.38% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.22480108083474204, "calmar": 36.698326844675584, "max_drawdown": 0.006125649318733384, "sharpe": 4.037597106960679, "sortino": 8.139106309058036, "total_return": 0.01737143763183502, "volatility": 0.05054202796319217}
- Report (oos): {"cagr": 0.407462280240398, "calmar": 41.338212025431595, "max_drawdown": 0.009856794967080917, "sharpe": 3.3631051254203057, "sortino": 11.524107766149022, "total_return": 0.033317209280487425, "volatility": 0.10319600656286831}
- Train: {"cagr": -0.009728416783950866, "calmar": -0.14596418049716176, "max_drawdown": 0.06664934335818118, "sharpe": -0.1114079164747479, "sortino": -0.18946866814004576, "total_return": -0.009728416783950866, "volatility": 0.06745309771055541}
- Val: {"cagr": 0.22480108083474204, "calmar": 36.698326844675584, "max_drawdown": 0.006125649318733384, "sharpe": 4.037597106960679, "sortino": 8.139106309058036, "total_return": 0.01737143763183502, "volatility": 0.05054202796319217}
- OOS: {"cagr": 0.407462280240398, "calmar": 41.338212025431595, "max_drawdown": 0.009856794967080917, "sharpe": 3.3631051254203057, "sortino": 11.524107766149022, "total_return": 0.033317209280487425, "volatility": 0.10319600656286831}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.4025829911589036,
      "calmar": 40.57113747776165,
      "max_drawdown": 0.009922891399817724,
      "sharpe": 3.3294222332643484,
      "sortino": 11.408689049039396,
      "total_return": 0.03297316828290975,
      "volatility": 0.10319600656286831
    },
    "x3": {
      "cagr": 0.3977205710621927,
      "calmar": 39.81591812011293,
      "max_drawdown": 0.009988984050609773,
      "sharpe": 3.295739341108391,
      "sortino": 11.293270331929765,
      "total_return": 0.03262923856130118,
      "volatility": 0.10319600656286831
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.3608078014548701,
      "calmar": 40.65774870678963,
      "max_drawdown": 0.008874269061400764,
      "sharpe": 3.3631051254203057,
      "sortino": 11.524107766149024,
      "total_return": 0.029982460545861178,
      "volatility": 0.0928764059065815
    },
    "plus_10pct_signal": {
      "cagr": 0.4555666191921335,
      "calmar": 42.03178716614974,
      "max_drawdown": 0.010838621193795528,
      "sharpe": 3.3631051254203053,
      "sortino": 11.524107766149024,
      "total_return": 0.03665253454687489,
      "volatility": 0.11351560721915518
    }
  }
}
```
