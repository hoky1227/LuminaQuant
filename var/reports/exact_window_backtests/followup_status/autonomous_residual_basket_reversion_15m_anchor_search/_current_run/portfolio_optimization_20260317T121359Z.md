# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_residual_basket_reversion_15m_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 44.56%
- market_neutral: 30.00%
- trend: 25.44%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 2 | residual_basket_reversion_15m_resid_btc_guarded_lo_64_2.20 | ResidualBasketReversionStrategy | cross_sectional | 15m | 30.00% | -5.493 | -2.55% | -0.824 | -0.63% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 25.44% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 14.56% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.15674812123516157, "calmar": 22.270106707393534, "max_drawdown": 0.0070384988852847385, "sharpe": 3.082853851299886, "sortino": 4.955173661391592, "total_return": 0.012443897026465756, "volatility": 0.04759790203575197}
- Report (oos): {"cagr": 0.39516781558677905, "calmar": 52.57024982751729, "max_drawdown": 0.007516947644025329, "sharpe": 3.1127544242831893, "sortino": 12.532938409875971, "total_return": 0.03244824326566076, "volatility": 0.10885535893794035}
- Train: {"cagr": -0.08073084178351819, "calmar": -0.7495788540555361, "max_drawdown": 0.10770159983400074, "sharpe": -1.3345230586775763, "sortino": -2.413753854940845, "total_return": -0.08073084178351819, "volatility": 0.06165372215118462}
- Val: {"cagr": 0.15674812123516157, "calmar": 22.270106707393534, "max_drawdown": 0.0070384988852847385, "sharpe": 3.082853851299886, "sortino": 4.955173661391592, "total_return": 0.012443897026465756, "volatility": 0.04759790203575197}
- OOS: {"cagr": 0.39516781558677905, "calmar": 52.57024982751729, "max_drawdown": 0.007516947644025329, "sharpe": 3.1127544242831893, "sortino": 12.532938409875971, "total_return": 0.03244824326566076, "volatility": 0.10885535893794035}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.3891320109727008,
      "calmar": 51.20406655810315,
      "max_drawdown": 0.0075996309889008185,
      "sharpe": 3.0728898585956426,
      "sortino": 12.372431001196107,
      "total_return": 0.03201909926698199,
      "volatility": 0.10885535893794035
    },
    "x3": {
      "cagr": 0.3831222473509286,
      "calmar": 49.87071931176211,
      "max_drawdown": 0.007682308429438844,
      "sharpe": 3.0330252929080967,
      "sortino": 12.211923592516246,
      "total_return": 0.03159012855030041,
      "volatility": 0.10885535893794035
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.350170068581358,
      "calmar": 51.74655318117875,
      "max_drawdown": 0.006767022092376229,
      "sharpe": 3.1127544242831897,
      "sortino": 12.532938409875975,
      "total_return": 0.02920764747281801,
      "volatility": 0.0979698230441463
    },
    "plus_10pct_signal": {
      "cagr": 0.4415021565662922,
      "calmar": 53.40872251507602,
      "max_drawdown": 0.008266480375778817,
      "sharpe": 3.1127544242831897,
      "sortino": 12.532938409875975,
      "total_return": 0.03568780855392606,
      "volatility": 0.11974089483173438
    }
  }
}
```
