# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_fourth_sleeve_probes/pair_spread_1h_core_ethusdt_solusdt_1.8_0.45.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- cross_sectional: 10.92%
- market_neutral: 70.00%
- trend: 19.08%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | pair_spread_1h_core_ethusdt_solusdt_1.8_0.45 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 4.195 | 3.07% | 1.613 | 1.69% |
| 2 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 35.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 3 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 19.08% | 4.428 | 4.99% | 2.295 | 2.95% |
| 4 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 10.92% | 1.641 | 1.84% | 1.464 | 3.24% |

## Portfolio metrics

- Fit (val): {"cagr": 0.38638985759282485, "calmar": 50.229514589737455, "max_drawdown": 0.007692486394677789, "sharpe": 6.006196853380296, "sortino": 9.761842346489846, "total_return": 0.028135934495976844, "volatility": 0.05465908044263039}
- Report (oos): {"cagr": 0.5170125355811246, "calmar": 37.6517547703428, "max_drawdown": 0.013731432671190147, "sharpe": 3.603606997649049, "sortino": 12.198143091903747, "total_return": 0.040770864086683956, "volatility": 0.11755193086167802}
- Train: {"cagr": 0.058713896960215006, "calmar": 0.9476565056178612, "max_drawdown": 0.061956939684526535, "sharpe": 0.6932622383970669, "sortino": 1.3533450514757137, "total_return": 0.058713896960215006, "volatility": 0.08780220829348129}
- Val: {"cagr": 0.38638985759282485, "calmar": 50.229514589737455, "max_drawdown": 0.007692486394677789, "sharpe": 6.006196853380296, "sortino": 9.761842346489846, "total_return": 0.028135934495976844, "volatility": 0.05465908044263039}
- OOS: {"cagr": 0.5170125355811246, "calmar": 37.6517547703428, "max_drawdown": 0.013731432671190147, "sharpe": 3.603606997649049, "sortino": 12.198143091903747, "total_return": 0.040770864086683956, "volatility": 0.11755193086167802}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5103978455126326,
      "calmar": 37.01029200575106,
      "max_drawdown": 0.013790700312046211,
      "sharpe": 3.5663914347875214,
      "sortino": 12.072169099366045,
      "total_return": 0.04033484228110762,
      "volatility": 0.11755193086167802
    },
    "x3": {
      "cagr": 0.5038119190611567,
      "calmar": 36.37640349933997,
      "max_drawdown": 0.013849965103622686,
      "sharpe": 3.5291758719259922,
      "sortino": 11.946195106828336,
      "total_return": 0.03989899792590057,
      "volatility": 0.11755193086167802
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.4559806382124796,
      "calmar": 36.877678797546345,
      "max_drawdown": 0.012364678393012585,
      "sharpe": 3.6036069976490483,
      "sortino": 12.198143091903743,
      "total_return": 0.03668080552040842,
      "volatility": 0.10579673777551021
    },
    "plus_10pct_signal": {
      "cagr": 0.5803911799247099,
      "calmar": 38.44472324446625,
      "max_drawdown": 0.015096770920525526,
      "sharpe": 3.6036069976490497,
      "sortino": 12.19814309190375,
      "total_return": 0.04486364577248447,
      "volatility": 0.12930712394784583
    }
  }
}
```
