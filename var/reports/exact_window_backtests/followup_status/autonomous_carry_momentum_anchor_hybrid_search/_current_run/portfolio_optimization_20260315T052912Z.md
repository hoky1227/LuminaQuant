# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/var/reports/exact_window_backtests/followup_status/autonomous_carry_momentum_anchor_hybrid_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 3

## Sleeve budgets

- carry: 10.00%
- cross_sectional: 30.00%
- market_neutral: 30.00%
- trend: 30.00%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 30.00% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 30.00% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55 | PairSpreadZScoreStrategy | market_neutral | 1h | 30.00% | 3.880 | 1.61% | 4.557 | 7.24% |
| 4 | perp_crowding_carry_30m_0.25_0.08 | PerpCrowdingCarryStrategy | carry | 30m | 10.00% | 0.000 | 0.00% | 3.438 | 0.06% |

## Portfolio metrics

- Fit (val): {"cagr": 0.22640417319714845, "calmar": 21.906757654285766, "max_drawdown": 0.010334901073452807, "sharpe": 3.092796628789857, "sortino": 4.917608626535617, "total_return": 0.0254805616159397, "volatility": 0.06670705501828826}
- Report (oos): {"cagr": 0.5174905323967609, "calmar": 23.078860549453633, "max_drawdown": 0.022422707190759117, "sharpe": 2.6871067100581105, "sortino": 9.764606849962435, "total_return": 0.04080230566127896, "volatility": 0.15982704280717464}
- Train: {"cagr": 0.008030579688577122, "calmar": 0.09825088887365384, "max_drawdown": 0.08173544057096604, "sharpe": 0.13430419851570924, "sortino": 0.25716365599748636, "total_return": 0.008693488428764828, "volatility": 0.08842606596746613}
- Val: {"cagr": 0.22640417319714845, "calmar": 21.906757654285766, "max_drawdown": 0.010334901073452807, "sharpe": 3.092796628789857, "sortino": 4.917608626535617, "total_return": 0.0254805616159397, "volatility": 0.06670705501828826}
- OOS: {"cagr": 0.5174905323967609, "calmar": 23.078860549453633, "max_drawdown": 0.022422707190759117, "sharpe": 2.6871067100581105, "sortino": 9.764606849962435, "total_return": 0.04080230566127896, "volatility": 0.15982704280717464}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.5097412323303432,
      "calmar": 22.635944156309485,
      "max_drawdown": 0.022519106285578072,
      "sharpe": 2.655038371792063,
      "sortino": 9.648074553597926,
      "total_return": 0.04029146603873124,
      "volatility": 0.1598270428071747
    },
    "x3": {
      "cagr": 0.5020313973892929,
      "calmar": 22.198556690198636,
      "max_drawdown": 0.022615497232347348,
      "sharpe": 2.622970033526018,
      "sortino": 9.53154225723342,
      "total_return": 0.03978086998239605,
      "volatility": 0.15982704280717464
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.45710284387899347,
      "calmar": 22.63096015573352,
      "max_drawdown": 0.02019811977633601,
      "sharpe": 2.6871067100581096,
      "sortino": 9.764606849962432,
      "total_return": 0.036757397946128645,
      "volatility": 0.1438443385264572
    },
    "plus_10pct_signal": {
      "cagr": 0.5800034266839864,
      "calmar": 23.535868179643522,
      "max_drawdown": 0.02464338354790918,
      "sharpe": 2.68710671005811,
      "sortino": 9.764606849962435,
      "total_return": 0.04483906056875808,
      "volatility": 0.17580974708789213
    }
  }
}
```
