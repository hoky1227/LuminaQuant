# Portfolio Optimization Report

- Source report: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/continue-the-autonomous-portfo/worktrees/worker-1/var/reports/exact_window_backtests/followup_status/autonomous_pair_exec_takeprofit_anchor_latest.json`
- Fit split: `val`
- Report split: `oos`
- Clusters: 2

## Sleeve budgets

- cross_sectional: 33.33%
- market_neutral: 33.33%
- trend: 33.33%

## Top strategy weights

| # | Name | Strategy | Family | TF | Weight | Fit Sharpe | Fit Return | Report Sharpe | Report Return |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80 | CompositeTrendStrategy | trend | 30m | 33.33% | 4.428 | 4.99% | 2.295 | 2.95% |
| 2 | topcap_tsmom_1h_balanced_16_4_0.015 | TopCapTimeSeriesMomentumStrategy | cross_sectional | 1h | 33.33% | 1.641 | 1.84% | 1.464 | 3.24% |
| 3 | pair_spread_1h_exec_tightstop_tp_bnbusdt_trxusdt_2.6_0.70 | PairSpreadZScoreStrategy | market_neutral | 1h | 33.33% | 5.000 | 1.61% | 7.481 | 11.10% |

## Portfolio metrics

- Fit (val): {"cagr": 0.38921854349697504, "calmar": 37.19521639060639, "max_drawdown": 0.010464209682492176, "sharpe": 3.8092446490654606, "sortino": 7.795811942041887, "total_return": 0.028313931887539612, "volatility": 0.08730416117436127}
- Report (oos): {"cagr": 0.7940650369074875, "calmar": 34.41387922891631, "max_drawdown": 0.023073976392648965, "sharpe": 3.3975187920959042, "sortino": 12.587795747235353, "total_return": 0.05764677056380729, "volatility": 0.17653272331436545}
- Train: {"cagr": -0.008242206815827435, "calmar": -0.10647314686277427, "max_drawdown": 0.07741113190211457, "sharpe": -0.037699820114395516, "sortino": -0.07351324792097677, "total_return": -0.008242206815827435, "volatility": 0.09666478849275657}
- Val: {"cagr": 0.38921854349697504, "calmar": 37.19521639060639, "max_drawdown": 0.010464209682492176, "sharpe": 3.8092446490654606, "sortino": 7.795811942041887, "total_return": 0.028313931887539612, "volatility": 0.08730416117436127}
- OOS: {"cagr": 0.7940650369074875, "calmar": 34.41387922891631, "max_drawdown": 0.023073976392648965, "sharpe": 3.3975187920959042, "sortino": 12.587795747235353, "total_return": 0.05764677056380729, "volatility": 0.17653272331436545}

## Sensitivity

```json
{
  "cost_stress": {
    "x2": {
      "cagr": 0.7847175227463519,
      "calmar": 33.86442884748582,
      "max_drawdown": 0.023172324159975055,
      "sharpe": 3.367881286429911,
      "sortino": 12.477988887991783,
      "total_return": 0.05711710850801377,
      "volatility": 0.17653272331436545
    },
    "x3": {
      "cagr": 0.7754185786529899,
      "calmar": 33.32172203100942,
      "max_drawdown": 0.023270663440844386,
      "sharpe": 3.3382437807639156,
      "sortino": 12.368182028748203,
      "total_return": 0.056587704127742944,
      "volatility": 0.17653272331436543
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.694495212481617,
      "calmar": 33.41229413805884,
      "max_drawdown": 0.020785618898599978,
      "sharpe": 3.397518792095906,
      "sortino": 12.58779574723536,
      "total_return": 0.051871705069206575,
      "volatility": 0.15887945098292888
    },
    "plus_10pct_signal": {
      "cagr": 0.8989295201566154,
      "calmar": 35.449370061310255,
      "max_drawdown": 0.025358123955430023,
      "sharpe": 3.3975187920959056,
      "sortino": 12.587795747235354,
      "total_return": 0.06342368212694183,
      "volatility": 0.194185995645802
    }
  }
}
```
