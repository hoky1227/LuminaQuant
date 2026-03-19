# Final Portfolio Validation\n\n- generated_at: `2026-03-19T14:18:23.026867+00:00`\n- status: `completed`\n- validation_oos_end: `2026-03-19T14:10:00Z`\n- recommended_action: `Saved incumbent remains directionally consistent on refreshed data; proceed with manual final review and evidence collation.`\n\n## OOS comparison\n\n| Metric | Saved artifact | Refreshed | Delta |\n|---|---:|---:|---:|\n| total_return | 5.7628% | 6.0803% | 0.3175% |\n| sharpe | 3.506 | 3.185 | -0.321 |\n| max_drawdown | 1.4277% | 1.4277% | 0.0000% |\n\n## Components\n\n| Name | Weight | OOS Return | OOS Sharpe | Portfolio Corr |\n|---|---:|---:|---:|---:|\n| `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | 60.00% | 7.2375% | 3.301 | 0.941 |\n| `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | 25.44% | 3.0993% | 1.896 | 0.722 |\n| `topcap_tsmom_1h_balanced_16_4_0.015` | 14.56% | 6.2615% | 2.070 | 0.807 |\n\n## OOS monthly returns\n\n- `2026-02` | return=5.8626% | days=28\n- `2026-03` | return=0.2056% | days=20\n\n## Sensitivity\n\n```json\n{
  "cost_stress": {
    "x2": {
      "cagr": 0.555886608908835,
      "calmar": 38.58612338816049,
      "max_drawdown": 0.014406386547745287,
      "sharpe": 3.1377771778906287,
      "sortino": 10.25325138505209,
      "total_return": 0.059854905857185337,
      "volatility": 0.14414061342771864
    },
    "x3": {
      "cagr": 0.5453410217022989,
      "calmar": 37.51826434542817,
      "max_drawdown": 0.014535347815703314,
      "sharpe": 3.0905390450918677,
      "sortino": 10.098892288440977,
      "total_return": 0.05890742731274079,
      "volatility": 0.14414061342771864
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.4991024451760908,
      "calmar": 38.82296938844958,
      "max_drawdown": 0.012855854486096607,
      "sharpe": 3.1850153106893906,
      "sortino": 10.317540544730791,
      "total_return": 0.054685613234852326,
      "volatility": 0.12972655208494674
    },
    "plus_10pct_signal": {
      "cagr": 0.6366141778292667,
      "calmar": 40.55488358835992,
      "max_drawdown": 0.01569759598599829,
      "sharpe": 3.18501531068939,
      "sortino": 10.317540544730793,
      "total_return": 0.06692871813346701,
      "volatility": 0.1585546747704905
    }
  }
}\n```\n