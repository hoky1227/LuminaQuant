# Final Portfolio Validation\n\n- generated_at: `2026-03-19T11:50:04.467973+00:00`\n- status: `completed`\n- validation_oos_end: `2026-03-19T11:39:00Z`\n- recommended_action: `Refreshed validation materially diverged from the saved incumbent; review data freshness and deployment assumptions before final sign-off.`\n\n## OOS comparison\n\n| Metric | Saved artifact | Refreshed | Delta |\n|---|---:|---:|---:|\n| total_return | 4.5220% | 5.5613% | 1.0393% |\n| sharpe | 2.683 | 2.785 | 0.102 |\n| max_drawdown | 2.4890% | 2.4890% | 0.0000% |\n\n## Components\n\n| Name | Weight | OOS Return | OOS Sharpe | Portfolio Corr |\n|---|---:|---:|---:|---:|\n| `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | 33.33% | 3.0022% | 1.839 | 0.808 |\n| `topcap_tsmom_1h_balanced_16_4_0.015` | 33.33% | 6.2908% | 2.079 | 0.927 |\n| `pair_spread_1h_core_bnbusdt_trxusdt_2.2_0.55` | 33.33% | 7.2375% | 3.301 | 0.813 |\n\n## OOS monthly returns\n\n- `2026-02` | return=4.8171% | days=28\n- `2026-03` | return=0.7101% | days=20\n\n## Sensitivity\n\n```json\n{
  "cost_stress": {
    "x2": {
      "cagr": 0.49578849302095573,
      "calmar": 19.786369448244784,
      "max_drawdown": 0.02505707246181721,
      "sharpe": 2.7267988055084453,
      "sortino": 8.808552855679816,
      "total_return": 0.054378708496001016,
      "volatility": 0.15179884598129356
    },
    "x3": {
      "cagr": 0.4825361760517277,
      "calmar": 19.129862739697977,
      "max_drawdown": 0.025224236191218274,
      "sharpe": 2.6681113756769124,
      "sortino": 8.618971091674798,
      "total_return": 0.05314548005077402,
      "volatility": 0.1517988459812935
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.44975310354459586,
      "calmar": 20.057930548633703,
      "max_drawdown": 0.022422707190759117,
      "sharpe": 2.7854862353399783,
      "sortino": 8.986545038792178,
      "total_return": 0.05005312549545504,
      "volatility": 0.1366189613831642
    },
    "plus_10pct_signal": {
      "cagr": 0.5706592846874095,
      "calmar": 20.863347688053167,
      "max_drawdown": 0.02735223959356159,
      "sharpe": 2.7854862353399783,
      "sortino": 8.986545038792174,
      "total_return": 0.06117283559060005,
      "volatility": 0.16697873057942286
    }
  }
}\n```\n