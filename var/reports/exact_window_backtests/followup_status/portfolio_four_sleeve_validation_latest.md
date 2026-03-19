# Final Portfolio Validation\n\n- generated_at: `2026-03-19T11:51:20.059569+00:00`\n- status: `completed`\n- validation_oos_end: `2026-03-19T11:39:00Z`\n- recommended_action: `Refreshed validation materially diverged from the saved incumbent; review data freshness and deployment assumptions before final sign-off.`\n\n## OOS comparison\n\n| Metric | Saved artifact | Refreshed | Delta |\n|---|---:|---:|---:|\n| total_return | 2.4114% | 7.2588% | 4.8474% |\n| sharpe | 0.163 | 0.429 | 0.266 |\n| max_drawdown | 4.9478% | 4.9481% | 0.0002% |\n\n## Components\n\n| Name | Weight | OOS Return | OOS Sharpe | Portfolio Corr |\n|---|---:|---:|---:|---:|\n| `composite_trend_stable_30m_stable_ls_highconv_ls_0.75_0.45_0.20_0.80` | 25.00% | 3.0673% | 1.878 | 0.829 |\n| `regime_breakout_1h_trend_ls_48_0.70` | 25.00% | 19.4116% | 2.824 | 0.952 |\n| `topcap_tsmom_1h_balanced_16_4_0.015` | 25.00% | 6.0116% | 1.992 | 0.865 |\n| `rolling_breakout_30m_guarded_ls_64_0.002` | 25.00% | 0.1094% | 0.026 | 0.281 |\n\n## OOS monthly returns\n\n- `2026-02` | return=2.8591% | days=1372\n- `2026-03` | return=4.2774% | days=328\n\n## Sensitivity\n\n```json\n{
  "cost_stress": {
    "x2": {
      "cagr": 0.0038890434010450114,
      "calmar": 0.04492414979748604,
      "max_drawdown": 0.08656910411385554,
      "sharpe": 0.12388249127200819,
      "sortino": 0.2432367483234459,
      "total_return": 0.01824262232349083,
      "volatility": 0.03663551579500871
    },
    "x3": {
      "cagr": -0.007256248007728727,
      "calmar": -0.05933770301881459,
      "max_drawdown": 0.12228730871884175,
      "sharpe": -0.18085271707749895,
      "sortino": -0.3571909384294084,
      "total_return": -0.03335062222280316,
      "volatility": 0.03663551579500871
    }
  },
  "param_drift": {
    "minus_10pct_signal": {
      "cagr": 0.01369179602267434,
      "calmar": 0.30725405805365164,
      "max_drawdown": 0.04456180696003542,
      "sharpe": 0.428617699621515,
      "sortino": 0.4850485628913342,
      "total_return": 0.06538622958676488,
      "volatility": 0.03297196421550784
    },
    "plus_10pct_signal": {
      "cagr": 0.016615714276201565,
      "calmar": 0.30547967322294495,
      "max_drawdown": 0.05439220914733367,
      "sharpe": 0.428617699621515,
      "sortino": 0.4850485628913343,
      "total_return": 0.07977464996649553,
      "volatility": 0.04029906737450959
    }
  }
}\n```\n