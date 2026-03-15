# autonomous portfolio risk search

- generated_at: `2026-03-15T09:46:47.974753+00:00`
- status: `discard`
- decision_reason: No bounded weight/vol-target/cost/correlation variant of the incumbent bundle beat the incumbent on locked-OOS return; none met the train-improvement + acceptable-OOS near-miss screen.
- run_count: `180`

## best OOS variant

- params: `{"correlation_threshold": 0.35, "cost_penalty": 0.0, "max_strategy_cap": 0.7, "target_vol": 0.06}`
- oos_total_return: 4.7692% vs incumbent 5.5960%
- oos_sharpe: 3.014 vs incumbent 3.447
- oos_max_dd: 1.8675% vs incumbent 1.4277%
- train_total_return: 2.8201% vs incumbent 3.1129%
- train_sharpe: 0.354 vs incumbent 0.402

## near miss count

- `0` variants met the train-improvement + acceptable-OOS screen.
