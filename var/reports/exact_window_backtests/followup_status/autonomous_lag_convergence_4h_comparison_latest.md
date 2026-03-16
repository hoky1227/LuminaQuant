# portfolio four-sleeve comparison

- generated_at: `2026-03-16T11:41:30.002615+00:00`
- selection_basis: `autonomous_lag_convergence_4h_anchor`

- current_one_shot_incumbent: return=5.7628% | sharpe=3.506 | sortino=12.200 | calmar=55.594 | max_dd=1.4277%
- equal_weight_diagnostic: return=5.8773% | sharpe=0.346 | sortino=0.487 | calmar=0.183 | max_dd=6.9219%
- prior_exact_window_frozen_tuned: return=2.6260% | sharpe=0.210 | sortino=0.250 | calmar=0.085 | max_dd=6.7676%
- anchored_four_sleeve_tuned: return=3.1332% | sharpe=4.103 | sortino=12.877 | calmar=62.100 | max_dd=0.6111%

## rolling gate
- selection_basis: `train_val_only`
- survives_train_val: `True`
- recommended_action: `activate_conditionally`
