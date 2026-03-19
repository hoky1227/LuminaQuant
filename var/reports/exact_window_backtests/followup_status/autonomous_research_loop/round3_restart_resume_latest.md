# Round-3 Restart Resume

- generated_at: `2026-03-17T14:22:57.528010+00:00`
- status: `ready_for_new_session`
- branch: `private-main`

## Incumbent to keep using
- artifact: `var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- locked_oos_return: `5.7628%`
- locked_oos_sharpe: `3.506`
- locked_oos_max_drawdown: `1.4277%`

## Exhausted work so far
- original new-hypothesis refresh queue: exhausted without improvement
- post-refresh shortlist items 1-5: exhausted without improvement
- round-2 shortlist items 1-5: exhausted without improvement

## Deferred item still waiting
- `pair_spread_4h_xpt_xpd_retune_when_coverage_matures`
- do not run before `2026-03-31T10:15:00+00:00`

## Best partial result from the latest exploratory rounds
- `Session-gated residual basket reversion 5m` improved drawdown versus the incumbent but still failed on locked-OOS return and Sharpe, so it remained a discard.

## Recommended next move
- Generate a fresh round-3 shortlist with genuinely new families rather than retuning the exhausted set.
- If you do not want new idea generation, stop until the deferred 2026-03-31 retune window.

## New-session starter prompt
```text
Continue the autonomous portfolio research loop from the saved follow-up artifacts. The incumbent remains var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json. Use var/reports/exact_window_backtests/followup_status/autonomous_research_loop/round3_restart_resume_latest.md and research_state_latest.json as the restart context. Treat the original new-hypothesis queue, the post-refresh shortlist, and the round-2 shortlist as exhausted. Do not run the deferred pair_spread_4h_xpt_xpd retune before 2026-03-31T10:15:00+00:00. Start by generating a round-3 shortlist of materially new families, then run only one heavy lane at a time under the 8 GiB RSS rule, update experiments.tsv / research_state / ideas_backlog / probe / decision artifacts, and report OOS return, Sharpe, max drawdown, peak RSS, delta vs incumbent, and keep/discard/crash after each lane.
```

## Optional shell entrypoint
```bash
cd /home/hoky/Quants-agent/LuminaQuant
omx team ralph 1:executor "Continue the autonomous portfolio research loop from the saved follow-up artifacts. The incumbent remains var/reports/exact_window_backtests/followup_status/portfolio_one_shot_current_opt/portfolio_optimization_latest.json. Use var/reports/exact_window_backtests/followup_status/autonomous_research_loop/round3_restart_resume_latest.md and research_state_latest.json as the restart context. Treat the original new-hypothesis queue, the post-refresh shortlist, and the round-2 shortlist as exhausted. Do not run the deferred pair_spread_4h_xpt_xpd retune before 2026-03-31T10:15:00+00:00. Start by generating a round-3 shortlist of materially new families, then run only one heavy lane at a time under the 8 GiB RSS rule, update experiments.tsv / research_state / ideas_backlog / probe / decision artifacts, and report OOS return, Sharpe, max drawdown, peak RSS, delta vs incumbent, and keep/discard/crash after each lane."
```
