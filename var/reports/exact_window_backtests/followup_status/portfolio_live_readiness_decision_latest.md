# Portfolio Live Readiness Decision

- generated_at: `2026-03-19T11:53:35.774943+00:00`
- decision: `keep_incumbent`
- decision_basis: `bounded validation-only search and latest-tail finalist validation did not produce a challenger that beat the incumbent on risk-adjusted locked-OOS performance while preserving drawdown discipline`
- refresh_cutoff_utc: `2026-03-19T11:39:51Z`
- feature_common_tail_utc: `2026-03-19T11:39:00Z`
- oos_start: `2026-02-01T00:00:00Z`
- direct_team_status: `blocked`
- fallback_used: `local_ralph_execution`

## Final ranked summary

| Candidate | OOS Return | OOS Sharpe | Max Drawdown |
|---|---:|---:|---:|
| incumbent | 6.0592% | 3.174 | 1.4277% |
| weight_only | 5.5613% | 2.785 | 2.4890% |
| anchored_four_sleeve | 7.2588% | 0.429 | 4.9481% |

## Promotion outcome

- incumbent: keep
- weight_only: reject
- anchored_four_sleeve: reject

## Rejection reasons

- weight_only: latest-tail OOS return below incumbent, latest-tail OOS sharpe below incumbent, latest-tail max drawdown above incumbent
- anchored_four_sleeve: latest-tail OOS sharpe materially below incumbent, latest-tail max drawdown materially above incumbent, robustness profile unsuitable for live promotion despite higher raw return

## Memory

- refresh_peak_rss_mib: `2422.03515625`
- incumbent_validation_peak_rss_mib: `1130.8046875`
- weight_search_peak_rss_mib: `41.40625`

## Next steps

- Keep the current one-shot incumbent as the live candidate for now.
- Do not promote the weight-only or anchored four-sleeve challengers.
- If more upside is required, run a separately approved sleeve-local retune program with the same validation-only and 8 GiB constraints.
- Before live deployment, add execution-layer checks outside this portfolio study: brokerage connectivity, order sizing, slippage, kill-switches, and monitoring.
