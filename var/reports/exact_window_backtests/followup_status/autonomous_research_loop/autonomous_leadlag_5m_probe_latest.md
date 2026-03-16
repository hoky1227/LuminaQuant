# autonomous leadlag 5m probe

- generated_at: `2026-03-16T10:21:34.175418+00:00`
- lane: targeted 5m lead-lag spillover exact-window screen
- status: `discard`
- run_id: `autonomous_leadlag_5m_20260316T1017Z`
- evaluated_count: 9
- best_variant: `leadlag_spillover_5m_0.50_lag4`
- best_variant_val_return: -42.7203%
- best_variant_oos_return: -75.3673%
- peak_rss_mib: 2899.4

## takeaway

- Worktree data visibility and the 5m lead-lag exact-window lane both executed successfully under the 8 GiB contract.
- The family is decisively negative on validation and OOS, so it is discarded without escalating to portfolio anchoring.
- Next backlog suggestion: advance to the next unused backlog lane, with alpha101_formula 1h now the highest-priority untested family.
