# autonomous carry 30m probe

- generated_at: `2026-03-16T10:16:43.289469+00:00`
- lane: targeted 30m perp-crowding carry exact-window rerun
- status: `discard`
- run_id: `autonomous_carry_30m_20260316T1011Z`
- evaluated_count: 3
- best_variant: `perp_crowding_carry_30m_0.25_0.08`
- best_variant_oos_return: 0.0260%
- anchor_portfolio_oos_return: 3.4074% vs incumbent 5.7628%
- peak_rss_mib: 3473.8

## takeaway

- Carry support now resolves correctly in the worktree and the targeted 30m lane ran successfully under the memory cap.
- The best carry sleeve had only trace OOS return and zero validation participation, so the anchored challenger remains a discard.
- Next backlog suggestion: advance to the next unused backlog lane, with leadlag_spillover 5m now the highest-priority untested family.
