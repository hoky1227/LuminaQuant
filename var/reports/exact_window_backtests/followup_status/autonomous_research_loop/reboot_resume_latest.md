# Autonomous Research Reboot Resume

- saved_at_utc: `20260314T161736Z`
- branch: `private-main`
- context_snapshot: `.omx/context/autonomous-portfolio-research-loop-resume-20260314T161736Z.md`
- reboot_resume_plan: `.omx/plans/reboot-resume-autonomous-portfolio-research-20260314T161736Z.md`

## What is already done
- autonomous research loop infra implemented:
  - `src/lumina_quant/workflows/autonomous_portfolio_research_loop.py`
  - `src/lumina_quant/cli/autonomous_research.py`
  - `scripts/run_autonomous_portfolio_research_loop.py`
- current autonomous artifact index exists:
  - `var/reports/exact_window_backtests/followup_status/autonomous_research_loop/experiments.tsv`
  - `.../stack_audit_latest.md`
  - `.../ideas_backlog_latest.md`
  - `.../research_state_latest.json`
  - `.../english_sources_latest.md`
- current ledger counts:
  - `keep=2`
  - `discard=15`
  - `crash=2`

## Best research keep so far
- candidate id: `portfolio::autonomous_cross_sectional_1h_stability_pair_topcap_opt`
- sleeves:
  - `pair_spread_1h_core_ethusdt_solusdt_1.8_0.45`
  - `topcap_tsmom_1h_balanced_16_4_0.015`
- metrics:
  - train total return: `-0.0817%`
  - val total return: `+2.4956%`
  - OOS total return: `+2.5471%`
  - OOS Sharpe: `1.8734`
  - promotion-score delta vs incumbent: `+0.1069`
- status:
  - **research keep only**
  - **not promotable yet** because OOS total return is still below incumbent under the current gate.

## Latest completed heavy runs
- `autonomous_intraday_5m_20260314T145221Z` -> discard
- `autonomous_cross_sectional_1h_20260314T150019Z` -> discard as a full bundle, but yielded the 2-sleeve keep candidate above
- `autonomous_mixed_assets_4h_20260314T155623Z` -> discard
- `autonomous_mixed_assets_30m_20260314T161124Z` -> completed, **next unresolved continuation point**

## Memory checkpoints
- exact-window budget is explicit: `8589934592` bytes
- recent heavy run peaks:
  - 5m run: ~2.63 GiB
  - 1h run: ~2.83 GiB
  - 4h mixed-assets run: ~1.56 GiB
  - 30m mixed-assets run: ~1.55 GiB

## Immediate next step after reboot
- Continue from the **30m mixed-assets batch**:
  - `var/reports/exact_window_backtests/autonomous_mixed_assets_30m_20260314T161124Z/30m/exact_window_suite_summary_latest.json`
- Objective:
  - build filtered 30m candidate subsets,
  - run portfolio optimization on them,
  - compare against incumbent,
  - append keep/discard to the ledger,
  - then proceed to the next heavy lane.

## First command to run in the new session
```bash
cd /home/hoky/Quants-agent/LuminaQuant && uv run lq autonomous-research --max-archive-crashes 2
```
