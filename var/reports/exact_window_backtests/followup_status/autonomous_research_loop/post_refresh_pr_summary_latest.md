# Post-refresh PR / Commit Summary

- generated_at: `2026-03-17T11:31:52.901213+00:00`
- incumbent: `portfolio_one_shot_current_opt/portfolio_optimization_latest.json`
- incumbent_locked_oos: return `5.7628%`, sharpe `3.506`, max_dd `1.4277%`
- outcome: `completed_no_promotion`

## Scope

- Drained the remaining new-hypothesis refresh queue end-to-end.
- Reused existing exact-window evidence where duplication risk was high and ran fresh heavy lanes only where the family was newly materialized.
- Updated probe / anchor / comparison / decision artifacts, `experiments.tsv`, `research_state_latest.json`, backlog markdown, and reboot/conclusion handoff notes.

## Queue results

| Lane | Status | OOS Return | Sharpe | Max DD | Peak RSS MiB | Δ Return | Δ Sharpe | Δ Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `sector-dispersion-reversion 30m` | `discard` | 1.8952% | 1.565 | 2.6446% | 3541.164 | -3.8676% | -1.941 | 1.2168% |
| `single-asset-zscore-reversion 15m` | `discard` | 2.7005% | 2.435 | 1.6926% | 2981.020 | -3.0623% | -1.071 | 0.2649% |
| `liquidity-shock-reversion 5m` | `discard` | 3.3317% | 3.363 | 0.9857% | 2808.316 | -2.4311% | -0.143 | -0.4421% |
| `regime-conditioned-composite-trend 30m` | `discard` | 4.0998% | 2.750 | 1.5384% | 3219.531 | -1.6630% | -0.756 | 0.1107% |
| `topcap-rotation-relative-momentum 1h` | `discard` | 2.3650% | 1.588 | 3.0478% | 3096.969 | -3.3978% | -1.918 | 1.6201% |

## New code / research surface added

- Added bounded `30m` sector-dispersion pair candidates so the queued lane was actually executable.
- Added a new `LiquidityShockReversionStrategy` family plus candidate-library coverage for `5m` and `15m`.
- Reused the existing anchored four-sleeve search instead of creating another portfolio-search path.

## Verification

- `uv run ruff check src/lumina_quant/strategy_defaults.py src/lumina_quant/strategy_factory/candidate_library.py src/lumina_quant/strategy_factory/research_runner.py tests/test_strategy_factory_library.py`
- `uv run pytest tests/test_strategy_factory_library.py -q` → `21 passed`
- LSP diagnostics on changed source/test files → `0 diagnostics`

## Simplifications made

- Preferred evidence reuse over rerunning duplicate heavy lanes.
- Kept the incumbent-centered anchored portfolio comparison contract intact.
- Limited fresh heavy work to the two lanes that needed new execution evidence: `sector-dispersion-reversion 30m` and `liquidity-shock-reversion 5m`.

## Remaining risks

- The new liquidity-shock family is still a first-pass heuristic implementation.
- The sector-dispersion 30m lane is implemented as a bounded crypto pair-spread approximation, not a full cross-sectional residual engine.
- Exact-window runs remain timeframe-sliced rather than family-isolated, so each run still evaluates neighboring families in the same timeframe.

## Recommended next move

- Do not recycle the exhausted queue again.
- Either wait for the deferred `2026-03-31T10:15:00+00:00` XPT/XPD retune window, or select one of the genuinely new family ideas in `post_refresh_new_family_shortlist_latest.md`.

## Post-refresh shortlist execution

| Lane | Status | OOS Return | Sharpe | Max DD | Peak RSS MiB | Δ Return | Δ Sharpe | Δ Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `BTC-beta-neutral single-asset zscore reversion 15m` | `discard` | 3.1744% | 2.414 | 1.6215% | 3136.551 | -2.5884% | -1.092 | 0.1938% |
| `Session-transition liquidity vacuum fade 5m` | `discard` | 2.9938% | 2.675 | 1.4309% | 2850.824 | -2.7690% | -0.831 | 0.0031% |
| `Vol-of-vol exhaustion fade 15m` | `discard` | 4.0743% | 2.683 | 2.2423% | 3139.344 | -1.6885% | -0.823 | 0.8145% |
| `Breadth-thrust failure reversal 30m` | `discard` | 4.0743% | 2.683 | 2.2423% | 3558.875 | -1.6885% | -0.823 | 0.8145% |
| `Cross-sectional residual basket reversion 15m` | `discard` | 3.2448% | 3.113 | 0.7517% | 3139.344 | -2.5180% | -0.393 | -0.6760% |

## Follow-up conclusion

- All five post-refresh shortlist implementations were processed and all were discarded.
- The closest partial improvement was `Cross-sectional residual basket reversion 15m`, which improved drawdown materially but still missed the incumbent on return and Sharpe.
- The correct next move is no longer to extend this shortlist; it is to wait for the deferred retune window or generate a new round-2 shortlist.

## Round-2 execution progress

- `Funding / liquidation crowding fade 30m` → `discard` | OOS return=4.0743% | Sharpe=2.683 | max DD=2.2423% | peak RSS=3594.805 MiB
- `Basis snapback reversion 30m` → `discard` | OOS return=4.0743% | Sharpe=2.683 | max DD=2.2423% | peak RSS=3560.578 MiB
- `Session-gated residual basket reversion 5m` → `discard` | OOS return=3.1728% | Sharpe=2.845 | max DD=1.2568% | peak RSS=2831.934 MiB
- `Cross-asset liquidation contagion fade 5m` → `discard` | OOS return=4.0743% | Sharpe=2.683 | max DD=2.2423% | peak RSS=2831.934 MiB
- `Multi-horizon trend exhaustion fade 30m` → `discard` | OOS return=3.4505% | Sharpe=3.109 | max DD=1.2627% | peak RSS=3552.691 MiB

- Round-2 items 1-5 are now fully exhausted without improvement.
- The best partial round-2 result was `Session-gated residual basket reversion 5m`, which improved drawdown but still missed the incumbent on return and Sharpe.
- The next autonomous step should be a new round-3 shortlist, not another retune of round-2.
