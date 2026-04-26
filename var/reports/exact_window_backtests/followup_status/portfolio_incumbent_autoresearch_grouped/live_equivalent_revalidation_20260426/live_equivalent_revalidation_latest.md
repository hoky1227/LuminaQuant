# Live-equivalent candidate revalidation — 2026-04-26

Generated: `2026-04-26T11:28:21.596485Z`

## 기준 변경

- 기존 full-universe/legacy/HYBRID 리포트의 일별 return stream 점수는 이제 **연구 참고값**이다.
- 실투자 후보는 동일한 `ArtifactPortfolioModeStrategy`를 live와 backtest가 같이 사용하고, `SimulatedExecutionHandler`/`Portfolio`를 통과한 이벤트 기반 결과만 selection eligible이다.
- OOS는 계속 report-only이며, selection/tuning/health prior에는 쓰지 않는다.
- cash efficiency는 점수에 넣지 않는다. MDD 25%까지 허용한다.

## 결론

- Best full-universe live-equivalent candidate: `NONE` — train/val 원시 market-data 기반 engine backtest가 완료된 후보가 없다.
- Best deployable true-HYBRID candidate: `NONE` — dynamic/true HYBRID는 아직 live-equivalent engine validation 미완료다.
- Conservative fallback/shadow: `risk_off_mode` (`eligible_conservative_cash_fallback`)

## 왜 이전 val return을 그대로 쓰면 안 되는가

이전 랭킹은 저장된 artifact/일별 return stream을 재조합했다. 거래 엔진의 주문 크기, 수수료/슬리피지, 체결 이벤트, 컴포넌트별 EXIT 처리, live portfolio mode symbol universe를 같은 경로로 강제하지 않았다. 따라서 실투자 승격 기준으로는 모두 재검증 대상이다.

## Live portfolio mode preflight / revalidation

| rank | mode | status | score | val ret | val Sharpe | val MDD | symbols/blocker |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `legacy_no_highvol_hybrid_mode` | `ready_for_live_equivalent_backtest` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT |
| 2 | `pair_fast_exit` | `ready_for_live_equivalent_backtest` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT |
| 3 | `pair_tactical_mode` | `ready_for_live_equivalent_backtest` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT |
| 4 | `state_vwap_pair` | `ready_for_live_equivalent_backtest` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT |
| 5 | `wave2_pair` | `ready_for_live_equivalent_backtest` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT |
| 6 | `hybrid_guarded_mode` | `eligible_conservative_cash_fallback` | n/a | +0.00% | 0.0000 | +0.00% |  |
| 7 | `risk_off_mode` | `eligible_conservative_cash_fallback` | n/a | +0.00% | 0.0000 | +0.00% | BNB/USDT,TRX/USDT,BTC/USDT,ETH/USDT,SOL/USDT |
| 8 | `aggressive_realized_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 9 | `autoresearch_55_45` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete |
| 10 | `balanced_overlay_80_20` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 11 | `balanced_overlay_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 12 | `blend_85_15` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 13 | `core_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 14 | `defensive_overlay_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 15 | `incumbent` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 16 | `incumbent_only` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 17 | `production_guarded_portfolio` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 18 | `production_guarded_state_vwap_pair_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 19 | `retuned_live_portfolio_hybrid_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 20 | `soft_three_way_regime` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 21 | `static_blend_76_24` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 22 | `strict_autoresearch_1x` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete |
| 23 | `strict_autoresearch_practical_mode` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |
| 24 | `three_way_regime` | `blocked_missing_raw_first_market_data` | n/a | +0.00% | 0.0000 | +0.00% | BTC/USDT:train_raw_first_incomplete;BTC/USDT:val_raw_first_incomplete;ETH/USDT:train_raw_first_incomplete;ETH/USDT:val_raw_first_incomplete;SOL/USDT:train_raw_first_incomplete;SOL/USDT:val_raw_first_incomplete |

## Research artifact reset sample

| research rank | name | previous score | mapped live mode | reset status |
|---:|---|---:|---|---|
| 1 | `legacy_metric_no_highvol_baseline_raw_score` | 95.4113 | `legacy_no_highvol_hybrid_mode` | `requires_live_equivalent_engine_backtest` |
| 2 | `legacy_metric_highvol_active_raw_score` | 92.5170 | `-` | `research_only_not_live_selectable` |
| 3 | `final_scaled_v35_train_high_vol` | 90.3846 | `-` | `research_only_not_live_selectable` |
| 4 | `full_hybrid_val_only_no_cash_penalty` | 90.1061 | `-` | `research_only_not_live_selectable` |
| 5 | `legacy_full_hybrid_wave2_val_only` | 90.1061 | `-` | `research_only_not_live_selectable` |
| 6 | `legacy_metric_highvol_optional_raw_score` | 89.5358 | `-` | `research_only_not_live_selectable` |
| 7 | `source_three_way_regime` | 89.0714 | `three_way_regime` | `requires_live_equivalent_engine_backtest` |
| 8 | `final_scaled_v36_dynamic_high_vol` | 88.5026 | `-` | `research_only_not_live_selectable` |
| 9 | `static_challenger_static_anchor_val_return` | 88.3893 | `-` | `research_only_not_live_selectable` |
| 10 | `static_challenger_static_dominant_val_return` | 88.0278 | `-` | `research_only_not_live_selectable` |
| 11 | `final_scaled_baseline_no_hv` | 87.7476 | `-` | `research_only_not_live_selectable` |
| 12 | `source_soft_three_way_regime` | 87.3457 | `soft_three_way_regime` | `requires_live_equivalent_engine_backtest` |
| 13 | `legacy_robust_val_low_cash_no_oos_health` | 86.6257 | `-` | `research_only_not_live_selectable` |
| 14 | `source_static_blend_76_24` | 86.4183 | `static_blend_76_24` | `requires_live_equivalent_engine_backtest` |
| 15 | `retuned_full_universe_hybrid_online` | 85.8314 | `-` | `research_only_not_live_selectable` |
| 16 | `full_hybrid_val_low_cash` | 85.2615 | `-` | `research_only_not_live_selectable` |
| 17 | `legacy_full_hybrid_val_low_cash` | 85.2615 | `-` | `research_only_not_live_selectable` |
| 18 | `legacy_robust_val_aggressive_cash_no_oos_health` | 85.2440 | `-` | `research_only_not_live_selectable` |
| 19 | `composite_trend_30m_0.45_0.60_0.20_0.80` | 83.6935 | `-` | `research_only_not_live_selectable` |
| 20 | `composite_trend_30m_0.60_0.75_0.20_0.80` | 83.6556 | `-` | `research_only_not_live_selectable` |
| 21 | `composite_trend_30m_0.60_0.75_0.25_0.80` | 83.6556 | `-` | `research_only_not_live_selectable` |
| 22 | `composite_trend_30m_0.45_0.75_0.20_0.80` | 83.6332 | `-` | `research_only_not_live_selectable` |
| 23 | `composite_trend_30m_0.45_0.60_0.25_0.80` | 83.5911 | `-` | `research_only_not_live_selectable` |
| 24 | `composite_trend_30m_0.45_0.75_0.25_0.80` | 83.5757 | `-` | `research_only_not_live_selectable` |
| 25 | `composite_trend_30m_0.60_0.60_0.25_0.80` | 83.5624 | `-` | `research_only_not_live_selectable` |
| 26 | `composite_trend_30m_0.60_0.60_0.20_0.80` | 83.5528 | `-` | `research_only_not_live_selectable` |
| 27 | `composite_trend_30m_0.75_0.75_0.20_0.80` | 83.4760 | `-` | `research_only_not_live_selectable` |
| 28 | `composite_trend_30m_0.75_0.75_0.25_0.80` | 83.4760 | `-` | `research_only_not_live_selectable` |
| 29 | `composite_trend_30m_0.75_0.60_0.25_0.80` | 82.8943 | `-` | `research_only_not_live_selectable` |
| 30 | `composite_trend_30m_0.75_0.60_0.20_0.80` | 82.8469 | `-` | `research_only_not_live_selectable` |
| 31 | `legacy_static_anchor_true_hybrid_val` | 82.7210 | `-` | `research_only_not_live_selectable` |
| 32 | `legacy_static_anchor_true_hybrid_low_cash` | 82.6909 | `-` | `research_only_not_live_selectable` |
| 33 | `composite_trend_30m_0.60_0.60_0.25_0.95` | 81.5092 | `-` | `research_only_not_live_selectable` |
| 34 | `composite_trend_30m_0.60_0.60_0.20_0.95` | 81.4826 | `-` | `research_only_not_live_selectable` |
| 35 | `full_hybrid_val_very_low_cash` | 81.3806 | `-` | `research_only_not_live_selectable` |
| 36 | `legacy_full_hybrid_val_very_low_cash` | 81.3806 | `-` | `research_only_not_live_selectable` |
| 37 | `composite_trend_30m_0.45_0.60_0.20_0.95` | 80.8341 | `-` | `research_only_not_live_selectable` |
| 38 | `composite_trend_30m_0.45_0.60_0.25_0.95` | 80.8294 | `-` | `research_only_not_live_selectable` |
| 39 | `composite_trend_30m_0.75_0.75_0.20_0.95` | 80.7708 | `-` | `research_only_not_live_selectable` |
| 40 | `composite_trend_30m_0.75_0.75_0.25_0.95` | 80.7708 | `-` | `research_only_not_live_selectable` |

## 명시적 caveats

- 현재 repo에는 BNB/TRX pair 계열 train/val raw-first coverage는 있으나, BTC/ETH/SOL 포함 포트폴리오의 exact-window raw-first coverage가 부족하다. 그래서 다중자산 포트폴리오 후보는 `blocked_missing_raw_first_market_data`로 내려갔다.
- 이 리포트의 핵심 변경은 `좋아 보이는 연구 점수`를 promotion evidence로 쓰지 않고, live-equivalent engine path를 통과한 후보만 승격시키는 것이다.
- raw-first coverage를 채운 뒤 `--execute-backtests`로 같은 스크립트를 다시 실행하면 같은 live portfolio mode 후보들이 train/val/OOS로 자동 재랭킹된다.
