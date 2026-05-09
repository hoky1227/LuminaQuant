# Profit Moonshot Under-8GB Mission Validation

- Status: `running`
- Passed: `False`
- Result path: `.omx/specs/autoresearch-profit-moonshot-alpha-v2/result.json`
- Budget bytes: `8589934592`

## Checks
- FAIL `declared_pass_status` — result is not yet status=passed with passed=true
- PASS `passing_candidate_artifact_exists` — var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/passing_candidate_latest.json
- PASS `candidate_lifecycle_label` — research/live-equivalent success label present
- PASS `rss_under_8gib_evidence` — [{"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_memory_latest.json", "peak_rss_bytes": 771981312, "under_8gib": true}, {"exists": true, "path": "var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/portfolio_target_budget_v1/_memory_guard/profit_moonshot_fresh_portfolio_tuning_rss_latest.jsonl", "peak_rss_bytes": 771981312, "under_8gib": true}]
- PASS `local_tests_evidence` — 10 test evidence item(s)
- FAIL `ci_success_evidence` — missing successful CI evidence
- FAIL `git_push_evidence_if_source_changed` — source_changed requires git push evidence
