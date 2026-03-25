import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export interface ExactWindowPayload {
  as_of: string;
  generated_at: string | null;
  status: string;
  error: string | null;
  root: string;
  run_root: string;
  summary: {
    candidate_count: number;
    evaluated_count: number;
    promoted_count: number;
    btc_beating_candidate_count: number;
    provisional_candidate_count: number;
    candidate_pool_count: number;
    requested_timeframes: string[];
    requested_symbols: string[];
    low_ram_profile: boolean;
  };
  decision: {
    next_action: string;
    promoted_total: number;
    total_evaluated: number;
    max_peak_rss_mib: number | null;
    valid_strategy_found: boolean;
  };
  memory: {
    status: string;
    peak_rss_mib: number | null;
    soft_limit_mib: number | null;
    hard_limit_mib: number | null;
  };
  portfolio: {
    construction_basis: string;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
  };
  time_window: Record<string, string>;
  timeframes: Array<{
    timeframe: string;
    candidate_id: string;
    name: string;
    family: string;
    promoted: boolean;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
    trade_count: number | null;
    reject_reasons: string[];
  }>;
  top_candidates: Array<{
    timeframe: string;
    candidate_id: string;
    name: string;
    family: string;
    promoted: boolean;
    oos_return: number | null;
    oos_sharpe: number | null;
    oos_max_drawdown: number | null;
    trade_count: number | null;
    reject_reasons: string[];
  }>;
  portfolio_weights: Array<{
    name: string;
    timeframe: string;
    family: string;
    weight: number | null;
    oos_return: number | null;
    oos_sharpe: number | null;
  }>;
  notes: Array<{ label: string; value: string }>;
  warnings: string[];
}

export async function loadExactWindowFromPython(): Promise<ExactWindowPayload> {
  return runUvPythonSnippetJson<ExactWindowPayload>(`
from lumina_quant.dashboard.exact_window_service import load_exact_window_summary_payload
import json
print(json.dumps(load_exact_window_summary_payload(), sort_keys=True))
`);
}
