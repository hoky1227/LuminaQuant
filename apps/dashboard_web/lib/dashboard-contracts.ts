export interface BridgeErrorBody {
  detail?: string;
  error?: string;
  ok?: boolean;
}

export interface OverviewMetric {
  key: string;
  label: string;
  value: number | string | null;
}

export interface OverviewPoint {
  timestamp: string;
  equity?: number;
  drawdown?: number;
}

export interface OverviewPayload {
  as_of: string;
  summary_metrics: OverviewMetric[];
  performance_metrics: {
    cagr?: number;
    annualized_volatility?: number;
    sharpe_ratio?: number;
    sortino_ratio?: number;
    calmar_ratio?: number;
    max_drawdown?: number;
  };
  recent_runs: Array<{
    run_id: string;
    mode: string;
    status: string;
    strategy: string;
    started_at: string | null;
  }>;
  workflow_jobs: WorkflowJobRecord[];
  equity_curve: Array<OverviewPoint & { equity: number }>;
  drawdown_curve: Array<OverviewPoint & { drawdown: number }>;
  source: {
    mode: string;
    backend: string;
    status: string;
    run_id?: string;
  };
}

export interface WorkflowJobRecord {
  job_id: string;
  workflow: string;
  status: string;
  requested_mode: string;
  strategy: string;
  run_id: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface WorkflowJobsPayload {
  jobs: WorkflowJobRecord[];
  status: string;
}

export interface RiskHealthPayload {
  as_of: string;
  run_id: string;
  summary: {
    risk_event_count: number;
    heartbeat_count: number;
    order_state_count: number;
  };
  risk_events: Array<{ event_time: string | null; reason: string | null }>;
  heartbeats: Array<{ heartbeat_time: string | null; status: string | null }>;
  order_states: Array<{ event_time: string | null; symbol: string | null; state: string | null; message: string | null }>;
  status: string;
}

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
