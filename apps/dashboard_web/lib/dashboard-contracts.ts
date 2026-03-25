export interface BridgeErrorBody {
  detail?: string;
  error?: string;
  ok?: boolean;
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
