import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../');

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

export async function loadRiskHealthFromPython(): Promise<RiskHealthPayload> {
  const stdout = execFileSync(
    'uv',
    ['run', 'python', '-c', `
from lumina_quant.dashboard.risk_health_service import load_risk_health_payload
import json
print(json.dumps(load_risk_health_payload(limit=25), sort_keys=True))
`],
    {
      cwd: REPO_ROOT,
      encoding: 'utf-8',
    },
  );
  return JSON.parse(stdout.trim()) as RiskHealthPayload;
}
