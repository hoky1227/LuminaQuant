import type { RiskHealthPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadRiskHealthFromPython(): Promise<RiskHealthPayload> {
  return runUvPythonSnippetJson<RiskHealthPayload>(`
from lumina_quant.dashboard.risk_health_service import load_risk_health_payload
import json
print(json.dumps(load_risk_health_payload(limit=25), sort_keys=True))
`);
}
