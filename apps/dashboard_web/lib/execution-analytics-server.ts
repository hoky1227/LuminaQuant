import type { ExecutionAnalyticsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadExecutionAnalyticsFromPython(): Promise<ExecutionAnalyticsPayload> {
  return runUvPythonSnippetJson<ExecutionAnalyticsPayload>(`
from lumina_quant.dashboard.cutover_surfaces_service import load_execution_analytics_payload
import json
print(json.dumps(load_execution_analytics_payload(fill_limit=200, order_limit=200), sort_keys=True))
`);
}
