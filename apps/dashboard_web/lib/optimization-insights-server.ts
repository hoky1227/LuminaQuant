import type { OptimizationInsightsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadOptimizationInsightsFromPython(): Promise<OptimizationInsightsPayload> {
  return runUvPythonSnippetJson<OptimizationInsightsPayload>(`
from lumina_quant.dashboard.cutover_surfaces_service import load_optimization_insights_payload
import json
print(json.dumps(load_optimization_insights_payload(point_limit=200), sort_keys=True))
`);
}
