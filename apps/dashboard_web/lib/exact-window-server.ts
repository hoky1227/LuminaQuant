import type { ExactWindowPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadExactWindowFromPython(): Promise<ExactWindowPayload> {
  return runUvPythonSnippetJson<ExactWindowPayload>(`
from lumina_quant.dashboard.exact_window_service import load_exact_window_summary_payload
import json
print(json.dumps(load_exact_window_summary_payload(), sort_keys=True))
`);
}
