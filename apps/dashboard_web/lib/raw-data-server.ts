import type { RawDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadRawDataFromPython(): Promise<RawDataPayload> {
  return runUvPythonSnippetJson<RawDataPayload>(`
from lumina_quant.dashboard.cutover_surfaces_service import load_raw_data_payload
import json
print(json.dumps(load_raw_data_payload(point_limit=60), sort_keys=True))
`);
}
