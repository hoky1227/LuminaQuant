import type { MarketDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadMarketDataFromPython(): Promise<MarketDataPayload> {
  return runUvPythonSnippetJson<MarketDataPayload>(`
from lumina_quant.dashboard.cutover_surfaces_service import load_market_data_payload
import json
print(json.dumps(load_market_data_payload(point_limit=240, fill_limit=80), sort_keys=True))
`);
}
