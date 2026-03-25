import type { ReportExportPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadReportExportFromPython(): Promise<ReportExportPayload> {
  return runUvPythonSnippetJson<ReportExportPayload>(`
from lumina_quant.dashboard.cutover_surfaces_service import load_report_export_payload
import json
print(json.dumps(load_report_export_payload(point_limit=240, fill_limit=200, event_limit=50), sort_keys=True))
`);
}
