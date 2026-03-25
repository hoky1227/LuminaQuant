import type { WorkflowJobsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonSnippetJson } from '@/lib/python-runtime';

export async function loadWorkflowJobsFromPython(): Promise<WorkflowJobsPayload> {
  return runUvPythonSnippetJson<WorkflowJobsPayload>(`
from lumina_quant.dashboard.workflow_jobs_service import load_recent_workflow_jobs_payload
import json
print(json.dumps(load_recent_workflow_jobs_payload(limit=10), sort_keys=True))
`);
}
