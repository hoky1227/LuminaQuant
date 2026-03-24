import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../');

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

export async function loadWorkflowJobsFromPython(): Promise<WorkflowJobsPayload> {
  const stdout = execFileSync(
    'uv',
    ['run', 'python', '-c', `
from lumina_quant.dashboard.workflow_jobs_service import load_recent_workflow_jobs_payload
import json
print(json.dumps(load_recent_workflow_jobs_payload(limit=10), sort_keys=True))
`],
    {
      cwd: REPO_ROOT,
      encoding: 'utf-8',
    },
  );
  return JSON.parse(stdout.trim()) as WorkflowJobsPayload;
}
