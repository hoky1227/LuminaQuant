import { NextResponse } from 'next/server';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../../../../../');

export const dynamic = 'force-dynamic';

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { job_id?: string; action?: string };
    const stdout = execFileSync(
      'uv',
      ['run', 'python', '-c', `
from lumina_quant.dashboard.workflow_jobs_service import control_workflow_job
import json
payload = control_workflow_job(dsn=None, job_id=${JSON.stringify(String(payload.job_id || ''))}, action=${JSON.stringify(String(payload.action || ''))})
print(json.dumps(payload, sort_keys=True))
`],
      {
        cwd: REPO_ROOT,
        encoding: 'utf-8',
      },
    );
    const result = JSON.parse(stdout.trim()) as Record<string, unknown>;
    return NextResponse.json(result, { status: result.ok ? 200 : 400 });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { ok: false, error: 'workflow_job_control_failed', detail },
      { status: 500 },
    );
  }
}
