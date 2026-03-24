import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../../../../');

export function GET() {
  try {
    const stdout = execFileSync(
      'uv',
      ['run', 'python', '-m', 'lumina_quant.dashboard.bridge', '--json', '--mode', 'next'],
      {
        cwd: REPO_ROOT,
        encoding: 'utf-8',
      },
    );
    return NextResponse.json(JSON.parse(stdout));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: 'dashboard_python_bridge_failed',
        detail,
      },
      { status: 500 },
    );
  }
}
