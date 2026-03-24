import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../');

export interface OverviewMetric {
  key: string;
  label: string;
  value: number | string | null;
}

export interface OverviewPoint {
  timestamp: string;
  equity?: number;
  drawdown?: number;
}

export interface OverviewPayload {
  as_of: string;
  summary_metrics: OverviewMetric[];
  equity_curve: Array<OverviewPoint & { equity: number }>;
  drawdown_curve: Array<OverviewPoint & { drawdown: number }>;
  source: {
    mode: string;
    backend: string;
    status: string;
    run_id?: string;
  };
}

export function loadOverviewPayloadFromPython(): OverviewPayload {
  const stdout = execFileSync(
    'uv',
    ['run', 'python', '-m', 'lumina_quant.dashboard.bridge', '--overview-json', '--mode', 'next'],
    {
      cwd: REPO_ROOT,
      encoding: 'utf-8',
    },
  );
  return JSON.parse(stdout.trim()) as OverviewPayload;
}
