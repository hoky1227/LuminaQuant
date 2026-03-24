import { cache } from 'react';
import { execFile } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const REPO_ROOT = resolve(__dirname, '../../../');
const execFileAsync = promisify(execFile);

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
  performance_metrics: {
    cagr?: number;
    annualized_volatility?: number;
    sharpe_ratio?: number;
    sortino_ratio?: number;
    calmar_ratio?: number;
    max_drawdown?: number;
  };
  recent_runs: Array<{
    run_id: string;
    mode: string;
    status: string;
    strategy: string;
    started_at: string | null;
  }>;
  equity_curve: Array<OverviewPoint & { equity: number }>;
  drawdown_curve: Array<OverviewPoint & { drawdown: number }>;
  source: {
    mode: string;
    backend: string;
    status: string;
    run_id?: string;
  };
}

export const loadOverviewPayloadFromPython = cache(async (): Promise<OverviewPayload> => {
  const { stdout } = await execFileAsync(
    'uv',
    ['run', 'python', '-m', 'lumina_quant.dashboard.bridge', '--overview-json', '--mode', 'next'],
    {
      cwd: REPO_ROOT,
      encoding: 'utf-8',
    },
  );
  return JSON.parse(stdout.trim()) as OverviewPayload;
});
