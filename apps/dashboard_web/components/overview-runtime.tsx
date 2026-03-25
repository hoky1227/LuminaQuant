'use client';

import { useEffect, useState } from 'react';

import { readJsonOrThrow } from '@/lib/bridge-fetch';

interface OverviewMetric {
  key: string;
  label: string;
  value: number | string | null;
}

interface OverviewPoint {
  timestamp: string;
  equity?: number;
  drawdown?: number;
}

interface OverviewPayload {
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
  workflow_jobs: Array<{
    job_id: string;
    workflow: string;
    status: string;
    requested_mode: string;
    strategy: string;
    run_id: string;
    started_at: string | null;
    ended_at: string | null;
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

function buildEmptyStateMessage(status: string): string {
  switch (status) {
    case 'missing_dsn':
      return 'Set LQ_POSTGRES_DSN so the Next overview can read the same runtime state store as the Streamlit dashboard.';
    case 'no_runs':
      return 'No runs were found yet. Start a backtest/live workflow from the existing Streamlit path, then refresh this page.';
    case 'no_equity':
      return 'A run exists but no equity rows are available yet. Let the run emit telemetry or choose a different run once selection parity lands.';
    default:
      return 'The Python overview bridge returned no data.';
  }
}

function buildSparklinePath(
  values: number[],
  width = 420,
  height = 120,
): string {
  if (values.length === 0) {
    return '';
  }
  if (values.length === 1) {
    return `M 0 ${height / 2} L ${width} ${height / 2}`;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

export function OverviewRuntime() {
  const [overview, setOverview] = useState<OverviewPayload | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    let active = true;
    fetch('/api/python/dashboard/overview', { cache: 'no-store' })
      .then(async (response) => {
        const payload = await readJsonOrThrow<OverviewPayload>(response, 'overview bridge failed');
        if (active) {
          setOverview(payload);
        }
      })
      .catch((fetchError: unknown) => {
        if (active) {
          setError(fetchError instanceof Error ? fetchError.message : String(fetchError));
        }
      });
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return <p>{error}</p>;
  }
  if (overview === null) {
    return <p>Loading Python-backed overview payload…</p>;
  }

  const recentEquity = overview.equity_curve.slice(-5);
  const recentRuns = overview.recent_runs.slice(0, 5);
  const recentJobs = overview.workflow_jobs.slice(0, 5);
  const performanceEntries: Array<[string, number]> = [
    ['CAGR', overview.performance_metrics.cagr],
    ['Ann. Volatility', overview.performance_metrics.annualized_volatility],
    ['Sharpe', overview.performance_metrics.sharpe_ratio],
    ['Sortino', overview.performance_metrics.sortino_ratio],
    ['Calmar', overview.performance_metrics.calmar_ratio],
    ['Max Drawdown', overview.performance_metrics.max_drawdown],
  ].filter((entry): entry is [string, number] => typeof entry[1] === 'number');
  const equitySparkline = buildSparklinePath(
    overview.equity_curve.map((point) => point.equity),
  );
  const drawdownSparkline = buildSparklinePath(
    overview.drawdown_curve.map((point) => point.drawdown),
  );
  const emptyStateMessage = buildEmptyStateMessage(overview.source.status);

  return (
    <>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Real overview payload</p>
            <h3>Equity and drawdown</h3>
          </div>
          <div className="metric-badge">{overview.source.status}</div>
        </div>
        {equitySparkline ? (
          <div className="card-grid">
            <article className="feature-card">
              <div className="feature-header">
                <h4>Equity curve</h4>
                <span className="status-pill status-available">live</span>
              </div>
              <svg viewBox="0 0 420 120" role="img" aria-label="Equity curve preview">
                <path d={equitySparkline} fill="none" stroke="currentColor" strokeWidth="3" />
              </svg>
            </article>
            <article className="feature-card">
              <div className="feature-header">
                <h4>Drawdown curve</h4>
                <span className="status-pill status-available">live</span>
              </div>
              <svg viewBox="0 0 420 120" role="img" aria-label="Drawdown curve preview">
                <path d={drawdownSparkline} fill="none" stroke="currentColor" strokeWidth="3" />
              </svg>
            </article>
          </div>
        ) : null}
        {recentEquity.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Equity</th>
                </tr>
              </thead>
              <tbody>
                {recentEquity.map((point) => (
                  <tr key={point.timestamp}>
                    <td>{point.timestamp}</td>
                    <td>{point.equity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>{emptyStateMessage}</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Workflow parity</p>
            <h3>Recent jobs</h3>
          </div>
          <div className="metric-badge">{recentJobs.length} jobs</div>
        </div>
        {recentJobs.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Workflow</th>
                  <th>Status</th>
                  <th>Mode</th>
                  <th>Strategy</th>
                  <th>Run ID</th>
                </tr>
              </thead>
              <tbody>
                {recentJobs.map((job) => (
                  <tr key={job.job_id}>
                    <td>{job.workflow}</td>
                    <td>{job.status}</td>
                    <td>{job.requested_mode || 'n/a'}</td>
                    <td>{job.strategy || 'n/a'}</td>
                    <td>{job.run_id || 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No managed workflow jobs have been recorded yet.</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Performance parity</p>
            <h3>Derived metrics</h3>
          </div>
          <div className="metric-badge">{performanceEntries.length} metrics</div>
        </div>
        {performanceEntries.length > 0 ? (
          <div className="metric-grid">
            {performanceEntries.map(([label, value]) => (
              <article key={label}>
                <span>{label}</span>
                <strong>{String(value)}</strong>
              </article>
            ))}
          </div>
        ) : (
          <p>{emptyStateMessage}</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Run selection parity</p>
            <h3>Recent runs</h3>
          </div>
          <div className="metric-badge">{recentRuns.length} rows</div>
        </div>
        {recentRuns.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Strategy</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((run) => (
                  <tr key={run.run_id}>
                    <td>{run.run_id}</td>
                    <td>{run.mode}</td>
                    <td>{run.status}</td>
                    <td>{run.strategy || 'unknown'}</td>
                    <td>{run.started_at ?? 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>{emptyStateMessage}</p>
        )}
      </section>
    </>
  );
}
