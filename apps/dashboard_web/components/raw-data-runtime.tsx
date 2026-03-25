'use client';

import { useEffect, useState } from 'react';

import { readJsonOrThrow } from '@/lib/bridge-fetch';
import type { RawDataPayload } from '@/lib/dashboard-contracts';

export function RawDataRuntime() {
  const [payload, setPayload] = useState<RawDataPayload | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    let active = true;
    fetch('/api/python/dashboard/raw-data', { cache: 'no-store' })
      .then(async (response) => {
        const body = await readJsonOrThrow<RawDataPayload>(response, 'raw data bridge failed');
        if (active) {
          setPayload(body);
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
  if (payload === null) {
    return <p>Loading raw data parity payload…</p>;
  }
  if (payload.status !== 'ok') {
    return <p>No raw data payload available yet.</p>;
  }

  return (
    <div className="page-stack">
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Raw state context</p>
            <h3>Source snapshot</h3>
          </div>
          <div className="metric-badge">{payload.context.source ?? 'n/a'}</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Run ID</span>
            <strong>{payload.context.run_id ?? 'n/a'}</strong>
          </article>
          <article>
            <span>Market</span>
            <strong>{payload.context.market ?? 'n/a'}</strong>
          </article>
          <article>
            <span>Frames</span>
            <strong>{payload.frame_summaries.length}</strong>
          </article>
        </div>
      </section>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Frame</th>
              <th>Rows</th>
              <th>Columns</th>
            </tr>
          </thead>
          <tbody>
            {payload.frame_summaries.map((frame) => (
              <tr key={frame.label}>
                <td>{frame.label}</td>
                <td>{frame.rows}</td>
                <td>{frame.columns}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card-grid">
        {payload.previews.map((preview) => (
          <article key={preview.label} className="feature-card">
            <div className="feature-header">
              <h4>{preview.label}</h4>
              <span className="status-pill status-available">{preview.rows.length} rows</span>
            </div>
            <pre className="code-block">{JSON.stringify(preview.rows, null, 2)}</pre>
          </article>
        ))}
      </div>
    </div>
  );
}
