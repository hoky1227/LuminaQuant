'use client';

import { useEffect, useState } from 'react';

import { readJsonOrThrow } from '@/lib/bridge-fetch';
import type { BridgeErrorBody, WorkflowJobsPayload } from '@/lib/dashboard-contracts';

export function WorkflowJobsRuntime() {
  const [payload, setPayload] = useState<WorkflowJobsPayload | null>(null);
  const [error, setError] = useState<string>('');

  async function refresh() {
    const response = await fetch('/api/python/dashboard/workflow-jobs', { cache: 'no-store' });
    const body = await readJsonOrThrow<WorkflowJobsPayload>(response, 'workflow jobs bridge failed');
    setPayload(body);
  }

  async function triggerAction(jobId: string, action: 'stop' | 'kill') {
    const response = await fetch('/api/python/dashboard/workflow-jobs/control', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, action }),
    });
    const body = (await response.json()) as BridgeErrorBody;
    if (!response.ok || body.ok === false) {
      throw new Error(body.detail ?? body.error ?? 'workflow job action failed');
    }
    await refresh();
  }

  useEffect(() => {
    let active = true;
    refresh()
      .then(() => undefined)
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
    return <p>Loading workflow jobs…</p>;
  }
  if (payload.jobs.length === 0) {
    return <p>No managed workflow jobs recorded yet.</p>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Workflow</th>
            <th>Status</th>
            <th>Mode</th>
            <th>Strategy</th>
            <th>Run ID</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {payload.jobs.map((job) => (
            <tr key={job.job_id}>
              <td>{job.workflow}</td>
              <td>{job.status}</td>
              <td>{job.requested_mode || 'n/a'}</td>
              <td>{job.strategy || 'n/a'}</td>
              <td>{job.run_id || 'n/a'}</td>
              <td>
                <button type="button" onClick={() => void triggerAction(job.job_id, 'stop')}>
                  Stop
                </button>
                <button type="button" onClick={() => void triggerAction(job.job_id, 'kill')}>
                  Kill
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
