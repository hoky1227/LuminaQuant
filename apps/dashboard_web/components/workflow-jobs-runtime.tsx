'use client';

import { useEffect, useState } from 'react';

interface WorkflowJobRecord {
  job_id: string;
  workflow: string;
  status: string;
  requested_mode: string;
  strategy: string;
  run_id: string;
  started_at: string | null;
  ended_at: string | null;
}

interface WorkflowJobsPayload {
  jobs: WorkflowJobRecord[];
  status: string;
}

export function WorkflowJobsRuntime() {
  const [payload, setPayload] = useState<WorkflowJobsPayload | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    let active = true;
    fetch('/api/python/dashboard/workflow-jobs', { cache: 'no-store' })
      .then(async (response) => {
        const body = (await response.json()) as WorkflowJobsPayload | { detail?: string };
        if (!response.ok) {
          throw new Error('detail' in body ? body.detail ?? 'workflow jobs bridge failed' : 'workflow jobs bridge failed');
        }
        if (active) {
          setPayload(body as WorkflowJobsPayload);
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
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
