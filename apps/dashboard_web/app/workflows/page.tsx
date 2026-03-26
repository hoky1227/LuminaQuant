import { WorkflowJobsRuntime } from '@/components/workflow-jobs-runtime';

export default function WorkflowJobsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Workflow parity</p>
        <h2>Managed jobs</h2>
        <p>
          This route exposes managed backtest, optimize, and live jobs from the Next dashboard.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed workflow feed</p>
            <h3>Recent managed jobs</h3>
          </div>
        </div>
        <WorkflowJobsRuntime />
      </section>
    </div>
  );
}
