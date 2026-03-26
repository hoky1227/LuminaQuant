import { ExecutionAnalyticsRuntime } from '@/components/execution-analytics-runtime';

export default function ExecutionAnalyticsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Execution Analytics parity</p>
        <h2>Fill quality and closed-trade outcomes</h2>
        <p>
          This route exposes execution analytics with lean Python-backed summaries for the latest run.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed execution feed</p>
            <h3>Latest fills and order-state context</h3>
          </div>
        </div>
        <ExecutionAnalyticsRuntime />
      </section>
    </div>
  );
}
