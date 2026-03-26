import { OptimizationInsightsRuntime } from '@/components/optimization-insights-runtime';

export default function OptimizationInsightsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Optimization Insights parity</p>
        <h2>Candidate quality and best-parameter previews</h2>
        <p>
          This route keeps optimization evidence visible in the Next dashboard while preserving the guarded default
          launcher flow.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed optimization feed</p>
            <h3>Latest stage medians and candidate rows</h3>
          </div>
        </div>
        <OptimizationInsightsRuntime />
      </section>
    </div>
  );
}
