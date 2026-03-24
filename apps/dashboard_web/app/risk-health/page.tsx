import { RiskHealthRuntime } from '@/components/risk-health-runtime';

export default function RiskHealthPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Risk & Health parity</p>
        <h2>Operational telemetry</h2>
        <p>
          This route mirrors the Streamlit risk and heartbeat tab with Python-backed telemetry from the latest run.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed risk feed</p>
            <h3>Latest telemetry</h3>
          </div>
        </div>
        <RiskHealthRuntime />
      </section>
    </div>
  );
}
