import { PerformancePriceRuntime } from '@/components/performance-price-runtime';

export default function PerformancePricePage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Performance &amp; Price parity</p>
        <h2>Equity, drawdown, and benchmark telemetry</h2>
        <p>
          This route covers the highest-priority performance slice while staying Python-backed and 8GB-safe.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed performance feed</p>
            <h3>Latest run context</h3>
          </div>
        </div>
        <PerformancePriceRuntime />
      </section>
    </div>
  );
}
