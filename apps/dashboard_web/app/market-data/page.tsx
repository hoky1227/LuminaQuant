import { MarketDataRuntime } from '@/components/market-data-runtime';

export default function MarketDataPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Market Data parity</p>
        <h2>Market bars and context under the cutover guardrail</h2>
        <p>
          This route exposes Python-backed OHLCV context while keeping the default dashboard launcher on Next.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed market feed</p>
            <h3>Latest bars and guarded indicator context</h3>
          </div>
        </div>
        <MarketDataRuntime />
      </section>
    </div>
  );
}
