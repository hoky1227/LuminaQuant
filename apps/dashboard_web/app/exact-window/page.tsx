import { ExactWindowRuntime } from '@/components/exact-window-runtime';

export default function ExactWindowPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Exact-window parity</p>
        <h2>Research bundle snapshot</h2>
        <p>
          This route keeps the migration SSR-first by reading the existing exact-window artifact bundle instead of re-running the heavy
          research workflow inside Next.js.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed bundle summary</p>
            <h3>Latest exact-window artifacts</h3>
          </div>
        </div>
        <ExactWindowRuntime />
      </section>
    </div>
  );
}
