import { RawDataRuntime } from '@/components/raw-data-runtime';

export default function RawDataPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Raw Data parity</p>
        <h2>Capped frame previews for debugging and cutover evidence</h2>
        <p>
          This route exposes bounded raw-state previews for the latest run so Next cutover work can be reviewed without
          changing the Streamlit-first launcher contract.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed raw-state feed</p>
            <h3>Frame counts and preview rows</h3>
          </div>
        </div>
        <RawDataRuntime />
      </section>
    </div>
  );
}
