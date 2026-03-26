import { ReportExportRuntime } from '@/components/report-export-runtime';

export default function ReportExportPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Report Export parity</p>
        <h2>Snapshot JSON and Markdown export</h2>
        <p>
          This route keeps report export explicit in Next.js for the default dashboard launcher.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed snapshot export</p>
            <h3>Preview and download</h3>
          </div>
        </div>
        <ReportExportRuntime />
      </section>
    </div>
  );
}
