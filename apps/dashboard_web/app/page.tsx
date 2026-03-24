import { buildOverviewCards, dashboardBridgeContract } from '@/lib/python-bridge';
import { loadOverviewPayloadFromPython } from '@/lib/python-bridge-server';

const overviewCards = buildOverviewCards();

function buildEmptyStateMessage(status: string): string {
  switch (status) {
    case 'missing_dsn':
      return 'Set LQ_POSTGRES_DSN so the Next overview can read the same runtime state store as the Streamlit dashboard.';
    case 'no_runs':
      return 'No runs were found yet. Start a backtest/live workflow from the existing Streamlit path, then refresh this page.';
    case 'no_equity':
      return 'A run exists but no equity rows are available yet. Let the run emit telemetry or choose a different run once selection parity lands.';
    default:
      return 'The Python overview bridge returned no data.';
  }
}

function buildSparklinePath(
  values: number[],
  width = 420,
  height = 120,
): string {
  if (values.length === 0) {
    return '';
  }
  if (values.length === 1) {
    return `M 0 ${height / 2} L ${width} ${height / 2}`;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

export default async function Home() {
  const overview = await loadOverviewPayloadFromPython();
  const recentEquity = overview.equity_curve.slice(-5);
  const recentRuns = overview.recent_runs.slice(0, 5);
  const equitySparkline = buildSparklinePath(
    overview.equity_curve.map((point) => point.equity),
  );
  const drawdownSparkline = buildSparklinePath(
    overview.drawdown_curve.map((point) => point.drawdown),
  );
  const emptyStateMessage = buildEmptyStateMessage(overview.source.status);
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Migration foundation</p>
        <h2>Overview parity slice</h2>
        <p>
          This web view now mirrors the Streamlit dashboard&apos;s landing intent with a real Python-backed overview payload while keeping
          the compatibility bridge explicit.
        </p>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">8GB guardrail</p>
            <h3>Memory budget</h3>
          </div>
          <div className="metric-badge">Target peak RSS: {dashboardBridgeContract.memoryBudget.targetPeakRssGb} GB</div>
        </div>
        <div className="metric-grid">
          {overview.summary_metrics.slice(0, 4).map((metric) => (
            <article key={metric.key}>
              <span>{metric.label}</span>
              <strong>{String(metric.value)}</strong>
            </article>
          ))}
        </div>
        <ul className="guidance-list">
          {dashboardBridgeContract.memoryBudget.guidance.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python compatibility bridge</p>
            <h3>Legacy-to-web route contract</h3>
          </div>
          <div className="metric-badge">{dashboardBridgeContract.compatibilityPath}</div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Surface</th>
                <th>Streamlit source</th>
                <th>Next route</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {dashboardBridgeContract.capabilities.map((capability) => (
                <tr key={capability.id}>
                  <td>{capability.title}</td>
                  <td>{capability.streamlitSource}</td>
                  <td>{capability.nextRoute}</td>
                  <td>
                    <span className={`status-pill status-${capability.status}`}>{capability.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Real overview payload</p>
            <h3>Equity and drawdown</h3>
          </div>
          <div className="metric-badge">{overview.source.status}</div>
        </div>
        {equitySparkline ? (
          <div className="card-grid">
            <article className="feature-card">
              <div className="feature-header">
                <h4>Equity curve</h4>
                <span className="status-pill status-available">live</span>
              </div>
              <svg viewBox="0 0 420 120" role="img" aria-label="Equity curve preview">
                <path d={equitySparkline} fill="none" stroke="currentColor" strokeWidth="3" />
              </svg>
            </article>
            <article className="feature-card">
              <div className="feature-header">
                <h4>Drawdown curve</h4>
                <span className="status-pill status-available">live</span>
              </div>
              <svg viewBox="0 0 420 120" role="img" aria-label="Drawdown curve preview">
                <path d={drawdownSparkline} fill="none" stroke="currentColor" strokeWidth="3" />
              </svg>
            </article>
          </div>
        ) : null}
        {recentEquity.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Equity</th>
                </tr>
              </thead>
              <tbody>
                {recentEquity.map((point) => (
                  <tr key={point.timestamp}>
                    <td>{point.timestamp}</td>
                    <td>{point.equity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>{emptyStateMessage}</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Run selection parity</p>
            <h3>Recent runs</h3>
          </div>
          <div className="metric-badge">{recentRuns.length} rows</div>
        </div>
        {recentRuns.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Strategy</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((run) => (
                  <tr key={run.run_id}>
                    <td>{run.run_id}</td>
                    <td>{run.mode}</td>
                    <td>{run.status}</td>
                    <td>{run.strategy || 'unknown'}</td>
                    <td>{run.started_at ?? 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>{emptyStateMessage}</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Foundation scope</p>
            <h3>Available now</h3>
          </div>
        </div>
        <div className="card-grid">
          {overviewCards.map((card) => (
            <article key={card.title} className="feature-card">
              <div className="feature-header">
                <h4>{card.title}</h4>
                <span className={`status-pill status-${card.status}`}>{card.status}</span>
              </div>
              <p>{card.description}</p>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
