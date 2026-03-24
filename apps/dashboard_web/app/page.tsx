import { buildOverviewCards, dashboardBridgeContract } from '@/lib/python-bridge';

const overviewCards = buildOverviewCards();

export default function Home() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Migration foundation</p>
        <h2>Overview placeholder</h2>
        <p>
          This first web view mirrors the Streamlit dashboard&apos;s landing intent while keeping data access stubbed behind a typed
          compatibility bridge.
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
          <article>
            <span>Host baseline</span>
            <strong>{dashboardBridgeContract.memoryBudget.hostRamGb} GB</strong>
          </article>
          <article>
            <span>Legacy entry point</span>
            <strong>{dashboardBridgeContract.legacyEntryPoint}</strong>
          </article>
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
