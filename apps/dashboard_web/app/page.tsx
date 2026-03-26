import { buildOverviewCards, dashboardBridgeContract, dashboardCutoverGate } from '@/lib/python-bridge';
import { OverviewRuntime } from '@/components/overview-runtime';

const overviewCards = buildOverviewCards();
export default function Home() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Next-default dashboard</p>
        <h2>Overview parity slice</h2>
        <p>
          This web view now serves as the default LuminaQuant dashboard runtime while keeping the Python-backed compatibility bridge
          explicit for the migrated surfaces.
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
            <span>Default entry point</span>
            <strong>{dashboardBridgeContract.defaultEntryPoint}</strong>
          </article>
          <article>
            <span>Compatibility path</span>
            <strong>{dashboardBridgeContract.compatibilityPath}</strong>
          </article>
          <article>
            <span>Available routes</span>
            <strong>
              {
                dashboardBridgeContract.capabilities.filter(
                  (capability) => capability.status === 'available',
                ).length
              }
            </strong>
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
                <th>Python source</th>
                <th>Next route</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {dashboardBridgeContract.capabilities.map((capability) => (
                <tr key={capability.id}>
                  <td>{capability.title}</td>
                  <td>{capability.sourceModule}</td>
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
            <p className="eyebrow">Cutover gate evidence</p>
            <h3>Default launcher is live</h3>
          </div>
          <div className="metric-badge">{dashboardCutoverGate.defaultLauncher}</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Ready Next routes</span>
            <strong>{dashboardCutoverGate.readyRoutes.length}</strong>
          </article>
          <article>
            <span>Launcher status</span>
            <strong>{dashboardCutoverGate.launcherStatus}</strong>
          </article>
        </div>
        <ul className="guidance-list">
          {dashboardCutoverGate.evidence.map((item) => (
            <li key={item.label}>
              <strong>{item.label}:</strong> {item.detail}
            </li>
          ))}
        </ul>
        <p>{dashboardCutoverGate.remainingGate}</p>
      </section>

      <OverviewRuntime />

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
