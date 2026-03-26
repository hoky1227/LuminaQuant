import Link from 'next/link';
import type { ReactNode } from 'react';

import { navigationItems } from '@/lib/python-bridge';

export function DashboardShell({ children }: { children: ReactNode }) {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand-block">
          <p className="eyebrow">LuminaQuant</p>
          <h1>Dashboard Web</h1>
          <p className="lede">
            Next.js dashboard foundation, kept intentionally lean for the 8GB baseline.
          </p>
        </div>
        <nav aria-label="Dashboard sections">
          <ul className="nav-list">
            {navigationItems.map((item) => (
              <li key={item.id} className="nav-item">
                <Link href={item.href} aria-current={item.status === 'available' ? 'page' : undefined}>
                  <span>{item.label}</span>
                  <span className={`status-pill status-${item.status}`}>{item.status}</span>
                </Link>
                <p>{item.summary}</p>
              </li>
            ))}
          </ul>
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
