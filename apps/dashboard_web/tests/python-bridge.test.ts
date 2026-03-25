import { describe, expect, it } from 'vitest';

import {
  buildOverviewCards,
  dashboardBridgeContract,
  dashboardCutoverGate,
  navigationItems,
} from '../lib/python-bridge';

describe('dashboard bridge contract', () => {
  it('keeps the 8GB baseline explicit', () => {
    expect(dashboardBridgeContract.memoryBudget.hostRamGb).toBe(8);
    expect(dashboardBridgeContract.memoryBudget.targetPeakRssGb).toBeLessThan(8);
  });

  it('maps both legacy dashboard entry views', () => {
    expect(dashboardBridgeContract.legacyViews.map((view) => view.id)).toEqual([
      'main-dashboard',
      'exact-window-suite',
    ]);
  });

  it('marks the migrated routes as available in navigation order', () => {
    const availableRoutes = navigationItems.filter((item) => item.status === 'available');

    expect(availableRoutes).toHaveLength(7);
    expect(availableRoutes.map((item) => item.href)).toEqual([
      '/',
      '/performance-price',
      '/execution-analytics',
      '/workflows',
      '/risk-health',
      '/exact-window',
      '/report-export',
    ]);
  });

  it('builds overview cards from capability metadata', () => {
    const cards = buildOverviewCards();

    expect(cards).toHaveLength(dashboardBridgeContract.capabilities.length);
    expect(cards.some((card) => card.title === 'Python compatibility contract')).toBe(true);
  });

  it('keeps cutover-gate evidence explicit while preserving Streamlit as the launcher', () => {
    expect(dashboardCutoverGate.defaultLauncher).toBe('streamlit');
    expect(dashboardCutoverGate.readyRoutes).toEqual([
      '/performance-price',
      '/execution-analytics',
      '/report-export',
    ]);
    expect(dashboardCutoverGate.evidence.at(-1)?.detail).toContain('default launcher');
  });
});
