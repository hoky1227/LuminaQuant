import { describe, expect, it } from 'vitest';

import {
  buildOverviewCards,
  dashboardBridgeContract,
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

    expect(availableRoutes).toHaveLength(4);
    expect(availableRoutes.map((item) => item.href)).toEqual(['/', '/workflows', '/risk-health', '/exact-window']);
  });

  it('builds overview cards from capability metadata', () => {
    const cards = buildOverviewCards();

    expect(cards).toHaveLength(dashboardBridgeContract.capabilities.length);
    expect(cards.some((card) => card.title === 'Python compatibility contract')).toBe(true);
  });
});
