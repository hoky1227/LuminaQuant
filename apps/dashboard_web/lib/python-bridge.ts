export type NavigationStatus = 'available' | 'planned';

export interface LegacyView {
  id: string;
  label: string;
  source: string;
}

export interface NavigationItem {
  id: string;
  href: string;
  label: string;
  summary: string;
  status: NavigationStatus;
}

export interface CapabilityItem {
  id: string;
  title: string;
  description: string;
  streamlitSource: string;
  nextRoute: string;
  status: NavigationStatus;
}

export interface OverviewCard {
  title: string;
  description: string;
  status: NavigationStatus;
}

export const dashboardBridgeContract = {
  legacyEntryPoint: 'uv run lq dashboard --run',
  compatibilityPath: '/api/python/dashboard/overview',
  memoryBudget: {
    hostRamGb: 8,
    targetPeakRssGb: 6.5,
    guidance: [
      'Keep heavy verification sequential.',
      'Prefer one active web build/test lane at a time.',
      'Retain Streamlit as the default compatibility path during migration.',
    ],
  },
  legacyViews: [
    {
      id: 'main-dashboard',
      label: 'Main Dashboard',
      source: 'apps/dashboard/app.py',
    },
    {
      id: 'exact-window-suite',
      label: 'Exact-Window Suite',
      source: 'apps/dashboard/exact_window_suite.py',
    },
  ] satisfies LegacyView[],
  capabilities: [
    {
      id: 'overview',
      title: 'Overview placeholder',
      description:
        'First migrated route that mirrors the Streamlit landing surface while staying data-contract first.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/',
      status: 'available',
    },
    {
      id: 'python-compatibility',
      title: 'Python compatibility contract',
      description:
        'Python-backed compatibility metadata for the first slice, exposed to Next.js via the bridge route.',
      streamlitSource: 'src/lumina_quant/dashboard/bridge.py',
      nextRoute: '/api/python/dashboard/overview',
      status: 'available',
    },
    {
      id: 'exact-window',
      title: 'Exact-window migration',
      description: 'Latest exact-window artifact summary and portfolio fallback parity from the Python bundle.',
      streamlitSource: 'apps/dashboard/exact_window_suite.py',
      nextRoute: '/exact-window',
      status: 'available',
    },
    {
      id: 'workflow-jobs',
      title: 'Workflow jobs',
      description: 'Managed backtest/optimize/live job status and control parity for the web dashboard.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/workflows',
      status: 'available',
    },
    {
      id: 'risk-health',
      title: 'Risk & Health',
      description: 'Recent risk events, heartbeats, and order-state changes for the active run.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/risk-health',
      status: 'available',
    },
  ] satisfies CapabilityItem[],
} as const;

export const navigationItems: NavigationItem[] = [
  {
    id: 'overview',
    href: '/',
    label: 'Overview',
    summary: 'First parity slice backed by the Python compatibility contract.',
    status: 'available',
  },
  {
    id: 'workflows',
    href: '/workflows',
    label: 'Workflow Jobs',
    summary: 'Managed run status, strategy, and mode parity from the Python workflow store.',
    status: 'available',
  },
  {
    id: 'risk-health',
    href: '/risk-health',
    label: 'Risk & Health',
    summary: 'Recent risk, heartbeat, and order-state telemetry from the latest run.',
    status: 'available',
  },
  {
    id: 'exact-window',
    href: '/exact-window',
    label: 'Exact-window',
    summary: 'Latest exact-window artifact summary from the Python research bundle.',
    status: 'available',
  },
];

export function buildOverviewCards(): OverviewCard[] {
  return dashboardBridgeContract.capabilities.map((capability) => ({
    title: capability.title,
    description: capability.description,
    status: capability.status,
  }));
}
