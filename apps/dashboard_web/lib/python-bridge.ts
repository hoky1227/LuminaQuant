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

export const dashboardCutoverGate = {
  defaultLauncher: 'streamlit',
  launcherStatus: 'guarded',
  readyRoutes: ['/performance-price', '/execution-analytics', '/report-export'],
  evidence: [
    {
      label: 'Performance & Price parity slice',
      detail: 'Python-backed route is available at /performance-price for the equity, drawdown, and benchmark view.',
      status: 'available',
    },
    {
      label: 'Execution Analytics parity slice',
      detail: 'Python-backed route is available at /execution-analytics for fills, streaks, and order status breakdowns.',
      status: 'available',
    },
    {
      label: 'Report Export parity slice',
      detail: 'Python-backed route is available at /report-export with JSON and Markdown snapshot exports.',
      status: 'available',
    },
    {
      label: 'Default launcher preserved',
      detail: 'uv run lq dashboard --run still targets the Streamlit default launcher until explicit cutover review closes the gate.',
      status: 'guarded',
    },
  ],
  remainingGate:
    'Keep Streamlit as the default launcher until end-to-end parity verification approves the Next cutover.',
} as const;

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
      id: 'performance-price',
      title: 'Performance & Price',
      description: 'Equity, drawdown, benchmark price, funding, and trade-marker parity from Python state.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/performance-price',
      status: 'available',
    },
    {
      id: 'execution-analytics',
      title: 'Execution Analytics',
      description: 'Fill quality, closed-trade outcomes, streaks, and order-status distribution from Python telemetry.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/execution-analytics',
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
    {
      id: 'report-export',
      title: 'Report Export',
      description: 'JSON + Markdown snapshot export preview while the Streamlit launcher remains the default path.',
      streamlitSource: 'apps/dashboard/app.py',
      nextRoute: '/report-export',
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
    id: 'performance-price',
    href: '/performance-price',
    label: 'Performance & Price',
    summary: 'Equity, drawdown, benchmark, funding, and trade-marker parity from the latest run.',
    status: 'available',
  },
  {
    id: 'execution-analytics',
    href: '/execution-analytics',
    label: 'Execution Analytics',
    summary: 'Closed-trade outcomes, streaks, and order status telemetry for the active run.',
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
  {
    id: 'report-export',
    href: '/report-export',
    label: 'Report Export',
    summary: 'Snapshot JSON/Markdown export preview plus explicit cutover gate evidence.',
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
