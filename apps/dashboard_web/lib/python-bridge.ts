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
      description: 'Deferred until later waves after the overview slice reaches parity.',
      streamlitSource: 'apps/dashboard/exact_window_suite.py',
      nextRoute: '/exact-window',
      status: 'planned',
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
    id: 'exact-window',
    href: '/exact-window',
    label: 'Exact-window',
    summary: 'Planned follow-on route after the migration foundation settles.',
    status: 'planned',
  },
];

export function buildOverviewCards(): OverviewCard[] {
  return dashboardBridgeContract.capabilities.map((capability) => ({
    title: capability.title,
    description: capability.description,
    status: capability.status,
  }));
}
