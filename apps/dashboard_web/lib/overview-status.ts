export function buildOverviewEmptyStateMessage(status: string): string {
  switch (status) {
    case 'missing_dsn':
      return 'Set LQ_POSTGRES_DSN so the Next overview can read the same runtime state store as the dashboard services.';
    case 'no_runs':
      return 'No runs were found yet. Start a backtest/live workflow, then refresh this page.';
    case 'no_equity':
      return 'A run exists but no equity rows are available yet. Let the run emit telemetry or choose a different run once selection parity lands.';
    default:
      return 'The Python overview bridge returned no data.';
  }
}
