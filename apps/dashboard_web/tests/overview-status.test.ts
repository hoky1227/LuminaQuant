import { describe, expect, it } from 'vitest';

import { buildOverviewEmptyStateMessage } from '../lib/overview-status';

describe('buildOverviewEmptyStateMessage', () => {
  it('keeps missing DSN guidance explicit', () => {
    expect(buildOverviewEmptyStateMessage('missing_dsn')).toBe(
      'Set LQ_POSTGRES_DSN so the Next overview can read the same runtime state store as the dashboard services.',
    );
  });

  it('keeps no-equity guidance explicit', () => {
    expect(buildOverviewEmptyStateMessage('no_equity')).toBe(
      'A run exists but no equity rows are available yet. Let the run emit telemetry or choose a different run once selection parity lands.',
    );
  });

  it('falls back to the generic message', () => {
    expect(buildOverviewEmptyStateMessage('unknown')).toBe(
      'The Python overview bridge returned no data.',
    );
  });
});
