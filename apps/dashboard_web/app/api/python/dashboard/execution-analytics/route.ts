import { NextResponse } from 'next/server';

import { loadExecutionAnalyticsFromPython } from '@/lib/execution-analytics-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadExecutionAnalyticsFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_execution_analytics_failed', detail },
      { status: 500 },
    );
  }
}
