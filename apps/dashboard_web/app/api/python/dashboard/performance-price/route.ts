import { NextResponse } from 'next/server';

import { loadPerformancePriceFromPython } from '@/lib/performance-price-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadPerformancePriceFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_performance_price_failed', detail },
      { status: 500 },
    );
  }
}
