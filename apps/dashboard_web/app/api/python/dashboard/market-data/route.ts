import { NextResponse } from 'next/server';

import { loadMarketDataFromPython } from '@/lib/market-data-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadMarketDataFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_market_data_failed', detail },
      { status: 500 },
    );
  }
}
