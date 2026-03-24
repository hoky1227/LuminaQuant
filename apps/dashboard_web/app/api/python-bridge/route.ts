import { NextResponse } from 'next/server';

import { dashboardBridgeContract } from '@/lib/python-bridge';

export const dynamic = 'force-static';

export function GET() {
  return NextResponse.json(dashboardBridgeContract);
}
