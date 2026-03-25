import type { BridgeErrorBody } from '@/lib/dashboard-contracts';

export async function readJsonOrThrow<T>(response: Response, fallbackMessage: string): Promise<T> {
  const body = (await response.json()) as T;
  if (!response.ok) {
    const errorBody = body as BridgeErrorBody;
    throw new Error(errorBody.detail ?? fallbackMessage);
  }
  return body;
}
