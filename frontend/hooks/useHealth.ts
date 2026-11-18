"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { DatabaseHealthResponse, HealthResponse } from "@/lib/types";

export interface HealthState {
  service: HealthResponse | null;
  db: DatabaseHealthResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/** Polls the backend liveness and database-readiness endpoints. */
export function useHealth(pollMs = 15000): HealthState {
  const [service, setService] = useState<HealthResponse | null>(null);
  const [db, setDb] = useState<DatabaseHealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [serviceResult, dbResult] = await Promise.all([
        api.health(),
        api.healthDb().catch((err: unknown) => {
          if (err instanceof Error) return null;
          return null;
        }),
      ]);
      setService(serviceResult);
      setDb(dbResult);
      setError(null);
    } catch (err) {
      setService(null);
      setDb(null);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), pollMs);
    return () => clearInterval(timer);
  }, [refresh, pollMs]);

  return { service, db, loading, error, refresh };
}
