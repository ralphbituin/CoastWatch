"use client";

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "coastwatch.displayTimezone";
const CHANGE_EVENT = "coastwatch:display-timezone";
export const DEFAULT_TIMEZONE = "GMT";

export function isValidTimeZone(value: string): boolean {
  if (!value.trim()) return false;
  try {
    new Intl.DateTimeFormat("en-GB", { timeZone: value }).format(new Date());
    return true;
  } catch {
    return false;
  }
}

export function normalizeTimeZone(value: string | null | undefined): string {
  const candidate = (value ?? "").trim();
  return candidate && isValidTimeZone(candidate) ? candidate : DEFAULT_TIMEZONE;
}

export function readDisplayTimezone(): string {
  if (typeof window === "undefined") return DEFAULT_TIMEZONE;
  return normalizeTimeZone(window.localStorage.getItem(STORAGE_KEY));
}

export function writeDisplayTimezone(value: string): string {
  const next = normalizeTimeZone(value);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, next);
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: next }));
  }
  return next;
}

export function timezoneLabel(value: string): string {
  return /^(GMT|UTC|Etc\/GMT|Etc\/UTC)$/i.test(value) ? "GMT" : value;
}

/**
 * Shared display-timezone preference.
 *
 * Timestamps are always stored in UTC by the backend. This only controls how
 * live-session and event times are rendered in the browser. The default is GMT
 * (UTC), matching the live-capture provenance.
 */
export function useDisplayTimezone() {
  const [timezone, setTimezoneState] = useState<string>(DEFAULT_TIMEZONE);

  useEffect(() => {
    setTimezoneState(readDisplayTimezone());
    const sync = () => setTimezoneState(readDisplayTimezone());
    const onCustom = (event: Event) => {
      const detail = (event as CustomEvent<string>).detail;
      setTimezoneState(normalizeTimeZone(detail));
    };
    window.addEventListener("storage", sync);
    window.addEventListener(CHANGE_EVENT, onCustom);
    return () => {
      window.removeEventListener("storage", sync);
      window.removeEventListener(CHANGE_EVENT, onCustom);
    };
  }, []);

  const setTimezone = useCallback((value: string) => {
    setTimezoneState(writeDisplayTimezone(value));
  }, []);

  const formatDateTime = useCallback(
    (value: string | number | Date | null | undefined): string => {
      if (value === null || value === undefined || value === "") return "—";
      const date = value instanceof Date ? value : new Date(value);
      if (Number.isNaN(date.getTime())) return "—";
      return date.toLocaleString("en-GB", {
        timeZone: timezone,
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
    },
    [timezone],
  );

  const formatTime = useCallback(
    (value: string | number | Date | null | undefined): string => {
      if (value === null || value === undefined || value === "") return "—";
      const date = value instanceof Date ? value : new Date(value);
      if (Number.isNaN(date.getTime())) return "—";
      return date.toLocaleTimeString("en-GB", {
        timeZone: timezone,
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      });
    },
    [timezone],
  );

  return {
    timezone,
    label: timezoneLabel(timezone),
    setTimezone,
    formatDateTime,
    formatTime,
  };
}
