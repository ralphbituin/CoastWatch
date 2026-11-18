"use client";

import { useEffect, useRef, useState } from "react";

import { WS_URL } from "@/lib/config";

export type RealtimeStatus = "connecting" | "open" | "closed" | "error";

export interface RealtimeState {
  status: RealtimeStatus;
  lastMessage: unknown;
  send: (data: string) => void;
}

/**
 * Connects to the backend WebSocket channel with basic reconnect backoff.
 * Phase 1 uses it to verify realtime connectivity; later phases push
 * shoreline observations, tide updates and anomaly events through it.
 */
export function useRealtimeStatus(): RealtimeState {
  const [status, setStatus] = useState<RealtimeStatus>("connecting");
  const [lastMessage, setLastMessage] = useState<unknown>(null);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let closedByUnmount = false;
    let retry = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;

    const connect = () => {
      try {
        const socket = new WebSocket(WS_URL);
        socketRef.current = socket;

        socket.onopen = () => {
          retry = 0;
          setStatus("open");
          socket.send("ping");
        };
        socket.onmessage = (event) => {
          try {
            setLastMessage(JSON.parse(event.data));
          } catch {
            setLastMessage(event.data);
          }
        };
        socket.onerror = () => setStatus("error");
        socket.onclose = () => {
          setStatus("closed");
          if (!closedByUnmount) {
            retry += 1;
            const delay = Math.min(1000 * 2 ** retry, 15000);
            reconnectTimer = setTimeout(connect, delay);
          }
        };
      } catch {
        setStatus("error");
      }
    };

    connect();

    return () => {
      closedByUnmount = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socketRef.current?.close();
    };
  }, []);

  const send = (data: string) => socketRef.current?.send(data);

  return { status, lastMessage, send };
}
