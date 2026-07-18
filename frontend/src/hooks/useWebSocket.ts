import { useCallback, useEffect, useRef, useState } from 'react';
import type { Snapshot } from '../types/snapshot';

export const useWebSocket = (url: string) => {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;

    // Close any existing socket before opening a new one
    if (socketRef.current) {
      socketRef.current.close();
    }

    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => {
      if (cancelled) return;
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      if (cancelled) return;
      try {
        const data: Snapshot = JSON.parse(event.data);
        setSnapshot(data);
      } catch (error) {
        console.error('Failed to parse snapshot:', error);
      }
    };

    socket.onerror = () => {
      if (cancelled) return;
      setIsConnected(false);
    };

    socket.onclose = () => {
      if (cancelled) return;
      setIsConnected(false);
    };

    return () => {
      cancelled = true;
      socket.close();
    };
  }, [url, retryCount]);

  const reconnect = useCallback(() => {
    setRetryCount((c) => c + 1);
  }, []);

  return { snapshot, isConnected, reconnect };
};
