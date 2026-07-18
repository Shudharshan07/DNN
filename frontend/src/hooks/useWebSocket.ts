import { useEffect, useState } from 'react';
import type { Snapshot } from '../types/snapshot';

export const useWebSocket = (url: string) => {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const socket = new WebSocket(url);

    socket.onopen = () => {
      if (cancelled) return;
      console.log(`Connected to ${url}`);
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      if (cancelled) return;
      try {
        const data: Snapshot = JSON.parse(event.data);
        console.log('Snapshot received:', data);
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
      console.log('WebSocket connection closed');
      setIsConnected(false);
    };

    return () => {
      cancelled = true;
      socket.close();
    };
  }, [url]);

  return { snapshot, isConnected };
};
