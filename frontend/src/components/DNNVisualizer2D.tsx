import { useCallback, useEffect, useRef } from 'react';
import type { Snapshot } from '../types/snapshot';

interface DNNVisualizer2DProps {
  snapshot: Snapshot | null;
}

function getNodeCounts(snapshot: Snapshot): number[] {
  if (!snapshot.layers.length) return [];
  const counts: number[] = [snapshot.layers[0].header.rows];
  for (const layer of snapshot.layers) counts.push(layer.header.cols);
  return counts;
}

function weightColor(w: number): string {
  const norm = Math.tanh(Math.abs(w) * 0.5);
  const a = (norm * 0.46 + 0.08).toFixed(2);
  if (w >= 0) {
    const g = Math.round((norm * 0.55 + 0.32) * 255);
    const b = Math.round((norm * 0.4 + 0.3) * 255);
    return `rgba(31,${g},${b},${a})`;
  }
  const r = Math.round((norm * 0.65 + 0.35) * 255);
  const b = Math.round((norm * 0.28 + 0.24) * 255);
  return `rgba(${r},61,${b},${a})`;
}

export const DNNVisualizer2D = ({ snapshot }: DNNVisualizer2DProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const snapshotRef = useRef(snapshot);
  snapshotRef.current = snapshot;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    if (W === 0 || H === 0) return;

    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#f7f7f5';
    ctx.fillRect(0, 0, W, H);

    const snap = snapshotRef.current;
    if (!snap) return;

    const nodeCounts = getNodeCounts(snap);
    if (!nodeCounts.length) return;

    const numCols = nodeCounts.length;
    const maxNodes = Math.max(...nodeCounts);

    const padX = 48;
    const padY = 44;
    const colSpacing = numCols > 1 ? (W - padX * 2) / (numCols - 1) : 0;
    const rawSpacing = maxNodes > 1 ? (H - padY * 2) / (maxNodes - 1) : 0;
    const nodeSpacing = Math.min(rawSpacing, 54);
    const nodeR = Math.min(13, nodeSpacing * 0.3, colSpacing * 0.22);

    // Pre-compute node positions
    const positions: { x: number; y: number }[][] = nodeCounts.map((count, colIdx) => {
      const x = numCols === 1 ? W / 2 : padX + colIdx * colSpacing;
      const totalH = (count - 1) * nodeSpacing;
      const startY = H / 2 - totalH / 2;
      return Array.from({ length: count }, (_, i) => ({
        x,
        y: count === 1 ? H / 2 : startY + i * nodeSpacing,
      }));
    });

    // Connections
    snap.layers.forEach((layer, layerIdx) => {
      const { rows, cols } = layer.header;
      const fromPos = positions[layerIdx];
      const toPos = positions[layerIdx + 1];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const from = fromPos[r];
          const to = toPos?.[c];
          if (!from || !to) continue;
          const w = layer.weights[r * cols + c] ?? 0;
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = weightColor(w);
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    });

    // Nodes
    nodeCounts.forEach((_count, colIdx) => {
      const isInput = colIdx === 0;
      const isOutput = colIdx === numCols - 1;
      const fill = isInput ? '#168a3a' : isOutput ? '#c2410c' : '#0759c9';
      const stroke = isInput ? '#082d16' : isOutput ? '#3a1205' : '#061f4f';

      positions[colIdx].forEach((pos) => {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    });

    // Column size labels
    ctx.fillStyle = '#86868b';
    ctx.font = '550 11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    nodeCounts.forEach((count, colIdx) => {
      const x = numCols === 1 ? W / 2 : padX + colIdx * colSpacing;
      const topY = positions[colIdx][0]?.y ?? padY;
      ctx.fillText(String(count), x, topY - nodeR - 7);
    });
  }, []);

  // Redraw when snapshot changes
  useEffect(() => {
    draw();
  }, [snapshot, draw]);

  // Redraw on container resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(draw);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      className="visualizer-canvas visualizer-canvas-2d"
      aria-label="2D DNN network visualizer"
    />
  );
};
