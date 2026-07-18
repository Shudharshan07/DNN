import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Snapshot } from '../types/snapshot';

interface DNNVisualizerProps {
  snapshot: Snapshot | null;
}

export interface DNNVisualizerHandle {
  resetView: () => void;
}

// Build the unique list of neuron counts per column (input, hidden..., output)
function getNodeCounts(snapshot: Snapshot): number[] {
  if (!snapshot.layers.length) return [];
  const counts: number[] = [snapshot.layers[0].header.cols];
  for (const layer of snapshot.layers) {
    counts.push(layer.header.rows);
  }
  return counts;
}

const getCameraDistance = (nodeCounts: number[]) => {
  if (!nodeCounts.length) return 20;
  const networkWidth = (nodeCounts.length - 1) * 4.5;
  return networkWidth * 1.2 + 10;
};

export const DNNVisualizer = forwardRef<DNNVisualizerHandle, DNNVisualizerProps>(({ snapshot }, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const networkGroupRef = useRef<THREE.Group | null>(null);
  const rafRef = useRef<number>(0);
  const userInteractedRef = useRef(false);
  const fittedTopologyRef = useRef<string | null>(null);
  const nodeCountsRef = useRef<number[]>([]);

  const resetView = () => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    const distance = getCameraDistance(nodeCountsRef.current);
    camera.position.set(0, 0, distance);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
    userInteractedRef.current = false;
  };

  useImperativeHandle(ref, () => ({ resetView }), []);

  // Setup scene once
  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f7f5);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 20);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0xf7f7f5, 1);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 80;
    controls.addEventListener('start', () => {
      userInteractedRef.current = true;
    });
    controlsRef.current = controls;

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);
    handleResize();

    return () => {
      cancelAnimationFrame(rafRef.current);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // Rebuild network geometry when snapshot changes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Remove old network
    if (networkGroupRef.current) {
      scene.remove(networkGroupRef.current);
      const geometries = new Set<THREE.BufferGeometry>();
      const materials = new Set<THREE.Material>();

      networkGroupRef.current.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          geometries.add(obj.geometry);
          const material = obj.material;
          if (Array.isArray(material)) {
            material.forEach((entry) => materials.add(entry));
          } else {
            materials.add(material);
          }
        }
      });

      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
    }

    if (!snapshot) return;

    const nodeCounts = getNodeCounts(snapshot);
    nodeCountsRef.current = nodeCounts;
    if (!nodeCounts.length) return;

    const controls = controlsRef.current;
    if (controls) {
      const homeDistance = getCameraDistance(nodeCounts);
      controls.minDistance = Math.max(2.5, homeDistance * 0.14);
      controls.maxDistance = Math.max(40, homeDistance * 3.2);
    }

    const group = new THREE.Group();
    networkGroupRef.current = group;
    scene.add(group);

    const maxNodes = Math.max(...nodeCounts);
    const layerSpacingX = 4.5;
    // Cap vertical spread so big layers do not explode out of view.
    const maxHeight = 14;
    const nodeRadius = Math.min(0.18, maxHeight / maxNodes / 2);
    const getNodeY = (i: number, count: number) => {
      if (count === 1) return 0;
      return ((i / (count - 1)) - 0.5) * Math.min(count * nodeRadius * 3, maxHeight);
    };

    // Pre-compute node world positions per column
    const positions: THREE.Vector3[][] = nodeCounts.map((count, colIdx) => {
      const x = (colIdx - (nodeCounts.length - 1) / 2) * layerSpacingX;
      return Array.from({ length: count }, (_, i) => new THREE.Vector3(x, getNodeY(i, count), 0));
    });

    // Draw weight connections for each layer
    snapshot.layers.forEach((layer, layerIdx) => {
      const { rows, cols } = layer.header;
      const fromPositions = positions[layerIdx];
      const toPositions = positions[layerIdx + 1];

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const from = fromPositions[c];
          const to = toPositions[r];
          if (!from || !to) continue; // guard against shape mismatch

          const w = layer.weights[r * cols + c] ?? 0;
          const norm = Math.tanh(Math.abs(w) * 0.5);

          const color = w >= 0
            ? new THREE.Color(0.12, norm * 0.55 + 0.32, norm * 0.4 + 0.3)
            : new THREE.Color(norm * 0.65 + 0.35, 0.24, norm * 0.28 + 0.24);

          const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
          const lineMat = new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: norm * 0.46 + 0.08,
          });
          group.add(new THREE.Line(lineGeo, lineMat));
        }
      }
    });

    // Draw nodes on top of connections
    const nodeGeo = new THREE.SphereGeometry(nodeRadius, 16, 16);
    nodeCounts.forEach((count, colIdx) => {
      const isInput = colIdx === 0;
      const isOutput = colIdx === nodeCounts.length - 1;

      const nodeColor = isInput ? 0x168a3a : isOutput ? 0xc2410c : 0x0759c9;
      const emissive = isInput ? 0x082d16 : isOutput ? 0x3a1205 : 0x061f4f;

      for (let i = 0; i < count; i++) {
        const mat = new THREE.MeshPhongMaterial({
          color: nodeColor,
          emissive,
          shininess: 72,
          specular: 0xdbeafe,
        });
        const mesh = new THREE.Mesh(nodeGeo, mat);
        mesh.position.copy(positions[colIdx][i]);
        group.add(mesh);
      }
    });

    // Fit once per topology until the user takes control. Incoming values should not
    // undo zoom, pan, or rotation while training data streams in.
    const camera = cameraRef.current;
    const topologyKey = nodeCounts.join('x');
    if (camera && !userInteractedRef.current && fittedTopologyRef.current !== topologyKey) {
      resetView();
      fittedTopologyRef.current = topologyKey;
    }
  }, [snapshot]);

  return <div ref={containerRef} className="visualizer-canvas" />;
});
