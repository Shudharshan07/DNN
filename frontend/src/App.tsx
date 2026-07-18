import { useRef, useState } from 'react';
import { House } from 'lucide-react';
import { DNNVisualizer, type DNNVisualizerHandle } from './components/DNNVisualizer';
import { DNNVisualizer2D } from './components/DNNVisualizer2D';
import { StatsPanel } from './components/StatsPanel';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

function App() {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws';
  const { snapshot, isConnected, reconnect } = useWebSocket(wsUrl);
  const visualizerRef = useRef<DNNVisualizerHandle>(null);
  const [is3D, setIs3D] = useState(true);

  return (
    <div className="app-shell">
      <main className="visualizer-pane" aria-label="DNN network visualizer">
        {is3D && (
          <button
            className="visualizer-home-button"
            type="button"
            onClick={() => visualizerRef.current?.resetView()}
            aria-label="Reset visualizer view"
            title="Reset view"
          >
            <House size={15} strokeWidth={2.2} />
          </button>
        )}
        {is3D
          ? <DNNVisualizer ref={visualizerRef} snapshot={snapshot} />
          : <DNNVisualizer2D snapshot={snapshot} />
        }
      </main>
      <aside className="side-panel" aria-label="DNN training stats">
        <StatsPanel
          snapshot={snapshot}
          isConnected={isConnected}
          onReconnect={reconnect}
          is3D={is3D}
          onToggle3D={setIs3D}
        />
      </aside>
    </div>
  );
}

export default App;
