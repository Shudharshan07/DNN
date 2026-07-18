import { DNNVisualizer } from './components/DNNVisualizer';
import { StatsPanel } from './components/StatsPanel';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

function App() {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8080/ws';
  const { snapshot, isConnected } = useWebSocket(wsUrl);

  return (
    <div className="app-shell">
      <main className="visualizer-pane" aria-label="DNN network visualizer">
        <DNNVisualizer snapshot={snapshot} />
      </main>
      <aside className="side-panel" aria-label="DNN training stats">
        <StatsPanel snapshot={snapshot} isConnected={isConnected} />
      </aside>
    </div>
  );
}

export default App;
