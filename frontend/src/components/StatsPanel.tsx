import type { Snapshot } from '../types/snapshot';

interface StatsPanelProps {
  snapshot: Snapshot | null;
  isConnected: boolean;
}

const formatNumber = (value: number | undefined, digits = 6) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a';
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.0001)) {
    return value.toExponential(3);
  }
  return value.toFixed(digits).replace(/\.?0+$/, '');
};

const summarize = (values: number[]) => {
  if (!values.length) return null;

  let min = values[0];
  let max = values[0];
  let sum = 0;

  for (const value of values) {
    if (value < min) min = value;
    if (value > max) max = value;
    sum += value;
  }

  return {
    min,
    max,
    mean: sum / values.length,
    preview: values.slice(0, 6),
  };
};

export const StatsPanel = ({ snapshot, isConnected }: StatsPanelProps) => {
  return (
    <div className="stats-panel">
      <header className="panel-header">
        <span className={`status-pill ${isConnected ? 'is-connected' : 'is-disconnected'}`}>
          <span aria-hidden="true" />
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </header>

      <section className="metrics-grid" aria-label="Training metrics">
        <div className="metric">
          <span>Epoch</span>
          <strong>{snapshot?.epoch ?? 'n/a'}</strong>
        </div>
        <div className="metric">
          <span>Step</span>
          <strong>{snapshot?.step ?? 'n/a'}</strong>
        </div>
        <div className="metric wide">
          <span>Loss</span>
          <strong>{formatNumber(snapshot?.loss)}</strong>
        </div>
      </section>

      {!snapshot && (
        <section className="empty-state">
          {isConnected ? 'Waiting for the first snapshot...' : 'Start the websocket stream to see values.'}
        </section>
      )}

      {snapshot && (
        <section className="layer-list" aria-label="Layer details">
          <div className="section-title">
            <h2>Layer Details ({snapshot?.layers.length ?? 0})</h2>
            <span>version {snapshot.version}</span>
          </div>

          {snapshot.layers.map((layer, idx) => {
            const weightStats = summarize(layer.weights);
            const biasStats = summarize(layer.biases);

            return (
              <article className="layer-card" key={`${idx}-${layer.header.rows}-${layer.header.cols}`}>
                <div className="layer-card-header">
                  <div>
                    <h3>Layer {idx + 1}</h3>
                    <p>{layer.header.cols} to {layer.header.rows}</p>
                  </div>
                  <span>{layer.header.bias_size} bias</span>
                </div>

                <div className="stat-table">
                  <div>
                    <span>Weights</span>
                    <strong>{layer.weights.length}</strong>
                  </div>
                  <div>
                    <span>Biases</span>
                    <strong>{layer.biases.length}</strong>
                  </div>
                  <div>
                    <span>Weight min/max</span>
                    <strong>
                      {weightStats ? `${formatNumber(weightStats.min, 4)} / ${formatNumber(weightStats.max, 4)}` : 'n/a'}
                    </strong>
                  </div>
                  <div>
                    <span>Weight mean</span>
                    <strong>{weightStats ? formatNumber(weightStats.mean, 5) : 'n/a'}</strong>
                  </div>
                  <div>
                    <span>Bias min/max</span>
                    <strong>
                      {biasStats ? `${formatNumber(biasStats.min, 4)} / ${formatNumber(biasStats.max, 4)}` : 'n/a'}
                    </strong>
                  </div>
                  <div>
                    <span>Bias mean</span>
                    <strong>{biasStats ? formatNumber(biasStats.mean, 5) : 'n/a'}</strong>
                  </div>
                </div>

                <div className="value-strip" aria-label={`Layer ${idx + 1} weight values`}>
                  {(weightStats?.preview.length ? weightStats.preview : [undefined]).map((value, valueIdx) => (
                    <code key={valueIdx}>{formatNumber(value, 3)}</code>
                  ))}
                </div>
              </article>
            );
          })}
        </section>
      )}
    </div>
  );
};
