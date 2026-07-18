interface SettingsPanelProps {
  onBack: () => void;
  is3D: boolean;
  onToggle3D: (value: boolean) => void;
}

export const SettingsPanel = ({ onBack, is3D, onToggle3D }: SettingsPanelProps) => {
  return (
    <div className="settings-panel">
      <header className="settings-header">
        <button
          className="panel-icon-button back-button"
          type="button"
          onClick={onBack}
          aria-label="Back to training stats"
          title="Back"
        >
          Back
        </button>
        <span>Settings</span>
      </header>

      <section className="settings-section" aria-label="Visualizer settings">
        <div className="settings-row">
          <div className="settings-row-label">
            <span>Visualizer mode</span>
            <small>{is3D ? '3D interactive view' : '2D flat view'}</small>
          </div>
          <button
            className={`toggle-button ${is3D ? 'is-on' : ''}`}
            type="button"
            role="switch"
            aria-checked={is3D}
            onClick={() => onToggle3D(!is3D)}
          >
            <span className="toggle-thumb" />
          </button>
        </div>
      </section>
    </div>
  );
};
