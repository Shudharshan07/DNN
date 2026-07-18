interface SettingsPanelProps {
  onBack: () => void;
}

export const SettingsPanel = ({ onBack }: SettingsPanelProps) => {
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

      <section className="settings-placeholder" aria-label="Settings page">
        <div>
          <p>Settings</p>
          <span>Controls will live here.</span>
        </div>
      </section>
    </div>
  );
};
