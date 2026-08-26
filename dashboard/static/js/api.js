// Thin wrappers around the backend API.
const API = {
  async registry() {
    const r = await fetch('/api/registry');
    if (!r.ok) throw new Error('registry failed');
    return r.json();
  },
  async validate(yamlContent) {
    const r = await fetch('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml_content: yamlContent }),
    });
    return r.json();
  },
  // -> {configs, protected, demo, missions, labels, reference, downloaded: [name]}
  async listConfigs() {
    const r = await fetch('/api/configs');
    return r.json();
  },
  async loadConfig(name) {
    const r = await fetch('/api/configs/' + encodeURIComponent(name));
    if (!r.ok) {
      // A demo config's file is downloaded on demand here, so a failure is
      // often a real, specific reason (network error, bad URL) worth showing
      // rather than a generic "load failed".
      throw new Error((await r.json().catch(() => ({}))).detail || 'load failed');
    }
    return r.json();
  },
  async saveConfig(name, yamlContent) {
    const r = await fetch('/api/configs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, yaml_content: yamlContent }),
    });
    if (!r.ok) {
      throw new Error((await r.json().catch(() => ({}))).detail || 'save failed');
    }
    return r.json();
  },
  async deleteConfig(name) {
    const r = await fetch('/api/configs/' + encodeURIComponent(name), { method: 'DELETE' });
    if (!r.ok) {
      throw new Error((await r.json().catch(() => ({}))).detail || 'delete failed');
    }
  },
  // Open the configs folder in the OS file browser (server-side, so this only
  // does anything when the dashboard is viewed on the machine running it).
  async revealConfigs() {
    const r = await fetch('/api/configs/reveal', { method: 'POST' });
    if (!r.ok) {
      throw new Error((await r.json().catch(() => ({}))).detail || 'could not open folder');
    }
    return r.json();
  },
  async run(yamlContent) {
    const r = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml_content: yamlContent }),
    });
    if (!r.ok) {
      const detail = (await r.json().catch(() => ({}))).detail || 'run failed';
      throw new Error(detail);
    }
    return r.json();
  },
  async stopRun() {
    await fetch('/api/run/stop', { method: 'POST' });
  },
  async continueRun() {
    await fetch('/api/run/continue', { method: 'POST' });
  },
  async rerunStep(parameters) {
    await fetch('/api/run/rerun', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parameters }),
    });
  },
  async runStatus() {
    return (await fetch('/api/run/status')).json();
  },
};
