// Turn the builder STATE into pipeline YAML and back. The builder is the source
// of truth; the YAML pane is a live preview plus an escape hatch for hand edits
// (synced back on demand via "YAML → builder").

const Config = {
  // Build the plain config object (pre-YAML) from STATE.
  toObject() {
    const cfg = {};

    // pipeline block: only include fields the user actually set.
    const pipeline = {};
    for (const spec of STATE.registry.pipeline_fields) {
      const val = STATE.pipeline.settings[spec.name];
      if (val !== null && val !== undefined && val !== '') pipeline[spec.name] = val;
    }
    if (Object.keys(pipeline).length) cfg.pipeline = pipeline;

    // steps
    cfg.steps = STATE.pipeline.items.map((item) => {
      const params = {};
      for (const spec of item.def.parameters || []) {
        const val = item.values[spec.name];
        if (spec.name === 'qc_settings' && isQcContainer(item.def)) {
          if (val && Object.keys(val).length) params[spec.name] = val;
          continue;
        }
        const required = spec.required;
        const changed = !('default' in spec) || !Forms.equal(val, spec.default);
        // Skip empty optional values (leave them to the step's own default).
        const empty = val === null || val === undefined || val === '';
        if (required || (changed && !empty)) params[spec.name] = val;
      }
      // Parameters the registry doesn't describe (framework ones such as
      // qc_handling_settings) aren't editable in the builder, but must survive
      // the round trip — dropping them silently changes what the step does.
      for (const [k, v] of Object.entries(item.extras || {})) params[k] = v;
      const step = { name: item.name };
      if (Object.keys(params).length) step.parameters = params;
      step.diagnostics = item.diagnostics;
      return step;
    });

    return cfg;
  },

  // Serialise via Forms.dump so scalar lists stay inline ([20, 45]) and integer
  // mapping keys (Argo QC flags in `variable_ranges`, `flag_mapping`, …) emit
  // unquoted, matching the hand-written config style. Steps are emitted node by
  // node so each section can be preceded by its banner comment.
  toYAML() {
    const cfg = Config.toObject();
    const steps = cfg.steps || [];
    let out = '';
    if (cfg.pipeline) out += Forms.dump({ pipeline: cfg.pipeline }) + '\n';
    if (!steps.length) return out + 'steps: []\n';

    // Indent one serialised step ("- name: X\n  parameters:…") under `steps:`.
    const emit = (obj) => Forms.dump([obj]).replace(/^(?=.)/gm, '  ');
    out += 'steps:\n';
    let i = 0;
    for (const node of STATE.pipeline.nodes) {
      if (isSection(node)) {
        out += Config.banner(node.title);
        for (let k = 0; k < node.steps.length; k++) out += emit(steps[i++]);
      } else {
        out += emit(steps[i++]);
      }
    }
    return out;
  },

  // A section header, in the banner-comment style the hand-written configs use.
  banner(title) {
    const bar = '# ' + '='.repeat(59);
    const text = String(title == null ? '' : title).trim();
    const pad = Math.max(0, Math.floor((59 - text.length) / 2));
    return `\n${bar}\n# ${' '.repeat(pad)}${text}\n${bar}\n`;
  },

  // Find section banners in raw YAML text. Comments are lost by the object
  // round-trip, so sections are recovered from the text instead. Each result is
  // {title, index} where index counts the steps that precede the banner.
  // Recognises the three-line `# ====` banner this emits, and the one-line
  // `# ---- TITLE ----` style also used in the example configs.
  sectionsFromYAML(text) {
    const lines = String(text).split('\n');
    const isBar = (l) => /^\s*#\s*[=~-]{5,}\s*$/.test(l || '');
    const oneLine = /^\s*#\s*[-=]{4,}\s*(\S.*?)\s*[-=]{4,}\s*$/;
    const found = [];
    let inSteps = false, stepsIndent = null, count = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!inSteps) { if (/^steps:\s*$/.test(line)) inSteps = true; continue; }
      const m = line.match(/^(\s*)-\s/);
      if (m) {
        if (stepsIndent === null) stepsIndent = m[1].length;
        if (m[1].length === stepsIndent) count++;
        continue;
      }
      const one = line.match(oneLine);
      if (one && /[A-Za-z0-9]/.test(one[1])) { found.push({ title: one[1], index: count }); continue; }
      if (isBar(line) && !isBar(lines[i + 1]) && isBar(lines[i + 2])) {
        const t = (lines[i + 1] || '').match(/^\s*#\s*(\S.*?)\s*$/);
        if (t) { found.push({ title: t[1], index: count }); i += 2; }
      }
    }
    return found;
  },

  // Load YAML text into the builder, sections included.
  fromYAML(text) {
    Config.fromObject(jsyaml.load(text) || {}, Config.sectionsFromYAML(text));
  },

  // Best-effort load of a config object back into the builder. `sections` is
  // the banner list from sectionsFromYAML (absent when loading a bare object).
  fromObject(cfg, sections = []) {
    STATE.pipeline = { settings: {}, nodes: [], items: [] };

    const pipeline = (cfg && cfg.pipeline) || {};
    for (const spec of STATE.registry.pipeline_fields) {
      STATE.pipeline.settings[spec.name] =
        spec.name in pipeline ? pipeline[spec.name] : Forms.defaultValue(spec);
    }

    const steps = (cfg && cfg.steps) || [];
    const lowered = {};
    for (const n of Object.keys(STATE.stepsByName)) lowered[n.toLowerCase()] = n;

    // Banner index -> section, opened as the step at that index is reached. A
    // section stays open until the next banner, so steps before the first
    // banner sit loose at the top level.
    const nodes = STATE.pipeline.nodes;
    let si = 0;
    const openSectionsAt = (k) => {
      while (si < sections.length && sections[si].index <= k) {
        nodes.push(makeSection(sections[si].title));
        si++;
      }
    };

    steps.forEach((s, idx) => {
      openSectionsAt(idx);
      if (!s || !s.name) return;
      const canonical = STATE.stepsByName[s.name] ? s.name : lowered[String(s.name).toLowerCase()];
      const def = STATE.stepsByName[canonical];
      if (!def) return; // unknown step: skip (surfaced by Validate)
      const item = {
        id: ++STATE._seq,
        name: canonical,
        def,
        values: initValues(def),
        diagnostics: !!s.diagnostics,
        collapsed: true,
      };
      const params = s.parameters || {};
      const described = new Set((def.parameters || []).map((p) => p.name));
      item.extras = {};
      for (const [k, v] of Object.entries(params)) {
        if (described.has(k)) item.values[k] = v;
        else item.extras[k] = v; // kept verbatim, re-emitted by toObject
      }
      if (isQcContainer(def) && !item.values.qc_settings) item.values.qc_settings = {};
      const last = nodes[nodes.length - 1];
      if (isSection(last)) last.steps.push(item);
      else nodes.push(item);
    });
    openSectionsAt(steps.length); // trailing banners with no steps under them

    renderSettings();
    renderPipeline();
  },

  // ---- persistence ----
  //
  // The shipped reference config (default.yaml) is read-only. Editing it
  // doesn't overwrite it: the pending save target switches to a fresh
  // custom_run_N, so the reference stays pristine and your work becomes a
  // config of its own. Editing any other config is ordinary — changes belong
  // to that file. The server enforces the same rule, so this is convenience,
  // not the lock itself.
  known: [],        // every config name on the server
  locked: [],       // the protected subset
  demo: [],         // the demo subset (also locked) — shown as their own group
  missions: {},     // demo config names grouped by deployment mission, in display order
  labels: {},       // display label per demo config name (glider names repeat across missions)
  reference: [],    // non-demo protected configs (default.yaml, demo_alr.yaml)
  downloaded: [],   // demo configs whose NetCDF file is already on disk
  current: null,    // name of the loaded config, or null for an unsaved one
  selected: '',     // name shown in the picker ('' once the config is unsaved)
  loading: false,   // true while loading/booting, so that isn't seen as an edit
  busy: false,      // true while a config is loading/downloading (blocks Run + the picker)

  isLocked(name) {
    return !!name && Config.locked.includes(Config.withExt(name));
  },

  withExt(name) {
    return /\.ya?ml$/.test(name) ? name : name + '.yaml';
  },

  // First unused custom_run_N.yaml.
  nextCustomName() {
    let n = 0;
    for (const name of Config.known) {
      const m = name.match(/^custom_run_(\d+)\.ya?ml$/);
      if (m) n = Math.max(n, Number(m[1]));
    }
    return `custom_run_${n + 1}`;
  },

  // Note which config is loaded, and reflect it in the toolbar.
  setCurrent(name) {
    Config.current = name ? Config.withExt(name) : null;
    const nameInput = document.getElementById('config-name');
    if (nameInput && name) nameInput.value = name.replace(/\.ya?ml$/, '');
    Config.selected = Config.current || '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateSaveLabel();
  },

  // The Save button always writes to whatever name is in the field — typing a
  // different name creates a new config rather than renaming the current one.
  // Swap its label/title so that's clear without a separate "rename" control.
  updateSaveLabel() {
    const btn = document.getElementById('btn-save');
    const nameInput = document.getElementById('config-name');
    if (!btn || !nameInput) return;
    const typed = nameInput.value.trim();
    const label = btn.querySelector('.btn-label') || btn;
    const overwriting = !typed || Config.withExt(typed) === Config.current;
    label.textContent = overwriting ? 'Save' : 'Save as new';
    btn.title = !typed
      ? 'Enter a name to save this pipeline'
      : overwriting
      ? `Save changes to ${Config.current}`
      : `Save as a new pipeline: ${Config.withExt(typed)}`;
  },

  // Delete is only offered for a config that exists and isn't locked.
  updateControls() {
    const del = document.getElementById('btn-delete');
    if (!del) return;
    const chosen = Config.selected;
    del.disabled = !chosen || Config.isLocked(chosen);
    del.title = Config.isLocked(chosen)
      ? `${chosen} is a locked reference config and cannot be deleted`
      : 'Delete the selected config';
  },

  // ---- picker ----
  //
  // Choosing a config loads it straight away — there is no separate Load step.
  // Reference configs are drawn in the accent colour rather than tagged.

  // Put `text` into the config editor and the builder, keeping the raw YAML if
  // it doesn't map cleanly onto the registry.
  apply(text) {
    Config.loading = true;
    // fromYAML below pushes this same text into the builder, so the editor's
    // own change event must not *also* schedule a deferred YAML→builder sync:
    // it would land 400ms later and rebuild every step object, undoing whatever
    // happened in between — which is how a page refresh landing on a paused
    // step lost the step's unlocked, expanded state.
    syncingFromBuilder = true;
    editor.setValue(text);
    syncingFromBuilder = false;
    try { Config.fromYAML(text); refreshYAML(); }
    catch (e) { Config.notice('Loaded as raw YAML — the builder could not read it: ' + e.message); }
    Config.loading = false;
  },

  // Downloads the demo's NetCDF file first if it isn't on disk yet (see
  // app.py's _ensure_demo_file), which can take a while for a large file —
  // Config.busy locks the picker and Run button for the duration and keeps a
  // banner up so that wait is never silent or mistakeable for "did nothing".
  async load(name) {
    const needsDownload = Config.demo.includes(name) && !Config.downloaded.includes(name);
    if (needsDownload) {
      Config.setBusy(true, `Downloading ${Config.demoLabel(name)} demo data — this can take a while…`);
    }
    try {
      const { yaml_content } = await API.loadConfig(name);
      Config.apply(yaml_content);
      Config.setCurrent(name);
      Config.notice('');
      // Pick up the now-downloaded status so the picker stops offering to
      // download it again.
      if (needsDownload) await Config.refreshList(Config.selected);
    } finally {
      if (needsDownload) Config.setBusy(false);
    }
  },

  // Locks the config picker and Run button while a config is loading (in
  // particular, while a demo's data is downloading) so the old config can't
  // be run — and can't look like it silently reverted — mid-swap.
  setBusy(isBusy, message) {
    Config.busy = isBusy;
    const trigger = document.querySelector('#config-select .cfg-trigger');
    const runBtn = document.getElementById('btn-run');
    if (isBusy) {
      Config._runWasDisabled = runBtn ? runBtn.disabled : false;
      if (trigger) trigger.disabled = true;
      if (runBtn) runBtn.disabled = true;
      Config.notice(message, { sticky: true });
    } else {
      if (trigger) trigger.disabled = false;
      if (runBtn) runBtn.disabled = !!Config._runWasDisabled;
    }
  },

  // "demo_nelson.yaml" -> "Nelson", from the server-supplied label where
  // available (glider names repeat across missions, e.g. two Churchills, so
  // the key alone can't always be title-cased back into the right label).
  demoLabel(name) {
    return Config.labels[name] || name.replace(/^demo_/, '').replace(/\.ya?ml$/, '')
      .replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
  },

  renderPicker() {
    const root = document.getElementById('config-select');
    if (!root) return;
    const label = root.querySelector('.cfg-current');
    const isDemo = Config.demo.includes(Config.selected);
    label.textContent = Config.selected
      ? (isDemo ? 'Demo: ' + Config.demoLabel(Config.selected) : Config.selected)
      : '— saved configs —';
    label.classList.toggle('ref', Config.isLocked(Config.selected));
    label.classList.toggle('demo', isDemo);

    const menu = root.querySelector('.cfg-menu');
    menu.innerHTML = '';
    if (!Config.known.length) {
      const li = document.createElement('li');
      li.className = 'cfg-opt';
      li.textContent = 'no configs found';
      menu.appendChild(li);
      return;
    }

    const makeOpt = (name, { demo = false, hint } = {}) => {
      const locked = Config.locked.includes(name);
      const needsDownload = demo && !Config.downloaded.includes(name);
      const li = document.createElement('li');
      li.className = 'cfg-opt' + (locked ? ' ref' : '') + (demo ? ' demo' : '') +
        (needsDownload ? ' needs-download' : '') + (name === Config.selected ? ' selected' : '');
      li.setAttribute('role', 'option');
      // Not-yet-downloaded demos get a download icon instead of the plain
      // reference dot, so picking one visibly means "fetch its data first".
      if (needsDownload) {
        const dl = document.createElement('span');
        dl.className = 'cfg-opt-download';
        dl.title = 'Not downloaded yet — picking this fetches its data file first';
        dl.appendChild(Icon.el('download', 12));
        li.appendChild(dl);
      }
      const text = document.createElement('span');
      text.className = 'cfg-opt-text';
      text.textContent = demo ? Config.demoLabel(name) : name;
      li.appendChild(text);
      const hintText = hint ?? (needsDownload ? 'download' : locked ? 'reference' : '');
      if (hintText) {
        const h = document.createElement('span');
        h.className = 'cfg-hint';
        h.textContent = hintText;
        li.appendChild(h);
      }
      li.addEventListener('click', async () => {
        if (Config.busy) return; // a load/download is already in flight
        Config.closePicker();
        try { await Config.load(name); }
        catch (e) {
          Config.notice('Could not load ' + name + ': ' + e.message, { sticky: true, err: true });
        }
      });
      return li;
    };

    const group = (title) => {
      const h = document.createElement('li');
      h.className = 'cfg-group';
      h.textContent = title;
      menu.appendChild(h);
    };

    const referenceNames = Config.known.filter((n) => Config.reference.includes(n));
    const otherNames = Config.known.filter(
      (n) => !Config.demo.includes(n) && !Config.reference.includes(n)
    );

    // Demo configs are grouped by deployment mission (Config.missions), not
    // lumped into one list — the same glider name can appear in more than one
    // mission (e.g. Churchill, Zephyr), so which group it's under matters.
    for (const [mission, names] of Object.entries(Config.missions)) {
      const inThisMission = names.filter((n) => Config.known.includes(n));
      if (!inThisMission.length) continue;
      group(mission);
      for (const name of inThisMission) menu.appendChild(makeOpt(name, { demo: true }));
    }
    if (referenceNames.length) {
      group('Default');
      for (const name of referenceNames) menu.appendChild(makeOpt(name));
    }
    if (otherNames.length) {
      group('Your configs');
      for (const name of otherNames) menu.appendChild(makeOpt(name));
    }
  },

  // Re-read the folder on open, so files added/removed outside the dashboard
  // (via the Folder button) show up without a page reload.
  async openPicker() {
    const root = document.getElementById('config-select');
    await Config.refreshList(Config.selected);
    root.classList.add('open');
    root.querySelector('.cfg-menu').classList.remove('hidden');
    root.querySelector('.cfg-trigger').setAttribute('aria-expanded', 'true');
  },

  closePicker() {
    const root = document.getElementById('config-select');
    if (!root) return;
    root.classList.remove('open');
    root.querySelector('.cfg-menu').classList.add('hidden');
    root.querySelector('.cfg-trigger').setAttribute('aria-expanded', 'false');
  },

  // Called on every edit. Editing a locked config forks it: the save target
  // becomes a new custom_run_N so the original can't be written over.
  noteEdit() {
    if (Config.loading || !Config.isLocked(Config.current)) return;
    const forked = Config.nextCustomName();
    const from = Config.current;
    Config.current = Config.withExt(forked);
    document.getElementById('config-name').value = forked;
    // Nothing on the server is selected any more — this is a new, unsaved config.
    Config.selected = '';
    Config.renderPicker();
    Config.updateControls();
    Config.updateSaveLabel();
    Config.notice(`${from} is locked — your changes will save as ${forked}.yaml`);
  },

  // sticky: stays up until the next notice() call instead of auto-clearing —
  // used for errors and while a demo download is in progress, so neither is
  // ever missed because it faded out.
  notice(text, { sticky = false, err = false } = {}) {
    const el = document.getElementById('config-note');
    if (!el) return;
    el.textContent = text;
    el.classList.toggle('hidden', !text);
    el.classList.toggle('err', !!err);
    clearTimeout(Config._noticeTimer);
    if (text && !sticky) Config._noticeTimer = setTimeout(() => Config.notice(''), 8000);
  },

  async refreshList(selected) {
    const info = await API.listConfigs();
    Config.known = info.configs || [];
    Config.locked = info.protected || [];
    Config.demo = info.demo || [];
    Config.missions = info.missions || {};
    Config.labels = info.labels || {};
    Config.reference = info.reference || [];
    Config.downloaded = info.downloaded || [];
    if (selected !== undefined) Config.selected = selected || '';
    if (Config.selected && !Config.known.includes(Config.selected)) Config.selected = '';
    Config.renderPicker();
    Config.updateControls();
  },
};
