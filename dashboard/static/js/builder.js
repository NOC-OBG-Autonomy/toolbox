// The pipeline builder: step palette, the ordered list of steps, and the
// schema-driven forms inside each. Everything here is derived from the live
// registry, so new steps need no builder changes.

// `nodes` is the ordered pipeline: each entry is either a step item or a
// section (a named, contiguous run of steps). `items` is the flattened step
// list, kept in sync by syncItems() — everything outside the builder (YAML
// generation, validation indices, YAML⇄builder highlighting) reads that.
const STATE = {
  registry: null,
  stepsByName: {},
  qcByName: {},
  pipeline: { settings: {}, nodes: [], items: [] },
  _seq: 0,
  onChange: () => {}, // wired by app.js to regenerate YAML
};

function isSection(n) { return !!n && n.kind === 'section'; }

function makeSection(title = 'New section') {
  return { kind: 'section', id: ++STATE._seq, title, collapsed: false, steps: [] };
}

function syncItems() {
  const out = [];
  for (const n of STATE.pipeline.nodes) {
    if (isSection(n)) out.push(...n.steps);
    else out.push(n);
  }
  STATE.pipeline.items = out;
}

// Where does a step live? -> {list, index, section} (section null when loose).
function locateStep(id) {
  const nodes = STATE.pipeline.nodes;
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (isSection(n)) {
      const k = n.steps.findIndex((s) => s.id === id);
      if (k >= 0) return { list: n.steps, index: k, section: n };
    } else if (n.id === id) {
      return { list: nodes, index: i, section: null };
    }
  }
  return null;
}

function sectionOfStep(id) {
  const loc = locateStep(id);
  return loc ? loc.section : null;
}

// A step is treated as a QC container if it declares a `qc_settings` dict param
// (i.e. the Apply QC step). Detected by shape, not by name, so a future
// QC-applying step gets the same smart editor for free.
function isQcContainer(def) {
  return (def.parameters || []).some(
    (p) => p.name === 'qc_settings' && (p.type === 'dict' || (Array.isArray(p.type) && p.type.includes('dict')))
  );
}

function initValues(def) {
  const v = {};
  for (const spec of def.parameters || []) v[spec.name] = Forms.defaultValue(spec);
  return v;
}

// ------------------------------------------------------- all-diagnostics
// A step-count-agnostic "on / off / custom" summary of every diagnostics
// switch in the pipeline (step-level, plus per-QC-test where those exist),
// so the pipeline-settings card can offer one place to flip them all at once.
// Only leaves that actually gate a plot are counted: for an Apply QC step
// with tests configured, that's each test's *effective* value (its own
// override, or the step's master); an Apply QC step with no tests yet counts
// its master, since that's what a newly added test would inherit.
function diagnosticsLeaves() {
  const leaves = [];
  for (const item of STATE.pipeline.items) {
    if (isQcContainer(item.def)) {
      const tests = Object.keys(item.values.qc_settings || {});
      if (!tests.length) { leaves.push(item.diagnostics); continue; }
      for (const name of tests) {
        const tv = item.values.qc_settings[name];
        leaves.push('diagnostics' in tv ? !!tv.diagnostics : !!item.diagnostics);
      }
    } else {
      leaves.push(!!item.diagnostics);
    }
  }
  return leaves;
}

function allDiagnosticsState() {
  const leaves = diagnosticsLeaves();
  if (!leaves.length) return 'off';
  if (leaves.every((v) => v)) return 'on';
  if (leaves.every((v) => !v)) return 'off';
  return 'custom';
}

// Flip every diagnostics switch in the pipeline to `v`, the same way a single
// step's own switch does (and, for Apply QC, clearing per-test overrides so
// every test unambiguously follows the master).
function setAllDiagnostics(v) {
  for (const item of STATE.pipeline.items) {
    item.diagnostics = v;
    if (isQcContainer(item.def)) {
      for (const name of Object.keys(item.values.qc_settings || {})) {
        delete item.values.qc_settings[name].diagnostics;
      }
    }
  }
  renderPipeline();
  STATE.onChange();
}

// ---------------------------------------------------------------- palette
function renderPalette(filter = '') {
  const list = document.getElementById('palette-list');
  list.innerHTML = '';
  const f = filter.trim().toLowerCase();
  const cats = [
    ['input_output', 'Input / Output'],
    ['processing', 'Processing'],
    ['quality_control', 'Quality Control'],
    ['other', 'Other'],
  ];
  const all = STATE.registry.steps;
  for (const [cat, title] of cats) {
    const items = all.filter(
      (s) => s.category === cat &&
        (!f || s.name.toLowerCase().includes(f) || (s.description || '').toLowerCase().includes(f))
    );
    if (!items.length) continue;
    const group = document.createElement('div');
    group.className = 'cat-group cat-' + cat;
    const h = document.createElement('div');
    h.className = 'cat-title'; h.textContent = title;
    group.appendChild(h);
    for (const s of items) {
      const el = document.createElement('div');
      el.className = 'palette-item';
      el.draggable = true;
      el.innerHTML = `<div class="pi-name">${s.name}</div>` +
        (s.description ? `<div class="pi-desc">${s.description}</div>` : '');
      el.onclick = () => addStep(s.name); // click still adds to the end
      el.addEventListener('dragstart', (e) => {
        el.classList.add('dragging');
        dragState = { kind: 'new', name: s.name };
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('text/plain', s.name); // Firefox needs some data set
      });
      el.addEventListener('dragend', () => {
        el.classList.remove('dragging'); clearDropIndicator(); dragState = null;
      });
      group.appendChild(el);
    }
    list.appendChild(group);
  }
}

// ---------------------------------------------------------------- add/mutate
function makeItem(name) {
  const def = STATE.stepsByName[name];
  if (!def) return null;
  const item = {
    id: ++STATE._seq,
    name,
    def,
    values: initValues(def),
    diagnostics: false,
    collapsed: false,
  };
  if (isQcContainer(def)) item.values.qc_settings = {}; // ordered map of qc tests
  return item;
}

// Clicking a palette item appends to the end of the pipeline, which means
// inside the trailing section when there is one.
function addStep(name) {
  const item = makeItem(name);
  if (!item) return;
  const nodes = STATE.pipeline.nodes;
  const last = nodes[nodes.length - 1];
  if (isSection(last)) last.steps.push(item);
  else nodes.push(item);
  renderPipeline();
  STATE.onChange();
}

// Insert a new step (dragged from the palette) into `list` at `index`.
function insertStepAt(name, list, index) {
  const item = makeItem(name);
  if (!item) return;
  list.splice(Math.max(0, Math.min(index, list.length)), 0, item);
  renderPipeline();
  STATE.onChange();
}

function removeStep(id) {
  const loc = locateStep(id);
  if (!loc) return;
  loc.list.splice(loc.index, 1);
  renderPipeline();
  STATE.onChange();
}

// Up/down buttons walk the step through the pipeline in execution order,
// crossing section boundaries (out of a section, or into the neighbouring one).
function moveStep(id, dir) {
  const loc = locateStep(id);
  if (!loc) return;
  const nodes = STATE.pipeline.nodes;
  const to = loc.index + dir;

  if (to >= 0 && to < loc.list.length) {
    const neighbour = loc.list[to];
    if (isSection(neighbour)) { // loose step at root meeting a section: step in
      const [it] = loc.list.splice(loc.index, 1);
      if (dir < 0) neighbour.steps.push(it); else neighbour.steps.unshift(it);
    } else {
      [loc.list[loc.index], loc.list[to]] = [loc.list[to], loc.list[loc.index]];
    }
  } else if (loc.section) { // off the edge of a section: pop out to root
    const at = nodes.indexOf(loc.section);
    const [it] = loc.list.splice(loc.index, 1);
    nodes.splice(dir < 0 ? at : at + 1, 0, it);
  } else {
    return; // already at the top/bottom of the pipeline
  }
  renderPipeline();
  STATE.onChange();
}

// Move an existing step to a drop position in `list` (index counts the dragged
// item itself when it is already in that list).
function moveStepTo(id, list, index) {
  const loc = locateStep(id);
  if (!loc) return;
  let to = index;
  if (loc.list === list && loc.index < index) to -= 1; // removal shifts later positions left
  const [it] = loc.list.splice(loc.index, 1);
  list.splice(Math.max(0, Math.min(to, list.length)), 0, it);
  renderPipeline();
  STATE.onChange();
}

// ---------------------------------------------------------------- sections
function addSection() {
  STATE.pipeline.nodes.push(makeSection());
  renderPipeline();
  STATE.onChange();
}

// Deleting a section removes it and every step inside it.
function removeSection(id) {
  const nodes = STATE.pipeline.nodes;
  const at = nodes.findIndex((n) => isSection(n) && n.id === id);
  if (at < 0) return;
  nodes.splice(at, 1);
  renderPipeline();
  STATE.onChange();
}

// Move a whole section (with its steps) to a root position.
function moveSectionTo(id, index) {
  const nodes = STATE.pipeline.nodes;
  const from = nodes.findIndex((n) => isSection(n) && n.id === id);
  if (from < 0) return;
  let to = index;
  if (from < index) to -= 1;
  to = Math.max(0, Math.min(to, nodes.length - 1));
  if (to === from) return;
  const [sec] = nodes.splice(from, 1);
  nodes.splice(to, 0, sec);
  renderPipeline();
  STATE.onChange();
}

// ------------------------------------------------------------- drag & drop
// Set on dragstart: {kind:'new', name} from the palette, {kind:'move', id} for
// reordering an existing step, or {kind:'section', id} for a whole section.
// Read by the pipeline drop target.
let dragState = null;
let dropIndicator = null;

// The drop host under the pointer: a section body, or the root (loose steps).
// Sections never nest, so a section drag always resolves to the root.
function dropHostAt(target) {
  const root = document.getElementById('pipeline-steps');
  if (dragState && dragState.kind === 'section') return root;
  const body = target && target.closest ? target.closest('.section-body') : null;
  return body || root;
}

// The list a host writes into: a section's steps, or the root node list.
function listForHost(host) {
  if (!host.classList.contains('section-body')) return STATE.pipeline.nodes;
  const sec = STATE.pipeline.nodes.find(
    (n) => isSection(n) && n.id === Number(host.dataset.secId)
  );
  return sec ? sec.steps : STATE.pipeline.nodes;
}

// Direct children that occupy a drop slot (the root also holds section cards).
function dropSlots(host) {
  return [...host.children].filter(
    (el) => el.classList.contains('step-card') || el.classList.contains('section-card')
  );
}

function computeDropIndex(host, y) {
  const slots = dropSlots(host);
  for (let i = 0; i < slots.length; i++) {
    const r = slots[i].getBoundingClientRect();
    if (y < r.top + r.height / 2) return i;
  }
  return slots.length;
}

function showDropIndicator(host, y) {
  if (!dropIndicator) {
    dropIndicator = document.createElement('div');
    dropIndicator.className = 'drop-indicator';
  }
  const index = computeDropIndex(host, y);
  const slots = dropSlots(host);
  if (index >= slots.length) host.appendChild(dropIndicator);
  else host.insertBefore(dropIndicator, slots[index]);
}

function clearDropIndicator() {
  if (dropIndicator && dropIndicator.parentNode) dropIndicator.remove();
  document.querySelectorAll('.drag-active').forEach((el) => el.classList.remove('drag-active'));
}

// Wire the pipeline area as a drop target. Delegated from the root, so section
// bodies added by later renders are picked up without re-wiring. Called once at
// boot.
function initBuilderDnD() {
  const root = document.getElementById('pipeline-steps');
  root.addEventListener('dragover', (e) => {
    if (!dragState) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = dragState.kind === 'new' ? 'copy' : 'move';
    const host = dropHostAt(e.target);
    clearDropIndicator();
    host.classList.add('drag-active');
    showDropIndicator(host, e.clientY);
  });
  root.addEventListener('dragleave', (e) => {
    if (!root.contains(e.relatedTarget)) clearDropIndicator();
  });
  root.addEventListener('drop', (e) => {
    if (!dragState) return;
    e.preventDefault();
    const host = dropHostAt(e.target);
    const index = computeDropIndex(host, e.clientY);
    const list = listForHost(host);
    const st = dragState;
    clearDropIndicator();
    dragState = null;
    if (st.kind === 'new') insertStepAt(st.name, list, index);
    else if (st.kind === 'section') moveSectionTo(st.id, index);
    else moveStepTo(st.id, list, index);
  });
}

// ---------------------------------------------------------------- settings
// Collapse state for the settings card, kept across re-renders (module-level
// rather than on STATE.pipeline, which is replaced whenever a config loads).
let settingsCollapsed = false;

function renderSettings() {
  const host = document.getElementById('pipeline-settings');
  host.innerHTML = '';
  const card = document.createElement('div');
  card.className = 'step-card settings';
  const head = document.createElement('div');
  head.className = 'step-head';
  head.innerHTML = '<span class="step-name">Pipeline settings</span>';
  const toggle = document.createElement('button');
  toggle.className = 'icon-btn';
  toggle.innerHTML = Icon.svg(settingsCollapsed ? 'right' : 'down');
  toggle.title = 'collapse';
  toggle.onclick = (e) => { e.stopPropagation(); settingsCollapsed = !settingsCollapsed; renderSettings(); };
  head.appendChild(toggle);
  head.addEventListener('click', (e) => {
    if (e.target.closest('button')) return;
    settingsCollapsed = !settingsCollapsed;
    renderSettings();
  });
  card.appendChild(head);
  const body = document.createElement('div');
  body.className = 'step-body' + (settingsCollapsed ? ' collapsed' : '');
  for (const spec of STATE.registry.pipeline_fields) {
    if (!(spec.name in STATE.pipeline.settings)) {
      STATE.pipeline.settings[spec.name] = Forms.defaultValue(spec);
    }
    body.appendChild(Forms.render(spec, STATE.pipeline.settings, STATE.onChange));
  }
  body.appendChild(makeAllDiagnosticsRow());
  card.appendChild(body);
  host.appendChild(card);
  applyRunLock();
  renderAllDiagnosticsRow();
}

// A UI-only control (nothing here is written to the YAML — it just drives
// every step's own `diagnostics`/per-test switch at once) that summarises
// whether diagnostics are uniformly on, uniformly off, or a mix, with one
// click to force them all one way. Kept live via renderAllDiagnosticsRow(),
// called on every pipeline change so it never shows a stale summary.
function makeAllDiagnosticsRow() {
  const row = document.createElement('div');
  row.id = 'all-diag-row';
  row.className = 'diag-row all-diag-row';
  const label = document.createElement('span');
  label.className = 'switch-label all-diag-label';
  label.textContent = 'All diagnostics';
  row.appendChild(label);
  const state = document.createElement('span');
  state.id = 'all-diag-state';
  state.className = 'all-diag-state';
  row.appendChild(state);
  const onBtn = document.createElement('button');
  onBtn.id = 'all-diag-on';
  onBtn.className = 'ghost seg-btn';
  onBtn.textContent = 'On';
  onBtn.onclick = () => setAllDiagnostics(true);
  row.appendChild(onBtn);
  const offBtn = document.createElement('button');
  offBtn.id = 'all-diag-off';
  offBtn.className = 'ghost seg-btn';
  offBtn.textContent = 'Off';
  offBtn.onclick = () => setAllDiagnostics(false);
  row.appendChild(offBtn);
  return row;
}

// Refresh just the all-diagnostics row's text/button state in place, without
// rebuilding the rest of the settings card (which would drop focus from
// whatever pipeline-settings field the user is mid-edit on).
function renderAllDiagnosticsRow() {
  const row = document.getElementById('all-diag-row');
  if (!row) return;
  const state = allDiagnosticsState();
  const label = { on: 'On', off: 'Off', custom: 'Custom' }[state];
  const stateEl = document.getElementById('all-diag-state');
  if (stateEl) {
    stateEl.textContent = label;
    stateEl.className = 'all-diag-state all-diag-' + state;
  }
  const onBtn = document.getElementById('all-diag-on');
  const offBtn = document.getElementById('all-diag-off');
  if (onBtn) onBtn.classList.toggle('active', state === 'on');
  if (offBtn) offBtn.classList.toggle('active', state === 'off');
}

// ---------------------------------------------------------------- run lock
//
// While a pipeline is running the config is frozen: the run is executing the
// YAML as it was submitted, so an edit anywhere else would leave the screen
// disagreeing with what is actually running. When the run pauses on a step, that
// step alone is unlocked — for a QC step split test by test, only the paused
// test — because that is exactly what Re-run will act on.
const RunLock = {
  running: false,
  index: null,   // step left editable while paused, or null for "all locked"
  test: null,    // QC test within it, when the step was split
  runningIndex: null, // step currently executing (not paused), for the highlight

  begin() { RunLock.set(true, null, null); },
  end() { RunLock.runningIndex = null; RunLock.set(false, null, null); },

  // Unlock the paused step and bring it into view, expanded. Passing a null
  // index re-locks everything (still running, no longer paused).
  pauseAt(index, test) {
    RunLock.runningIndex = null; // it's paused now, not mid-execution
    if (index === null || index === undefined) {
      RunLock.set(true, null, null);
      return;
    }
    // Collapse everything else, so the one open card is the editable one.
    STATE.pipeline.items.forEach((s, i) => { s.collapsed = i !== index; });
    settingsCollapsed = true;
    // Open the paused test *before* rendering, or an already-expanded card
    // would keep its old sections open.
    const item = STATE.pipeline.items[index];
    if (item && test) item.qcOpen = { [test]: true };
    RunLock.set(true, index, test || null);
    focusStepInBuilder(index);
  },

  // Called as each step (or split QC test) starts executing. Collapses every
  // card — so the previous step's editor doesn't just sit there open while
  // it's greyed out and no longer relevant — and calls out the running one
  // with a highlight, scrolled into view.
  stepStarted(index) {
    if (index === RunLock.runningIndex) return; // duplicate marker, e.g. a re-run
    RunLock.runningIndex = index;
    STATE.pipeline.items.forEach((s) => { s.collapsed = true; s.qcOpen = null; });
    settingsCollapsed = true;
    // The card stays collapsed, but its section must be open or it wouldn't
    // be in view to scroll to at all.
    const item = STATE.pipeline.items[index];
    const sec = item && sectionOfStep(item.id);
    if (sec) sec.collapsed = false;
    renderPipeline();
    const card = document.querySelectorAll('#pipeline-steps .step-card')[index];
    if (card) card.scrollIntoView({ block: 'center', behavior: 'smooth' });
  },

  set(running, index, test) {
    RunLock.running = running;
    RunLock.index = index;
    RunLock.test = test;
    document.body.classList.toggle('run-locked', running);
    // The YAML pane is the other way into the config, so it locks with it.
    if (typeof editor !== 'undefined' && editor) {
      editor.setOption('readOnly', running ? 'nocursor' : false);
    }
    renderSettings();
    renderPipeline();
  },

  // True for the one step card the user may edit right now.
  editable(index) {
    return !RunLock.running || RunLock.index === index;
  },
};

// Disable every control under `root`, optionally sparing the subtree `keep`.
function freezeControls(root, keep) {
  for (const el of root.querySelectorAll('input, select, textarea, button')) {
    if (keep && keep.contains(el)) continue;
    el.disabled = true;
  }
}

// Apply the run lock to what has just been rendered. Called at the end of
// renderPipeline/renderSettings so every path through the builder is covered.
function applyRunLock() {
  const running = RunLock.running;
  // Anything that would add, remove or replace steps wholesale, including
  // loading another config out from under the run.
  for (const id of ['btn-clear', 'btn-add-section']) {
    const b = document.getElementById(id);
    if (b) b.disabled = running;
  }
  const picker = document.querySelector('#config-select .cfg-trigger');
  if (picker) {
    picker.disabled = running;
    picker.title = running ? 'Locked while the pipeline is running' : '';
  }
  if (!running) return;
  const settings = document.getElementById('pipeline-settings');
  if (settings) {
    settings.classList.add('locked');
    freezeControls(settings);
  }
  const cards = document.querySelectorAll('#pipeline-steps .step-card');
  cards.forEach((card, index) => {
    if (!RunLock.editable(index)) {
      card.classList.add('locked');
      card.classList.toggle('running', index === RunLock.runningIndex);
      freezeControls(card);
      return;
    }
    card.classList.add('unlocked');
    // Structural controls stay frozen even on the unlocked card: reordering or
    // removing it mid-run would break the step indices the runner is using.
    for (const b of card.querySelectorAll('.step-head button')) {
      if (b.title !== 'collapse') b.disabled = true;
    }
    // A split QC step exposes only the paused test's fields.
    const test = RunLock.test && card.querySelector(
      `.qc-test[data-test="${CSS.escape(RunLock.test)}"] .qc-test-body`);
    if (RunLock.test) freezeControls(card.querySelector('.step-body'), test || undefined);
  });
}

// ---------------------------------------------------------------- steps
function renderPipeline() {
  const host = document.getElementById('pipeline-steps');
  host.innerHTML = '';
  syncItems();
  document.getElementById('empty-hint').style.display =
    STATE.pipeline.nodes.length ? 'none' : 'block';

  let secIndex = 0;
  for (const node of STATE.pipeline.nodes) {
    host.appendChild(isSection(node)
      ? renderSection(node, secIndex++)
      : renderStepCard(node));
  }
  applyRunLock();
  renderAllDiagnosticsRow(); // step list just changed shape: on/off/custom may have too
}

// A section's colour: dark green if it holds CHLA Quenching, orange if it holds
// Find Profiles, otherwise plain alternating blue-grey / grey by position.
function sectionColourClass(sec, index) {
  const names = sec.steps.map((s) => (s.name || '').toLowerCase());
  if (names.some((n) => n.includes('chla') && n.includes('quench'))) return 'sec-chla';
  if (names.some((n) => n.includes('find profiles'))) return 'sec-profiles';
  return index % 2 === 0 ? 'sec-alt-0' : 'sec-alt-1';
}

// One section: a titled, collapsible container holding a contiguous run of
// steps. The body is its own drop target (see dropHostAt).
function renderSection(sec, index = 0) {
  const card = document.createElement('div');
  card.className = 'section-card ' + sectionColourClass(sec, index) +
    (sec.collapsed ? ' collapsed' : '');

  const head = document.createElement('div');
  head.className = 'section-head';
  head.innerHTML =
    `<span class="drag" title="Drag to move the whole section">${Icon.svg('grip')}</span>` +
    `<span class="sec-chevron">${Icon.svg(sec.collapsed ? 'right' : 'down', 14)}</span>`;

  const title = document.createElement('input');
  title.className = 'section-title';
  title.value = sec.title;
  title.placeholder = 'Section name';
  title.oninput = () => { sec.title = title.value; STATE.onChange(); };
  title.onclick = (e) => e.stopPropagation(); // clicking the name shouldn't collapse
  head.appendChild(title);

  const count = document.createElement('span');
  count.className = 'sec-count';
  count.textContent = sec.steps.length + (sec.steps.length === 1 ? ' step' : ' steps');
  head.appendChild(count);

  const del = document.createElement('button');
  del.className = 'icon-btn';
  del.innerHTML = Icon.svg('close');
  del.title = 'remove section and its steps';
  del.onclick = (e) => { e.stopPropagation(); removeSection(sec.id); };
  head.appendChild(del);

  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag')) return;
    sec.collapsed = !sec.collapsed;
    renderPipeline();
  });

  // Same handle-gated drag as step cards, so the title stays editable.
  const handle = head.querySelector('.drag');
  handle.addEventListener('mousedown', () => { card.draggable = true; });
  handle.addEventListener('mouseup', () => { card.draggable = false; });
  card.addEventListener('dragstart', (e) => {
    dragState = { kind: 'section', id: sec.id };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', 'section-' + sec.id);
    card.classList.add('dragging');
    e.stopPropagation();
  });
  card.addEventListener('dragend', () => {
    card.draggable = false;
    card.classList.remove('dragging');
    clearDropIndicator();
    dragState = null;
  });

  const body = document.createElement('div');
  body.className = 'section-body';
  body.dataset.secId = String(sec.id);
  if (sec.collapsed) body.style.display = 'none';
  for (const item of sec.steps) body.appendChild(renderStepCard(item));
  if (!sec.steps.length) {
    const hint = document.createElement('div');
    hint.className = 'hint sec-empty';
    hint.textContent = 'Drop steps here.';
    body.appendChild(hint);
  }

  card.appendChild(head);
  card.appendChild(body);
  return card;
}

function renderStepCard(item) {
  const card = document.createElement('div');
  card.className = 'step-card cat-' + item.def.category;

  // header
  const head = document.createElement('div');
  head.className = 'step-head';
  head.innerHTML =
    `<span class="drag" title="Drag to reorder">${Icon.svg('grip')}</span>`;

  // Up/down stacked into one compact control, sat right next to the grip.
  const move = document.createElement('span');
  move.className = 'step-move';
  const mkMove = (ico, delta, title) => {
    const b = document.createElement('button');
    b.className = 'icon-btn move-btn'; b.innerHTML = Icon.svg(ico, 12); b.title = title;
    b.onclick = (e) => { e.stopPropagation(); moveStep(item.id, delta); };
    return b;
  };
  move.appendChild(mkMove('up', -1, 'move up'));
  move.appendChild(mkMove('down', 1, 'move down'));
  head.appendChild(move);

  // Name, plus — for a collapsed Apply QC step — the QC tests it holds as chips
  // inline on the same bar, so its contents are visible without opening it.
  // (Expanded, the QC editor below shows them, so they'd only be duplicated.)
  const title = document.createElement('span');
  title.className = 'step-title';
  const name = document.createElement('span');
  name.className = 'step-name'; name.textContent = item.name;
  title.appendChild(name);
  if (isQcContainer(item.def) && item.collapsed) {
    for (const t of Object.keys(item.values.qc_settings || {})) {
      const chip = document.createElement('span');
      chip.className = 'qc-chip';
      chip.textContent = t;
      title.appendChild(chip);
    }
  }
  head.appendChild(title);

  // Expand/collapse: a quiet chevron that rotates to point at its state.
  const expand = document.createElement('button');
  expand.className = 'icon-btn step-expand' + (item.collapsed ? ' collapsed' : '');
  expand.innerHTML = Icon.svg('down', 15);
  expand.title = item.collapsed ? 'expand' : 'collapse';
  expand.setAttribute('aria-expanded', String(!item.collapsed));
  expand.onclick = (e) => { e.stopPropagation(); item.collapsed = !item.collapsed; renderPipeline(); };
  head.appendChild(expand);

  const del = document.createElement('button');
  del.className = 'icon-btn step-del'; del.innerHTML = Icon.svg('close'); del.title = 'remove';
  del.onclick = (e) => { e.stopPropagation(); removeStep(item.id); };
  head.appendChild(del);
  card.appendChild(head);

  // Click anywhere on the header (except a button or the drag handle) toggles.
  head.addEventListener('click', (e) => {
    if (e.target.closest('button') || e.target.closest('.drag')) return;
    item.collapsed = !item.collapsed;
    renderPipeline();
  });

  // Reorder by dragging the ⋮⋮ handle: the card is only draggable while the
  // handle is held, so text selection in the body still works normally.
  const handle = head.querySelector('.drag');
  handle.addEventListener('mousedown', () => { card.draggable = true; });
  handle.addEventListener('mouseup', () => { card.draggable = false; });
  card.addEventListener('dragstart', (e) => {
    dragState = { kind: 'move', id: item.id };
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', String(item.id));
    card.classList.add('dragging');
  });
  card.addEventListener('dragend', () => {
    card.draggable = false;
    card.classList.remove('dragging');
    clearDropIndicator();
    dragState = null;
  });

  // Interacting with a step highlights its lines in the YAML pane.
  card.addEventListener('mousedown', () => highlightYamlForStep(item.id));
  card.addEventListener('focusin', () => highlightYamlForStep(item.id));

  // body
  const body = document.createElement('div');
  body.className = 'step-body' + (item.collapsed ? ' collapsed' : '');

  if (!item.def.schema_declared) {
    const note = document.createElement('div');
    note.className = 'hint';
    note.textContent = 'This step has not declared a parameter schema yet; edit its parameters in the YAML pane.';
    body.appendChild(note);
  }

  for (const spec of item.def.parameters || []) {
    if (spec.name === 'qc_settings' && isQcContainer(item.def)) {
      body.appendChild(renderQcEditor(item, spec));
    } else {
      body.appendChild(Forms.render(spec, item.values, STATE.onChange));
    }
  }

  // diagnostics toggle (every step supports it). For the QC container the
  // diagnostics controls live inside the QC editor (a master + per-test), so
  // don't render a second one here.
  if (!isQcContainer(item.def)) {
    const diag = document.createElement('div');
    diag.className = 'diag-row';
    const sw = Forms.switchEl(item.diagnostics, (v) => { item.diagnostics = v; renderAllDiagnosticsRow(); STATE.onChange(); });
    sw.input.id = 'diag-' + item.id;
    const lbl = document.createElement('label');
    lbl.className = 'switch-label';
    lbl.htmlFor = sw.input.id;
    lbl.textContent = 'diagnostics'; lbl.style.margin = '0';
    diag.appendChild(sw.el); diag.appendChild(lbl);
    body.appendChild(diag);
  }

  card.appendChild(body);
  return card;
  }

// Expand and scroll to a step card by index (driven by the YAML cursor). Only
// re-renders when it has to un-collapse the target, to stay cheap on every
// cursor move.
function focusStepInBuilder(index) {
  const items = STATE.pipeline.items;
  if (index < 0 || index >= items.length) return;
  const sec = sectionOfStep(items[index].id);
  if (items[index].collapsed || (sec && sec.collapsed)) {
    items[index].collapsed = false;
    if (sec) sec.collapsed = false;
    renderPipeline();
  }
  const card = document.querySelectorAll('#pipeline-steps .step-card')[index];
  if (!card) return;
  card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  card.classList.add('flash');
  setTimeout(() => card.classList.remove('flash'), 600);
}

// ---------------------------------------------------------------- QC editor
function renderQcEditor(item, spec) {
  const wrap = document.createElement('div');
  wrap.className = 'field';
  const label = document.createElement('label');
  label.textContent = 'QC tests';
  wrap.appendChild(label);
  if (spec.description) {
    const hint = document.createElement('div');
    hint.className = 'hint'; hint.textContent = spec.description;
    wrap.appendChild(hint);
  }

  const editor = document.createElement('div');
  editor.className = 'qc-editor';

  // master diagnostics toggle: turns diagnostic plots on/off for ALL tests.
  // Bound to the step-level `item.diagnostics` (the inherited default); flipping
  // it clears every per-test override so "all on / all off" is unambiguous.
  const master = document.createElement('div');
  master.className = 'qc-master';
  const msw = Forms.switchEl(item.diagnostics, (v) => {
    item.diagnostics = v;
    for (const name of Object.keys(item.values.qc_settings)) {
      delete item.values.qc_settings[name].diagnostics;
    }
    renderPipeline();
    STATE.onChange();
  });
  msw.input.id = 'qc-master-' + item.id;
  const mlbl = document.createElement('label');
  mlbl.className = 'switch-label';
  mlbl.htmlFor = msw.input.id;
  mlbl.textContent = 'diagnostics — all tests';
  master.appendChild(msw.el); master.appendChild(mlbl);
  editor.appendChild(master);

  // add-test row
  const addRow = document.createElement('div');
  addRow.className = 'qc-add';
  const sel = document.createElement('select');
  const ph = document.createElement('option');
  ph.value = ''; ph.textContent = '— add a QC test —';
  sel.appendChild(ph);
  for (const qc of STATE.registry.qc) {
    const o = document.createElement('option');
    o.value = qc.name; o.textContent = qc.name;
    sel.appendChild(o);
  }
  const addBtn = document.createElement('button');
  addBtn.className = 'ghost'; addBtn.innerHTML = Icon.svg('plus') + 'Add';
  addBtn.onclick = () => {
    const name = sel.value;
    if (!name || name in item.values.qc_settings) return;
    item.values.qc_settings[name] = initValues(STATE.qcByName[name]);
    sel.value = '';
    renderPipeline();
    STATE.onChange();
  };
  addRow.appendChild(sel); addRow.appendChild(addBtn);
  editor.appendChild(addRow);

  // configured tests (insertion order = application order)
  for (const qcName of Object.keys(item.values.qc_settings)) {
    const qcDef = STATE.qcByName[qcName];
    const testValues = item.values.qc_settings[qcName];
    const test = document.createElement('div');
    test.className = 'qc-test';
    test.dataset.test = qcName;

    const th = document.createElement('div');
    th.className = 'qc-test-head';
    // Which tests are expanded is remembered on the item (UI state, never
    // serialised), so a re-render — or a pause unlocking one test — can open
    // the right one.
    const openNow = !!(item.qcOpen && item.qcOpen[qcName]);
    const chev = document.createElement('span');
    chev.className = 'qc-chevron';
    chev.innerHTML = Icon.svg(openNow ? 'down' : 'right', 14);
    th.appendChild(chev);
    th.insertAdjacentHTML('beforeend', `<span class="qc-test-name">${qcName}</span>`);
    const rm = document.createElement('button');
    rm.className = 'icon-btn'; rm.innerHTML = Icon.svg('close'); rm.title = 'remove test';
    rm.onclick = (e) => {
      e.stopPropagation();
      delete item.values.qc_settings[qcName];
      renderPipeline(); STATE.onChange();
    };
    th.appendChild(rm);

    const tb = document.createElement('div');
    tb.className = 'qc-test-body';
    tb.style.display = openNow ? 'block' : 'none';
    th.onclick = () => {
      const open = tb.style.display === 'none';
      tb.style.display = open ? 'block' : 'none';
      chev.innerHTML = Icon.svg(open ? 'down' : 'right', 14);
      item.qcOpen = Object.assign({}, item.qcOpen, { [qcName]: open });
    };

    // metadata: what the test does, needs, and produces (self-documenting even
    // when the test has no tunable parameters).
    if (qcDef && qcDef.description) {
      const d = document.createElement('div');
      d.className = 'hint qc-meta'; d.textContent = qcDef.description;
      tb.appendChild(d);
    }
    const reqs = (qcDef && qcDef.required_variables) || [];
    const outs = (qcDef && qcDef.qc_outputs) || [];
    if (reqs.length) {
      const r = document.createElement('div');
      r.className = 'hint qc-meta'; r.textContent = 'Requires: ' + reqs.join(', ');
      tb.appendChild(r);
    }
    if (outs.length) {
      const o = document.createElement('div');
      o.className = 'hint qc-meta'; o.textContent = 'Outputs: ' + outs.join(', ');
      tb.appendChild(o);
    }

    // per-test diagnostics override. Absent => inherit the master; toggling
    // writes an explicit true/false for this test only.
    const diagDefault = 'diagnostics' in testValues ? testValues.diagnostics : item.diagnostics;
    const drow = document.createElement('div');
    drow.className = 'diag-row';
    const dsw = Forms.switchEl(!!diagDefault, (v) => { testValues.diagnostics = v; renderAllDiagnosticsRow(); STATE.onChange(); });
    dsw.input.id = 'qc-diag-' + item.id + '-' + qcName.replace(/\s+/g, '_');
    const dlbl = document.createElement('label');
    dlbl.className = 'switch-label'; dlbl.htmlFor = dsw.input.id;
    dlbl.textContent = 'diagnostics'; dlbl.style.margin = '0';
    drow.appendChild(dsw.el); drow.appendChild(dlbl);
    tb.appendChild(drow);

    const params = (qcDef && qcDef.parameters) || [];
    if (!params.length) {
      const none = document.createElement('div');
      none.className = 'hint'; none.textContent = 'No parameters.';
      tb.appendChild(none);
    }
    for (const p of params) tb.appendChild(Forms.render(p, testValues, STATE.onChange));

    test.appendChild(th); test.appendChild(tb);
    editor.appendChild(test);
  }

  wrap.appendChild(editor);
  return wrap;
}
