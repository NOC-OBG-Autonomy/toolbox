// Bootstrap: load the registry, build the UI, wire controls together.

let editor = null;         // CodeMirror instance for the YAML pane
let syncingFromBuilder = false;
let errorLineHandle = null; // CodeMirror line handle currently marked red, if any
let activeStepId = null;    // builder step whose YAML lines are highlighted
let stepHighlightLines = null; // {start, end} of the current YAML highlight
let lastFocusedStep = null; // step index last expanded from the YAML cursor

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ---- YAML ⇄ builder cross-highlighting ----
// Line spans (0-based, inclusive) of each step under `steps:`, in order. Steps
// are the list items marked `- ` at the `steps:` child indentation; the array
// index lines up with STATE.pipeline.items.
function stepLineRanges() {
  const lines = editor.getValue().split('\n');
  let inSteps = false, stepsIndent = null;
  const starts = [];
  for (let i = 0; i < lines.length; i++) {
    if (!inSteps) { if (/^steps:\s*$/.test(lines[i])) inSteps = true; continue; }
    const m = lines[i].match(/^(\s*)-\s/);
    if (!m) continue;
    if (stepsIndent === null) stepsIndent = m[1].length;
    if (m[1].length === stepsIndent) starts.push(i);
  }
  return starts.map((start, k) => ({
    start,
    end: (k + 1 < starts.length ? starts[k + 1] - 1 : lines.length - 1),
  }));
}

function stepIndexAtLine(line) {
  const ranges = stepLineRanges();
  for (let i = 0; i < ranges.length; i++) {
    if (line >= ranges[i].start && line <= ranges[i].end) return i;
  }
  return null;
}

function clearStepHighlight() {
  if (editor && stepHighlightLines) {
    for (let ln = stepHighlightLines.start; ln <= stepHighlightLines.end && ln < editor.lineCount(); ln++) {
      editor.removeLineClass(ln, 'background', 'cm-step-highlight');
    }
  }
  stepHighlightLines = null;
}

// Highlight (and scroll to) the YAML lines for a builder step. Called when a
// step card is focused/edited; re-applied after every YAML regeneration since
// setValue drops line classes.
function highlightYamlForStep(id) {
  activeStepId = id;
  applyStepHighlight();
}

function applyStepHighlight() {
  if (!editor) return;
  clearStepHighlight();
  if (activeStepId == null) return;
  const idx = STATE.pipeline.items.findIndex((i) => i.id === activeStepId);
  if (idx < 0) return;
  const r = stepLineRanges()[idx];
  if (!r) return;
  for (let ln = r.start; ln <= r.end && ln < editor.lineCount(); ln++) {
    editor.addLineClass(ln, 'background', 'cm-step-highlight');
  }
  stepHighlightLines = r;
  editor.scrollIntoView({ from: { line: r.start, ch: 0 }, to: { line: r.end, ch: 0 } });
}

function refreshYAML() {
  if (!editor) return;
  Config.noteEdit(); // a builder change may fork a locked config
  syncingFromBuilder = true;
  editor.setValue(Config.toYAML());
  syncingFromBuilder = false;
  clearYamlError();
  applyStepHighlight(); // setValue drops line classes; re-mark the active step
  scheduleValidate();
  Inspect.schedule();
}

// Run schema validation against the server and show the result. Debounced so it
// runs while the user types rather than on a button press.
async function runValidate() {
  if (!editor) return;
  showValidating();
  const result = await API.validate(editor.getValue());
  renderValidation(result);
}
const scheduleValidate = debounce(runValidate, 400);

// Parse the YAML pane and push it into the builder. On a syntax error, mark the
// offending line red instead of overwriting the builder from stale state. The
// editor text is left untouched here so hand-typed formatting is preserved.
function syncYamlToBuilder() {
  if (!editor) return;
  clearYamlError();
  let cfg;
  try {
    cfg = jsyaml.load(editor.getValue());
  } catch (e) {
    markYamlError(e);
    return;
  }
  try {
    Config.fromObject(cfg || {}, Config.sectionsFromYAML(editor.getValue()));
    // fromObject builds fresh item objects, so a paused step's card is now a
    // different object: re-apply the lock so it is still the unlocked, expanded
    // one rather than a locked card like any other.
    if (typeof RunLock !== 'undefined' && RunLock.running && RunLock.index !== null) {
      RunLock.pauseAt(RunLock.index, RunLock.test);
    }
  } catch (e) { /* structurally odd but parseable: leave builder, Validate reports it */ }
}
const scheduleYamlToBuilder = debounce(syncYamlToBuilder, 400);

function clearYamlError() {
  if (editor && errorLineHandle != null) {
    editor.removeLineClass(errorLineHandle, 'background', 'cm-error-line');
    editor.removeLineClass(errorLineHandle, 'gutter', 'cm-error-line');
  }
  errorLineHandle = null;
}

function markYamlError(e) {
  clearYamlError();
  const line = e && e.mark && typeof e.mark.line === 'number' ? e.mark.line : null;
  if (line == null || line >= editor.lineCount()) return;
  errorLineHandle = editor.addLineClass(line, 'background', 'cm-error-line');
  editor.addLineClass(errorLineHandle, 'gutter', 'cm-error-line');
}

// Placeholder so the status row is never empty (e.g. before the first
// validation completes on page load). By default a no-op once a result is
// already showing, so ordinary typing doesn't flicker the pill every
// keystroke -- pass force:true (e.g. right after loading a different config)
// to replace whatever's showing immediately, so a slow validate call (the
// file-content check can take a moment) never leaves the *previous* config's
// result on screen looking like it belongs to the new one.
function showValidating(force = false) {
  const statusHost = document.getElementById('validation-status');
  if (!statusHost || (statusHost.firstChild && !force)) return;
  statusHost.innerHTML = '';
  statusHost.appendChild(statusBar('pending', 'Validating…', 'Checking the config against the step schemas'));
  if (force) document.getElementById('validation').innerHTML = '';
}

function renderValidation(result) {
  const host = document.getElementById('validation');
  // The status pill lives in the builder toolbar row; issue cards below it.
  const statusHost = document.getElementById('validation-status');
  host.innerHTML = '';
  statusHost.innerHTML = '';

  if (result.yaml_error) {
    const y = formatYamlError(result.yaml_error);
    statusHost.appendChild(statusBar('err', 'YAML syntax error', y.location || 'Check the YAML pane'));
    const card = document.createElement('div');
    card.className = 'v-issue v-yaml';
    let body = `<div class="v-msg">${escapeHtml(y.message)}</div>`;
    if (y.snippet) body += `<pre class="v-snippet">${escapeHtml(y.snippet)}</pre>`;
    card.innerHTML = `<div class="v-issue-body">${body}</div>`;
    host.appendChild(card);
    return;
  }

  if (result.ok) {
    statusHost.appendChild(statusBar('ok', 'Valid', "Matches every step's parameter schema"));
    return;
  }

  const n = result.issues.length;
  statusHost.appendChild(statusBar('err', `${n} issue${n === 1 ? '' : 's'}`, 'Fix before running'));
  // No data to work with is the most fundamental thing that can be wrong with
  // a config, so surface it above any other issue rather than in list order.
  const ranked = [...result.issues].sort((a, b) =>
    (parseIssue(b.error).critical ? 1 : 0) - (parseIssue(a.error).critical ? 1 : 0));
  for (const issue of ranked) host.appendChild(issueCard(issue));
}

// The green/red header pill at the top of the validation panel.
function statusBar(kind, title, sub) {
  const bar = document.createElement('div');
  bar.className = `v-status v-${kind}`;
  const icon = kind === 'ok' ? 'check' : kind === 'pending' ? 'rerun' : 'alert';
  bar.innerHTML = `<span class="v-icon">${Icon.svg(icon, 14)}</span>` +
    `<div class="v-text"><strong>${escapeHtml(title)}</strong>` +
    `<span>${escapeHtml(sub)}</span></div>`;
  return bar;
}

// One schema issue, rendered as a card. Clicking it locates the offending
// step in the YAML pane (highlight + scroll) via the existing machinery.
function issueCard(issue) {
  const parsed = parseIssue(issue.error);
  const card = document.createElement('div');
  card.className = 'v-issue' + (parsed.critical ? ' v-issue-critical' : '');
  const where = issue.index == null
    ? 'Pipeline'
    : `Step ${issue.index + 1}${issue.name ? ' · ' + escapeHtml(issue.name) : ''}`;
  const icon = parsed.critical ? `<span class="v-issue-icon">${Icon.svg('alert', 14)}</span>` : '';
  card.innerHTML =
    `<div class="v-issue-head">${icon}` +
    `<span class="v-tag v-tag-${parsed.kind}">${escapeHtml(parsed.tag)}</span>` +
    `<span class="v-where">${where}</span></div>` +
    `<div class="v-issue-body">${parsed.html}</div>`;

  const item = issue.index == null ? null : STATE.pipeline.items[issue.index];
  if (item) {
    card.classList.add('v-clickable');
    card.title = 'Show in YAML and the builder';
    card.onclick = () => {
      highlightYamlForStep(item.id);
      focusStepInBuilder(issue.index);
    };
  }
  return card;
}

// Turn a parameter_spec ValueError string into a {kind, tag, html} object. The
// message shapes come from utils/parameter_spec.py resolve(); the "[label] "
// prefix is dropped since the step name is already shown as the location.
function parseIssue(raw) {
  const msg = String(raw).replace(/^\[[^\]]*\]\s*/, '');

  // Loading the base data is foundational -- these three get their own,
  // more prominent card rather than the generic "Error" fallback below.
  let m = msg.match(/^Multiple data-loading steps found:\s*(.+?)\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'Multiple data sources', critical: true,
      html: `<div class="v-msg">Only one step should load or generate the ` +
        `pipeline's base data.</div><div class="v-detail v-muted">${escapeHtml(m[1])}</div>` };
  }
  m = msg.match(/^'Load OG1' has no 'file_path' set[^.]*\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'No file set', critical: true,
      html: `<div class="v-msg">This config does not include a data file.</div>` +
        `<div class="v-detail">${escapeHtml(m[1])}</div>` };
  }
  m = msg.match(/^Missing variables for(?: QC test)? '([^']+)':\s*([^.]+)\.\s*(?:[^.]*\.\s*)*No data-loading step[^.]*\.\s*(.*)$/s);
  if (m) {
    return { kind: 'load', tag: 'No data source', critical: true,
      html: `<div class="v-msg"><span class="v-param">${escapeHtml(m[1])}</span> needs ` +
        `${chips(splitNames(m[2]), 'bad')}, but nothing in the pipeline loads data.</div>` +
        `<div class="v-detail">${escapeHtml(m[3])}</div>` };
  }

  m = msg.match(/^invalid parameter value\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'value', tag: 'Invalid value',
      html: m[1].split(';').map(formatValueSegment).join('') };
  }
  m = msg.match(/^invalid parameter type\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'type', tag: 'Wrong type',
      html: m[1].split(';').map(formatTypeSegment).join('') };
  }
  m = msg.match(/^missing required parameter\(s\):\s*(.+)$/s);
  if (m) {
    return { kind: 'missing', tag: 'Missing',
      html: `<div class="v-detail">Add ${chips(splitNames(m[1]))}</div>` };
  }
  m = msg.match(/^unknown parameter\(s\):\s*([^.]+)\.\s*Valid parameters:\s*(.+?)\.?$/s);
  if (m) {
    return { kind: 'unknown', tag: 'Unknown',
      html: `<div class="v-detail">${chips(splitNames(m[1]), 'bad')} not recognised.</div>` +
        `<div class="v-detail v-muted">Valid: ${chips(splitNames(m[2]))}</div>` };
  }
  // Fallback: show the message verbatim (minus the [label] prefix).
  return { kind: 'other', tag: 'Error', html: `<div class="v-msg">${escapeHtml(msg)}</div>` };
}

// "to_derive (expected one of ['A','B'], got ['X'])" → param + allowed chips + bad value.
function formatValueSegment(seg) {
  const m = seg.match(/^\s*(\S+)\s*\(expected one of\s*(.+?),\s*got\s*(.+?)\)\s*$/s);
  if (!m) return `<div class="v-detail">${escapeHtml(seg.trim())}</div>`;
  const options = pyTokens(m[2]);
  const got = pyTokens(m[3]);
  return `<div class="v-detail"><span class="v-param">${escapeHtml(m[1])}</span>: ` +
    `${chips(got, 'bad')} not allowed.</div>` +
    `<div class="v-detail v-muted">Allowed: ${chips(options)}</div>`;
}

// "temperature (expected float, got str value 'x')" → param + expected/got.
function formatTypeSegment(seg) {
  const m = seg.match(/^\s*(\S+)\s*\(expected\s*(.+?),\s*got\s*(.+?)\)\s*$/s);
  if (!m) return `<div class="v-detail">${escapeHtml(seg.trim())}</div>`;
  return `<div class="v-detail"><span class="v-param">${escapeHtml(m[1])}</span>: ` +
    `expected <span class="v-good">${escapeHtml(m[2])}</span>, ` +
    `got <span class="v-got">${escapeHtml(m[3])}</span></div>`;
}

// Render a list of values as inline chips ("bad" = red styling).
function chips(items, cls = '') {
  if (!items.length) return '<span class="v-muted">(none)</span>';
  return `<span class="v-chips">` +
    items.map((t) => `<span class="v-chip ${cls === 'bad' ? 'v-chip-bad' : ''}">${escapeHtml(t)}</span>`).join('') +
    `</span>`;
}

// Split a comma-separated name list ("a, b, c") into trimmed names.
function splitNames(s) {
  return s.split(',').map((x) => x.trim()).filter(Boolean);
}

// Pull the values out of a Python-style literal like "['A', 'B']" or "'x'".
// Prefers quoted tokens; falls back to a bare scalar with brackets stripped.
function pyTokens(s) {
  const quoted = [...s.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => m[1] ?? m[2]);
  if (quoted.length) return quoted;
  return [s.replace(/^[\[\(]|[\]\)]$/g, '').trim()];
}

// Turn PyYAML's multi-block error dump into {message, location, snippet}.
// PyYAML emits a "problem" (expected X, but found Y), a "line N, column M"
// mark, and a caret-pointed context snippet — we surface a friendly version.
function formatYamlError(raw) {
  const text = String(raw);
  const marks = [...text.matchAll(/line (\d+), column (\d+)/g)];
  const last = marks.length ? marks[marks.length - 1] : null;
  const location = last ? `Line ${last[1]}, column ${last[2]}` : '';

  // Grab the code line sitting just above the last caret ("^") line.
  const lines = text.split('\n');
  let snippet = '';
  for (let i = lines.length - 1; i > 0; i--) {
    if (/^\s*\^\s*$/.test(lines[i])) {
      snippet = `${lines[i - 1].trim()}\n${'^'.padStart(caretCol(lines[i], lines[i - 1]))}`;
      break;
    }
  }

  const problem = (text.match(/expected (.+?), but found (.+?)(?:\n|$)/) || [])[0];
  return { message: friendlyYaml(text, problem), location, snippet };
}

// Re-align the caret to the trimmed snippet line (PyYAML indents both).
function caretCol(caretLine, codeLine) {
  const lead = codeLine.length - codeLine.trimStart().length;
  const col = caretLine.indexOf('^');
  return Math.max(1, col - lead + 1);
}

// Map common PyYAML problems to a plain-English explanation + fix hint.
function friendlyYaml(text, problem) {
  if (/expected <block end>, but found/.test(text))
    return 'Unexpected extra content — a value continues where a list or block should have ended. Look for a stray character, or a missing comma or closing bracket.';
  if (/could not find expected ':'/.test(text))
    return "Missing colon — a key needs a ':' after it before its value.";
  if (/mapping values are not allowed here/.test(text))
    return "Unexpected ':' — check the indentation, or wrap the value in quotes if it genuinely contains a colon.";
  if (/found unexpected end of (stream|document)/.test(text))
    return 'The document ended early — a bracket, brace or quote is probably left unclosed.';
  if (/found character '\\t'|found a tab character/.test(text))
    return 'Tab character in indentation — YAML must be indented with spaces, not tabs.';
  if (/found duplicate key/.test(text))
    return 'Duplicate key — the same key is defined twice in this mapping.';
  if (problem) return problem.replace(/\n.*/s, '').trim();
  return (text.split('\n')[0] || 'Could not parse the YAML.').trim();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

async function boot() {
  Icon.hydrate(); // fill [data-icon] placeholders in the static HTML
  STATE.registry = await API.registry();
  for (const s of STATE.registry.steps) STATE.stepsByName[s.name] = s;
  for (const q of STATE.registry.qc) STATE.qcByName[q.name] = q;
  STATE.onChange = refreshYAML;

  // CodeMirror YAML editor. Theme is remembered per browser; all themes are
  // defined locally in style.css (no CDN theme files).
  let savedTheme = 'vscode-dark';
  try { savedTheme = localStorage.getItem('yamlTheme') || savedTheme; } catch (e) { /* private mode */ }
  editor = CodeMirror.fromTextArea(document.getElementById('yaml-editor'), {
    mode: 'yaml',
    theme: savedTheme,
    lineNumbers: true,
    lineWrapping: true,
  });

  const themeSel = document.getElementById('theme-select');
  themeSel.value = savedTheme;
  themeSel.addEventListener('change', () => {
    editor.setOption('theme', themeSel.value);
    try { localStorage.setItem('yamlTheme', themeSel.value); } catch (e) { /* ignore */ }
  });

  // Live sync + auto-validate: typing in the YAML pane pushes into the builder
  // and re-validates on a debounce, so neither has to be triggered by a button.
  editor.on('change', () => {
    if (syncingFromBuilder) return;
    Config.noteEdit(); // hand-edited YAML forks a locked config too
    scheduleYamlToBuilder();
    scheduleValidate();
    Inspect.schedule();
  });

  // YAML → builder focus: as the cursor moves through the YAML, expand and
  // scroll to the matching step card in the builder. Working in the YAML pane
  // means the builder is a read-out, so drop any builder→YAML highlight.
  editor.on('focus', () => { activeStepId = null; clearStepHighlight(); });
  editor.on('cursorActivity', () => {
    if (syncingFromBuilder) return;
    const idx = stepIndexAtLine(editor.getCursor().line);
    if (idx == null || idx === lastFocusedStep) return;
    lastFocusedStep = idx;
    focusStepInBuilder(idx);
  });

  showValidating();
  renderPalette();
  renderSettings();
  renderPipeline();
  initBuilderDnD();
  Config.loading = true;
  refreshYAML();
  Config.loading = false;
  await Config.refreshList();

  // Auto-load default.yaml on startup if present, so the dashboard opens on a
  // ready-made config rather than an empty pipeline. If a run is already in
  // flight (a page refresh mid-run), load what that run is executing instead —
  // otherwise the builder would show a different config from the one the
  // pause markers are indexing into.
  try {
    let opening = Config.known.includes('default.yaml') ? 'default.yaml' : null;
    const running = await API.runStatus().then((s) => s.running).catch(() => false);
    if (running && Config.known.includes('_last_run.yaml')) opening = '_last_run.yaml';
    if (opening) await Config.load(opening);
  } catch (e) { /* no default; start empty */ }

  // palette search
  document.getElementById('palette-search').addEventListener('input', (e) =>
    renderPalette(e.target.value));

  // tabs
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const which = tab.dataset.tab;
      document.querySelectorAll('.tab-panel, .tab-actions').forEach((p) =>
        p.classList.toggle('hidden', p.dataset.panel !== which));
      if (which === 'yaml') setTimeout(() => editor.refresh(), 0);
      Run.onTabChange();
    });
  });

  // builder controls
  document.getElementById('btn-clear').addEventListener('click', () => {
    if (!confirm('Clear all steps?')) return;
    STATE.pipeline.nodes = [];
    renderPipeline();
    refreshYAML();
  });
  document.getElementById('btn-add-section').addEventListener('click', addSection);

  // YAML pane controls (Validate + YAML→builder are automatic now)
  document.getElementById('btn-copy').addEventListener('click', () => {
    navigator.clipboard.writeText(editor.getValue());
  });

  // config picker: selecting a config loads it (Config.load does the work).
  const picker = document.getElementById('config-select');
  picker.querySelector('.cfg-trigger').addEventListener('click', () => {
    if (picker.classList.contains('open')) Config.closePicker();
    else Config.openPicker();
  });
  document.addEventListener('click', (e) => {
    if (!picker.contains(e.target)) Config.closePicker();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') Config.closePicker();
  });

  // config persistence
  document.getElementById('btn-reveal').addEventListener('click', async () => {
    try {
      const { path } = await API.revealConfigs();
      Config.notice('Opened ' + path);
    } catch (e) {
      Config.notice('Could not open the configs folder: ' + e.message);
    }
    await Config.refreshList();
  });
  document.getElementById('config-name').addEventListener('input', () => Config.updateSaveLabel());
  document.getElementById('btn-save').addEventListener('click', async () => {
    const input = document.getElementById('config-name');
    let name = input.value.trim();
    if (!name) { alert('Enter a config name to save.'); return; }
    if (Config.isLocked(name)) {
      name = Config.nextCustomName();
      input.value = name;
      Config.notice(`That name is locked — saved as ${name}.yaml instead.`);
    }
    try {
      await API.saveConfig(name, editor.getValue());
    } catch (e) {
      alert(e.message);
      return;
    }
    await Config.refreshList(Config.withExt(name));
    Config.setCurrent(name);
  });
  document.getElementById('btn-delete').addEventListener('click', async () => {
    const name = Config.selected;
    if (!name || Config.isLocked(name)) return;
    if (!confirm('Delete ' + name + '?')) return;
    try {
      await API.deleteConfig(name);
    } catch (e) {
      alert(e.message);
      return;
    }
    if (Config.current === name) Config.current = null;
    await Config.refreshList();
  });

  // run controls
  Run.initScroll();
  // Doubles as Continue while paused — see Run.setRunButton().
  document.getElementById('btn-run').addEventListener('click', () =>
    Run.pausedStep !== null ? Run.continueRun() : Run.start(editor.getValue()));
  document.getElementById('btn-stop').addEventListener('click', () =>
    Run.stopBtnMode === 'clear' ? Run.clearRun() : Run.stop());
  // From another tab: jump back to the paused step's review panel.
  document.getElementById('btn-review').addEventListener('click', () => {
    Review.showLog = false;
    Run.showTab();
  });
  // From the review panel (or another tab): swap to the run log.
  document.getElementById('btn-log').addEventListener('click', () => {
    Review.showLog = true;
    Run.showTab();
  });
  document.getElementById('btn-rerun').addEventListener('click', () => Run.rerunStep());

  // If a pipeline is already running (e.g. the page was refreshed mid-run),
  // reattach to its log stream instead of showing a Run button that would 409.
  Run.resumeIfRunning();

  // Suspending the laptop or backgrounding the tab kills the SSE stream while
  // the pipeline keeps going, so re-check whenever the page comes back.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) Run.ensureConnected();
  });
  window.addEventListener('online', () => Run.ensureConnected());
}

boot().catch((e) => {
  document.body.innerHTML =
    `<pre style="padding:20px;color:#dc2626">Failed to start dashboard:\n${e.stack || e}</pre>`;
});
