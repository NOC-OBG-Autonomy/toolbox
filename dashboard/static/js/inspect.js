// Inspect tab: reads the 'Load OG1' step's file_path out of the current YAML
// and shows the file's variables, sensors and global attributes. Re-fetches
// whenever the YAML changes and the resolved file_path is different.

const Inspect = {
  lastPath: null,
  data: null,

  // First 'Load OG1' step's file_path, straight out of the YAML text (not
  // the builder state) so this works while the builder is mid-resync too.
  extractFilePath(yamlText) {
    let cfg;
    try { cfg = jsyaml.load(yamlText); } catch (e) { return null; }
    const steps = (cfg && cfg.steps) || [];
    const step = steps.find((s) => s && String(s.name || '').toLowerCase() === 'load og1');
    const fp = step && step.parameters && step.parameters.file_path;
    return fp ? String(fp).trim() : null;
  },

  refresh(yamlText) {
    const filePath = Inspect.extractFilePath(yamlText);
    if (!filePath) {
      Inspect.lastPath = null;
      Inspect.data = null;
      renderInspectEmpty();
      return;
    }
    if (filePath === Inspect.lastPath) return; // same file, nothing to refetch
    Inspect.lastPath = filePath;
    renderInspectLoading();
    fetch('/api/inspect?file_path=' + encodeURIComponent(filePath))
      .then(async (r) => {
        if (!r.ok) {
          const detail = (await r.json().catch(() => ({}))).detail || 'Could not read file.';
          throw new Error(detail);
        }
        return r.json();
      })
      .then((data) => {
        if (filePath !== Inspect.lastPath) return; // superseded by a later edit
        Inspect.data = data;
        renderInspect(data);
      })
      .catch((e) => {
        if (filePath !== Inspect.lastPath) return;
        Inspect.data = null;
        renderInspectError(e.message);
      });
  },
};

let inspectDebounceT = null;
Inspect.schedule = function () {
  clearTimeout(inspectDebounceT);
  inspectDebounceT = setTimeout(() => {
    if (typeof editor !== 'undefined' && editor) Inspect.refresh(editor.getValue());
  }, 400);
};

function inspectShowEmptyState(html) {
  document.getElementById('inspect-body').classList.add('hidden');
  document.getElementById('inspect-path').textContent = '';
  const empty = document.getElementById('inspect-empty');
  empty.innerHTML = html;
  empty.classList.remove('hidden');
}

function renderInspectEmpty() {
  inspectShowEmptyState(`Add a file to '<strong>Load OG1</strong>'.`);
}

function renderInspectLoading() {
  inspectShowEmptyState('Reading file…');
}

function renderInspectError(message) {
  inspectShowEmptyState(`<strong>Could not read file.</strong><br>${escapeHtml(message || '')}`);
}

function inspectRow(name, tag, desc) {
  const row = document.createElement('div');
  row.className = 'inspect-row';
  row.innerHTML =
    `<div class="inspect-row-head">` +
    `<span class="inspect-row-name">${escapeHtml(name)}</span>` +
    (tag ? `<span class="inspect-row-tag">${escapeHtml(tag)}</span>` : '') +
    `</div>` +
    (desc ? `<div class="inspect-row-desc">${escapeHtml(desc)}</div>` : '');
  row.dataset.search = `${name} ${tag || ''} ${desc || ''}`.toLowerCase();
  return row;
}

function inspectFillSection(section, rows, emptyText) {
  const host = document.getElementById(`inspect-rows-${section}`);
  host.innerHTML = '';
  rows.forEach((row) => host.appendChild(row));
  document.getElementById(`inspect-count-${section}`).textContent = rows.length ? `(${rows.length})` : '';
  if (!rows.length) {
    const none = document.createElement('div');
    none.className = 'inspect-none';
    none.textContent = emptyText;
    host.appendChild(none);
  }
}

function renderInspect(data) {
  document.getElementById('inspect-empty').classList.add('hidden');
  document.getElementById('inspect-body').classList.remove('hidden');
  document.getElementById('inspect-path').textContent = data.path;

  inspectFillSection(
    'parameters',
    (data.variables || []).map((v) => inspectRow(v.name, v.units, v.description)),
    'No variables found.'
  );
  inspectFillSection(
    'sensors',
    (data.sensors || []).map((s) => inspectRow(s, '', '')),
    'No sensors listed.'
  );
  inspectFillSection(
    'attributes',
    Object.entries(data.global_attributes || {}).map(([k, v]) => inspectRow(k, '', v)),
    'No attributes found.'
  );

  applyInspectFilter();
}

function applyInspectFilter() {
  const q = (document.getElementById('inspect-search').value || '').trim().toLowerCase();
  document.querySelectorAll('.inspect-section').forEach((section) => {
    let shown = 0;
    section.querySelectorAll('.inspect-row').forEach((row) => {
      const match = !q || row.dataset.search.includes(q);
      row.classList.toggle('hidden', !match);
      if (match) shown++;
    });
    const none = section.querySelector('.inspect-none');
    if (none) none.classList.toggle('hidden', !!q); // "no matches" beats a static empty note
    let noMatch = section.querySelector('.inspect-no-match');
    const hasRows = section.querySelectorAll('.inspect-row').length > 0;
    if (q && hasRows && shown === 0) {
      if (!noMatch) {
        noMatch = document.createElement('div');
        noMatch.className = 'inspect-none inspect-no-match';
        noMatch.textContent = 'No matches.';
        section.querySelector('.inspect-rows').appendChild(noMatch);
      }
    } else if (noMatch) {
      noMatch.remove();
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const search = document.getElementById('inspect-search');
  if (search) search.addEventListener('input', applyInspectFilter);
});
