// Run the current config and stream the pipeline's logs via Server-Sent Events.

const Run = {
  source: null,
  progressEl: null, // the single <span> a progress bar redraws in place
  stopping: false,  // set once the user hits Stop, so the end event reads as "stopped"
  stopBtnMode: 'idle', // 'idle' | 'stop' | 'clear' — what btn-stop currently does
  // The step currently executing, from the __PELAGOS_STEP__ marker:
  // {index, name, test, key}. Figures are attributed by *index* (so a config
  // that repeats a step name still gets one group per occurrence) and, for a QC
  // step the runner split test by test, by the test name as well.
  currentStep: null,
  plotCount: 0,

  // Captured figures, grouped into attempts: one group per (unit, re-run), in
  // the order they were produced. Both the Plots tab gallery and the paused
  // step review panel render from this — nothing reads the DOM back.
  //   {index, key, step, test, figs: [{fname, caption, spec}], params}
  groups: [],
  activeGroup: null,  // group new figures land in; nulled by a re-run
  pendingParams: null, // parameters sent with the re-run that is in flight

  pausedStep: null, // index of the step the run is paused after, or null
  pausedName: null, // its step name, used to guard a re-run against edits
  pausedTest: null, // the QC test within it, when the step was split

  // Marker prefixes run_bootstrap.py prints on stdout:
  //   __PELAGOS_FIG__ <filename>\t<caption>          a saved diagnostic figure
  //   __PELAGOS_STEP__ <index>\t<step>[\t<qc test>]  about to execute
  //   __PELAGOS_PAUSE__ <index>\t<step>[\t<qc test>] paused, awaiting the user
  //   __PELAGOS_RERUN__ <index>                      re-running the paused unit
  //   __PELAGOS_MEM__ <rss>\t<peak>\t<data>\t<label> RSS after a step (MB)
  //   __PELAGOS_REPORT__ <abspath>\t<filename>          a PDF report was written
  FIG_MARKER: '__PELAGOS_FIG__ ',
  STEP_MARKER: '__PELAGOS_STEP__ ',
  PAUSE_MARKER: '__PELAGOS_PAUSE__ ',
  RERUN_MARKER: '__PELAGOS_RERUN__ ',
  MEM_MARKER: '__PELAGOS_MEM__ ',
  REPORT_MARKER: '__PELAGOS_REPORT__ ',

  report: null, // {path, name} of the PDF report the run produced, if any

  // ---- ANSI colour ----
  // The server forwards the pipeline's SGR colour codes (everything else is
  // stripped there), so the console shows the same colours as the terminal.
  ANSI_SGR: /\x1b\[([0-9;]*)m/g,
  ANSI_COLORS: {
    30: 'ansi-black', 31: 'ansi-red', 32: 'ansi-green', 33: 'ansi-yellow',
    34: 'ansi-blue', 35: 'ansi-magenta', 36: 'ansi-cyan', 37: 'ansi-white',
    90: 'ansi-bright-black', 91: 'ansi-bright-red', 92: 'ansi-bright-green',
    93: 'ansi-bright-yellow', 94: 'ansi-bright-blue', 95: 'ansi-bright-magenta',
    96: 'ansi-bright-cyan', 97: 'ansi-bright-white',
  },

  stripAnsi(text) {
    return text.replace(Run.ANSI_SGR, '');
  },

  // Turn SGR codes into styled spans. Text is HTML-escaped first, so log
  // output can never inject markup.
  ansiToHtml(text) {
    const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
    let out = '', fg = null, bold = false, dim = false, open = false, last = 0;
    const close = () => { if (open) { out += '</span>'; open = false; } };
    const openSpan = () => {
      const cls = [fg, bold ? 'ansi-bold' : null, dim ? 'ansi-dim' : null].filter(Boolean).join(' ');
      if (cls) { out += `<span class="${cls}">`; open = true; }
    };
    for (const m of text.matchAll(Run.ANSI_SGR)) {
      out += esc(text.slice(last, m.index));
      last = m.index + m[0].length;
      close();
      const parts = (m[1] || '0').split(';');
      for (let i = 0; i < parts.length; i++) {
        const n = Number(parts[i] || 0);
        if (n === 0) { fg = null; bold = false; dim = false; }
        else if (n === 1) bold = true;
        else if (n === 2) dim = true;
        else if (n === 22) { bold = false; dim = false; }
        else if (n === 39) fg = null;
        // 256-colour foreground (ESC[38;5;<n>m): only the SEVERE amber (202)
        // the pipeline emits is mapped, everything else is skipped.
        else if (n === 38 && parts[i + 1] === '5') {
          if (Number(parts[i + 2]) === 202) fg = 'ansi-amber';
          i += 2;
        }
        else if (Run.ANSI_COLORS[n]) fg = Run.ANSI_COLORS[n];
      }
      openSpan();
    }
    out += esc(text.slice(last));
    close();
    return out;
  },

  levelClass(line) {
    if (/ - ERROR - | ERROR:| Traceback/.test(line)) return 'lvl-error';
    if (/ - WARNING - | WARN/.test(line)) return 'lvl-warn';
    if (/ - SEVERE - /.test(line)) return 'lvl-severe';
    if (/STOP|Pipeline stopped/.test(line)) return 'lvl-stop';
    return '';
  },

  // A tqdm-style progress bar line, e.g. "Progress:  45%|████  | 713/1602 ...".
  looksLikeProgress(line) {
    return /\d+%\|/.test(line);
  },

  // Auto-scroll only while "stuck" to the tail. The user scrolling up detaches
  // it (so the log holds still to read); the jump-to-latest button re-attaches.
  stick: true,

  atBottom(c) {
    // A few px of slack so sub-pixel rounding still counts as "at the bottom".
    return c.scrollHeight - c.scrollTop - c.clientHeight < 8;
  },

  // Follow the tail if stuck; otherwise leave the view put and surface the
  // jump-to-latest button so the user can catch back up.
  autoScroll() {
    const c = document.getElementById('log-console');
    if (Run.stick) c.scrollTop = c.scrollHeight;
    document.getElementById('log-to-bottom').classList.toggle('hidden', Run.stick);
  },

  scrollToBottom() {
    const c = document.getElementById('log-console');
    Run.stick = true;
    c.scrollTop = c.scrollHeight;
    document.getElementById('log-to-bottom').classList.add('hidden');
  },

  // Wire the console's scroll + the jump-to-latest button once, at page load.
  initScroll() {
    const c = document.getElementById('log-console');
    c.addEventListener('scroll', () => {
      Run.stick = Run.atBottom(c);
      document.getElementById('log-to-bottom').classList.toggle('hidden', Run.stick);
    });
    document.getElementById('log-to-bottom')
      .addEventListener('click', () => Run.scrollToBottom());
  },

  append(line) {
    const c = document.getElementById('log-console');
    const span = document.createElement('span');
    // Fallback colouring for lines the pipeline sent uncoloured; any real ANSI
    // colour inside wins, being set on a descendant span.
    const cls = Run.levelClass(Run.stripAnsi(line));
    if (cls) span.className = cls;
    span.innerHTML = Run.ansiToHtml(line) + '\n';
    c.appendChild(span);
    Run.autoScroll();
  },

  // "<PREFIX><index>\t<name>[\t<qc test>]" -> [index, name, test|null].
  splitMarker(plain, prefix) {
    const parts = plain.slice(prefix.length).split('\t');
    return [
      parseInt(parts[0], 10),
      (parts[1] || '').trim(),
      parts.length > 2 ? parts[2].trim() : null,
    ];
  },

  // Identifies one pausable unit: a step, or one QC test within a split step.
  // Figure attribution, attempt numbering and the review panel all key off it.
  unitKey(index, test) {
    return test ? index + ' ' + test : String(index);
  },

  // The earliest marker in a line, or null. A marker does not always start its
  // line: a tqdm bar that closes with leave=False erases itself with a bare
  // "\r" and no newline, so the next print lands on the same line and arrives
  // as "…bar…__PELAGOS_STEP__ 3\tApply QC". Searching rather than testing the
  // prefix keeps such a marker from being swallowed as progress output.
  markerAt(plain) {
    let best = null;
    for (const marker of [Run.FIG_MARKER, Run.STEP_MARKER, Run.PAUSE_MARKER,
      Run.RERUN_MARKER, Run.MEM_MARKER, Run.REPORT_MARKER]) {
      const at = plain.indexOf(marker);
      if (at >= 0 && (best === null || at < best.at)) best = { marker, at };
    }
    return best;
  },

  // A committed (newline-terminated) line from the server.
  handleLine(line) {
    // Marker/step detection reads the uncoloured text, so a leading colour code
    // can't hide a marker; only the console rendering keeps the escapes.
    const plain = Run.stripAnsi(line);
    const hit = Run.markerAt(plain);
    if (hit) {
      // Anything before the marker is real console output that got glued on.
      if (hit.at > 0) Run.renderLine(plain.slice(0, hit.at));
      Run.handleMarker(hit.marker, plain.slice(hit.at));
      return;
    }
    Run.renderLine(line);
  },

  handleMarker(marker, plain) {
    if (marker === Run.MEM_MARKER) {
      // Payload is "<rss>\t<peak>\t<data>\t<label>" — the RAM meter's, not a
      // step marker. Kept out of the console: it's a visual, not a log line.
      Mem.add(plain.slice(marker.length));
      return;
    }
    if (marker === Run.REPORT_MARKER) {
      // Payload is "<abspath>\t<filename>" — not a step marker.
      const parts = plain.slice(marker.length).split('\t');
      const path = (parts[0] || '').trim();
      const name = (parts[1] || '').trim() || path;
      if (path) {
        Run.showReport(path, name);
        Run.append('  · report: ' + name + ' (open it in the Report tab)');
      }
      return;
    }
    const [idx, rest, test] = Run.splitMarker(plain, marker);
    if (marker === Run.FIG_MARKER) {
      // FIG's payload is "<filename>\t<caption>\t<spec>\t<reason>", not
      // "<index>\t<name>". <spec> names the interactive plot spec; when it is
      // empty the figure stays PNG-only and <reason> says what stopped it.
      const parts = plain.slice(marker.length).split('\t');
      const fname = (parts[0] || '').trim();
      const caption = (parts[1] || '').trim();
      const spec = (parts[2] || '').trim();
      const reason = (parts[3] || '').trim();
      Run.addPlot(fname, caption, spec);
      Run.append('  · plot: ' + (caption || fname) +
        (spec ? ' (zoomable)' : reason ? ` (image only — ${reason})` : ''));
    } else if (marker === Run.STEP_MARKER) {
      // Which step is executing, so its figures group under it. A garbled index
      // would make every figure its own group (NaN !== NaN), so ignore it.
      if (!Number.isInteger(idx)) return;
      Run.currentStep = { index: idx, name: rest, test, key: Run.unitKey(idx, test) };
      Run.activeGroup = null; // each execution of a step opens a fresh attempt
      RunLock.stepStarted(idx);
    } else if (marker === Run.PAUSE_MARKER) {
      if (!Number.isInteger(idx)) return;
      Run.showPause(idx, rest, test);
    } else if (marker === Run.RERUN_MARKER) {
      Run.setStatus('re-running step…', 'running');
    }
  },

  // Draw one non-marker line in the console.
  renderLine(line) {
    // The final bar frame arrives newline-terminated: finalise it in place
    // rather than appending a duplicate below the live progress line.
    if (Run.progressEl && Run.looksLikeProgress(Run.stripAnsi(line))) {
      Run.progressEl.innerHTML = Run.ansiToHtml(line) + '\n';
      Run.progressEl = null;
      Run.autoScroll();
      return;
    }
    Run.progressEl = null; // any active bar is now permanent as last drawn
    Run.append(line);
  },

  // A transient in-place redraw: update the one live progress span.
  handleProgress(line) {
    const c = document.getElementById('log-console');
    if (!Run.progressEl) {
      Run.progressEl = document.createElement('span');
      Run.progressEl.className = 'lvl-progress';
      c.appendChild(Run.progressEl);
    }
    Run.progressEl.innerHTML = Run.ansiToHtml(line) + '\n';
    Run.autoScroll();
  },

  // Record a captured figure against the current step/attempt.
  addPlot(fname, caption, spec) {
    if (!fname) return;
    const cur = Run.currentStep ||
      { index: -1, name: 'Diagnostics', test: null, key: Run.unitKey(-1, null) };
    if (!Run.activeGroup || Run.activeGroup.key !== cur.key) {
      Run.activeGroup = {
        index: cur.index,
        key: cur.key,
        step: cur.name,
        test: cur.test,
        figs: [],
        params: Run.pendingParams,
      };
      Run.pendingParams = null;
      Run.groups.push(Run.activeGroup);
    }
    Run.activeGroup.figs.push({
      fname, caption, spec: spec || null, url: Viewer.freshUrl(fname),
    });
    Run.plotCount += 1;
    Run.renderGallery();
    Run.updatePlotTab();
    if (Review.active && Review.key === cur.key) Review.renderPlots();
  },

  groupsFor(key) {
    return Run.groups.filter((g) => g.key === key);
  },

  // Attempt numbers are derived from position, never stored: a stored counter
  // drifts whenever groups are dropped (a duplicate re-run) or re-keyed, which
  // is how the gallery ended up showing a row of "attempt 1"s.
  attemptNo(group) {
    return Run.groupsFor(group.key).indexOf(group) + 1;
  },

  // Forget a figure group (a superseded attempt), keeping the counts honest.
  dropGroup(group) {
    const at = Run.groups.indexOf(group);
    if (at < 0) return;
    Run.groups.splice(at, 1);
    Run.plotCount -= group.figs.length;
    if (Run.activeGroup === group) Run.activeGroup = null;
    Run.renderGallery();
    Run.updatePlotTab();
  },

  // Continuing accepts one attempt: keep its figure in the run archive and drop
  // the discarded experiments, so the Plots tab stays a record of the run
  // rather than of every knob you turned.
  keepOnlyAttempt(key, group) {
    for (const g of Run.groupsFor(key)) {
      if (g !== group) Run.dropGroup(g);
    }
  },

  // Re-key figure groups that were captured before any step marker was seen
  // (index -1) onto a known unit.
  adoptOrphans(index, name, test) {
    const orphans = Run.groups.filter((g) => g.index === -1);
    if (!orphans.length) return;
    for (const g of orphans) {
      g.index = index; g.step = name; g.test = test;
      g.key = Run.unitKey(index, test);
    }
    Run.renderGallery();
  },

  // The Plots tab: the whole run's figures, in order, grouped by step and (for
  // a step re-run more than once) by attempt.
  renderGallery() {
    const gallery = document.getElementById('plots-gallery');
    gallery.innerHTML = '';
    document.getElementById('plots-empty').classList.toggle('hidden', !!Run.groups.length);
    for (const g of Run.groups) {
      const total = Run.groupsFor(g.key).length;
      const sec = document.createElement('section');
      sec.className = 'plot-step';
      const h = document.createElement('h4');
      h.className = 'plot-step-title';
      const title = g.test ? `${g.step} · ${g.test}` : g.step;
      h.textContent = total > 1 ? `${title} · attempt ${Run.attemptNo(g)}` : title;
      sec.appendChild(h);
      const cards = document.createElement('div');
      cards.className = 'plot-cards';
      g.figs.forEach((_, i) => cards.appendChild(Viewer.card(g.figs, i)));
      sec.appendChild(cards);
      gallery.appendChild(sec);
    }
  },

  updatePlotTab() {
    const tab = document.querySelector('.tab[data-tab="plots"]');
    if (tab) tab.textContent = Run.plotCount ? `Plots (${Run.plotCount})` : 'Plots';
  },

  clearPlots() {
    Run.groups = [];
    Run.activeGroup = null;
    Run.pendingParams = null;
    Run.plotCount = 0;
    Run.currentStep = null;
    // Spec filenames restart at fig_001.json each run, so a cached spec would
    // be the previous run's data under this run's name.
    Plot._cache = {};
    Run.renderGallery();
    Run.updatePlotTab();
    Run.clearReport();
  },

  // Show the PDF report a report step just wrote: a preview iframe plus an
  // "Open" link, and a dot on the Report tab so it's noticed on another tab.
  showReport(path, name) {
    Run.report = { path, name };
    const url = '/api/run/report?path=' + encodeURIComponent(path);
    const view = document.getElementById('report-view');
    view.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'report-head';
    const meta = document.createElement('div');
    meta.className = 'report-meta';
    meta.innerHTML = `<strong>${escapeHtml(name)}</strong>` +
      `<span class="report-path">${escapeHtml(path)}</span>`;
    const open = document.createElement('a');
    open.className = 'primary report-open';
    open.href = url;
    open.target = '_blank';
    open.rel = 'noopener';
    open.innerHTML = `${Icon.svg('external', 14)}Open PDF`;
    head.appendChild(meta);
    head.appendChild(open);
    const frame = document.createElement('iframe');
    frame.className = 'report-frame';
    frame.title = name;
    frame.src = url;
    view.appendChild(head);
    view.appendChild(frame);
    view.classList.remove('hidden');
    document.getElementById('report-empty').classList.add('hidden');
    const tab = document.querySelector('.tab[data-tab="report"]');
    if (tab) tab.textContent = 'Report •';
  },

  clearReport() {
    Run.report = null;
    const view = document.getElementById('report-view');
    if (view) { view.innerHTML = ''; view.classList.add('hidden'); }
    const empty = document.getElementById('report-empty');
    if (empty) empty.classList.remove('hidden');
    const tab = document.querySelector('.tab[data-tab="report"]');
    if (tab) tab.textContent = 'Report';
  },

  showPause(idx, name, test) {
    Run.pausedStep = idx;
    Run.pausedName = name;
    Run.pausedTest = test || null;
    const key = Run.unitKey(idx, test);
    // Attempt 1's parameters aren't known until the step has run: fill them in
    // from the config now, so the comparison strip can diff against them.
    // Safety net: figures captured while no step marker had been seen belong to
    // the step we have just paused after, since only it can have drawn them.
    Run.adoptOrphans(idx, name, test);
    const group = Run.groupsFor(key).slice(-1)[0];
    if (group && !group.params) group.params = Run.paramsAt(idx, test);
    // A re-run that drew nothing leaves these set; don't carry them into the
    // next step's first figure.
    Run.pendingParams = null;
    // A split QC step pauses per test, so the test is the headline and the
    // step it belongs to is the context.
    document.getElementById('run-pause-name').textContent = test || name;
    document.getElementById('run-pause-sub').textContent = test
      ? `Paused after ${name} — step ${idx + 1}`
      : `Paused after step ${idx + 1}`;
    const rerunBtn = document.getElementById('btn-rerun');
    rerunBtn.lastChild.textContent = test ? 'Re-run test' : 'Re-run step';
    rerunBtn.title = test
      ? 'Re-run just this QC test with the parameters in the builder'
      : 'Re-run just this step with the parameters in the builder';
    Run.setStatus('paused', 'running');
    Run.setRunButton('paused');
    // Unlock this step (or QC test) in the builder and scroll it into view —
    // that is where its parameters are edited.
    RunLock.pauseAt(idx, test);
    if (Review.active && Review.key === key) {
      Review.setBusy(false, '');   // a re-run finished: same panel, new plots
      Review.select(null);         // the new attempt becomes the selected one
      Review.apply();
    } else {
      Review.show(idx, name, test);
    }
  },

  hidePause() {
    Run.pausedStep = null;
    Run.pausedName = null;
    Run.pausedTest = null;
    // Still running, just no longer paused: the whole config locks again.
    if (RunLock.running) RunLock.pauseAt(null, null);
    Run.setRunButton(RunLock.running ? 'running' : 'idle');
    Review.hide();
  },

  // The parameters of step `idx` as they currently stand in the YAML pane (the
  // source of truth: it reflects both builder edits and hand edits). For a
  // split QC step, only the paused test's settings — that is all the unit runs.
  paramsAt(idx, test) {
    let steps;
    try {
      steps = (jsyaml.load(editor.getValue()) || {}).steps || [];
    } catch (e) {
      return null;
    }
    const step = steps[idx];
    if (!step) return null;
    const params = step.parameters || {};
    if (!test) return params;
    const settings = (params.qc_settings || {})[test];
    return settings === undefined ? null : { qc_settings: { [test]: settings } };
  },

  // Carry on from the pause. Whatever ran last is what the pipeline is holding,
  // so Continue simply accepts it — no re-running behind your back. To move on
  // with different values, edit them and Re-run first.
  async continueRun() {
    if (Run.pausedStep === null) return;
    const key = Run.unitKey(Run.pausedStep, Run.pausedTest);
    // Keep the accepted result in the run archive, drop the experiments.
    Run.keepOnlyAttempt(key, Run.groupsFor(key).slice(-1)[0]);
    Run.hidePause();
    Run.setStatus('running…', 'running');
    await API.continueRun();
  },

  // Re-run the paused unit with its current parameters. The step index must
  // still line up with the running pipeline, so refuse if the config was edited
  // in a way that moved this step somewhere else.
  async rerunStep() {
    if (Run.pausedStep === null) return;
    const idx = Run.pausedStep;
    const test = Run.pausedTest;
    let steps;
    try {
      steps = (jsyaml.load(editor.getValue()) || {}).steps || [];
    } catch (e) {
      alert('Cannot parse the YAML to re-run: ' + e.message);
      return;
    }
    const step = steps[idx];
    if (!step) {
      alert('Could not find step ' + (idx + 1) + ' in the current config.');
      return;
    }
    if (step.name !== Run.pausedName) {
      alert(`Step ${idx + 1} is now '${step.name}', not '${Run.pausedName}'. ` +
        'Undo the reordering, or Continue and start a fresh run.');
      return;
    }
    // A split QC unit runs one test, so it is re-run with that test alone.
    const params = Run.paramsAt(idx, test);
    if (!params) {
      alert(test
        ? `QC test '${test}' is no longer configured on step ${idx + 1}.`
        : `Could not read the parameters of step ${idx + 1}.`);
      return;
    }
    // Re-running unchanged parameters gives a figure identical to the one on
    // screen, so replace that attempt instead of stacking a duplicate next to it.
    const latest = Run.groupsFor(Run.unitKey(idx, test)).slice(-1)[0];
    if (latest && latest.params && Forms.equal(latest.params, params)) {
      Run.append('  · re-run with unchanged parameters — replacing attempt ' +
        Run.attemptNo(latest));
      Run.dropGroup(latest);
    }
    // Record exactly what leaves the browser: the log is then a full account of
    // what each attempt ran with, rather than something to be inferred.
    Run.append('  · re-run step ' + (idx + 1) + (test ? ` (${test})` : '') +
      ' with: ' + JSON.stringify(params));
    // Handed to the group the re-run's figures will land in, so the comparison
    // strip can say what changed between attempts.
    Run.pendingParams = params;
    Run.activeGroup = null; // next figure starts a new attempt
    Review.setBusy(true, 're-running…');
    Run.setRunButton('busy');
    await API.rerunStep(params);
  },

  setStatus(text, cls) {
    const el = document.getElementById('run-status');
    el.textContent = text;
    el.className = 'run-status' + (cls ? ' ' + cls : '');
  },

  // btn-run doubles as Continue while paused, so its label/action follow the
  // run state: 'idle' (nothing running), 'running', 'paused' (Continue,
  // enabled) or 'busy' (paused but a re-run is in flight — Continue disabled).
  setRunButton(mode) {
    const btn = document.getElementById('btn-run');
    if (mode === 'paused' || mode === 'busy') {
      btn.disabled = mode === 'busy';
      btn.innerHTML = Icon.svg('play') + 'Continue';
    } else {
      btn.disabled = mode === 'running';
      btn.innerHTML = Icon.svg('play') + 'Run pipeline';
    }
  },

  // btn-stop does double duty: "Stop" while a run is live, "Clear" once it has
  // finished/failed/been stopped (so the log/plots don't just sit there stale),
  // disabled with the "Stop" label before anything has run yet.
  setStopButton(mode) {
    const btn = document.getElementById('btn-stop');
    Run.stopBtnMode = mode;
    if (mode === 'clear') {
      btn.disabled = false;
      btn.innerHTML = Icon.svg('trash2') + 'Clear';
    } else {
      btn.disabled = mode === 'idle';
      btn.innerHTML = Icon.svg('stop') + 'Stop';
    }
  },

  async start(yamlContent) {
    Run.setRunButton('running');
    Run.setStopButton('stop');
    Run.setStatus('starting…', 'running');
    try {
      await API.run(yamlContent);
    } catch (e) {
      // The UI thought nothing was running but the server disagrees — the usual
      // cause is a dropped stream (sleep/suspend) that left the buttons stale.
      // Attach to the run that is actually there instead of stranding the user.
      if (/already running/i.test(e.message)) {
        Run.setStatus('re-attaching to the running pipeline…', 'running');
        Run.connect(true);
        return;
      }
      Run.setStatus('failed to start: ' + e.message, 'err');
      Run.setRunButton('idle');
      Run.setStopButton('idle');
      return;
    }
    Run.connect(true);
  },

  // Attach to the run's SSE stream. Shared by a fresh start and by reconnecting
  // to an already-running pipeline after a page refresh. The stream replays the
  // captured backlog first, so a reconnect repaints the whole log.
  connect(clearConsole) {
    if (clearConsole) {
      document.getElementById('log-console').textContent = '';
      Run.scrollToBottom(); // fresh log starts stuck to the tail
      Run.clearPlots();
      Mem.reset();
    }
    Run.hidePause();
    RunLock.begin(); // freeze the config for as long as the run owns it
    Run.setRunButton('running');
    Run.setStopButton('stop');
    Run.setStatus('running…', 'running');
    Run.progressEl = null;
    Run.stopping = false;
    if (Run.source) Run.source.close();
    Run.source = new EventSource('/api/run/stream');
    Run.source.onmessage = (ev) => Run.handleLine(ev.data);
    Run.source.addEventListener('progress', (ev) => Run.handleProgress(ev.data));
    Run.source.addEventListener('end', (ev) => {
      const code = Number(ev.data);
      if (Run.stopping) {
        Run.setStatus('stopped', 'err');
      } else {
        Run.setStatus(code === 0 ? 'finished' : `exited (code ${ev.data})`,
          code === 0 ? 'ok' : 'err');
      }
      Run.cleanup();
    });
    Run.source.onerror = () => Run.handleDrop();
  },

  // The stream died. Suspending the laptop or sleeping the tab kills the SSE
  // connection while the pipeline carries on (very often sitting paused), so a
  // drop must not be read as "the run is over" — that is what left the Run
  // button live, the pause panel gone, and Run answering "already running".
  // Re-attach instead; the stream replays the whole backlog, rebuilding the log,
  // the figures and the paused-step panel exactly as they were.
  handleDrop() {
    if (Run.source) { Run.source.close(); Run.source = null; }
    clearTimeout(Run._retry);
    if (Run.stopping) return; // Stop is in flight; the end event settles it
    Run.setStatus('reconnecting…', 'running');
    API.runStatus().then((s) => {
      if (s.running) {
        Run._retry = setTimeout(() => Run.connect(true), 1500);
      } else {
        Run.setStatus('run ended while disconnected', '');
        Run.cleanup();
      }
    }).catch(() => {
      // Server unreachable (still asleep?): keep trying rather than give up.
      Run._retry = setTimeout(() => Run.handleDrop(), 3000);
    });
  },

  // Re-attach if we have no stream but the server still has a run. Called when
  // the tab is shown again, which is the moment a suspended laptop comes back.
  async ensureConnected() {
    if (Run.source || Run.stopping) return;
    try {
      const s = await API.runStatus();
      if (s.running) Run.connect(true);
    } catch (e) { /* server not reachable yet; the next event will retry */ }
  },

  // On page load, reattach to a pipeline that's still running (e.g. after a
  // refresh) so the UI reflects it instead of offering a Run that 409s.
  async resumeIfRunning() {
    try {
      const s = await API.runStatus();
      if (s.running) {
        Run.showTab();
        Run.connect(true);
      }
    } catch (e) { /* server not ready; ignore */ }
  },

  // Switch the output panel to the Run tab (used when auto-resuming, and to get
  // back to the review panel from the pause banner on another tab).
  showTab() {
    document.querySelectorAll('.tab').forEach((t) =>
      t.classList.toggle('active', t.dataset.tab === 'run'));
    document.querySelectorAll('.tab-panel, .tab-actions').forEach((p) =>
      p.classList.toggle('hidden', p.dataset.panel !== 'run'));
    Run.onTabChange();
  },

  // The pause banner's "Review step" button is only needed when the review
  // panel isn't already on screen, so it has to be re-evaluated whenever the
  // visible tab changes.
  onTabChange() {
    if (Review.active) Review.apply();
  },

  // The run is genuinely over (finished, stopped, or gone). Only called on a
  // terminal state — never on a transient stream drop, see handleDrop.
  cleanup() {
    Run.progressEl = null;
    clearTimeout(Run._retry);
    Run.hidePause();
    RunLock.end();
    if (Run.source) { Run.source.close(); Run.source = null; }
    Run.setRunButton('idle');
    Run.setStopButton('clear');
  },

  async stop() {
    Run.stopping = true;
    Run.setStatus('stopping…', 'running');
    await API.stopRun();
    // Stopping discards the run, so its figures go with it — the next run
    // starts from a clean gallery either way.
    Run.clearPlots();
  },

  // Reset a finished/stopped run's log and status back to idle, ready for the
  // next Run. Only reachable once btn-stop is in its "Clear" state.
  clearRun() {
    document.getElementById('log-console').textContent = '';
    Run.clearPlots();
    Run.setStatus('', '');
    Run.setStopButton('idle');
  },
};
