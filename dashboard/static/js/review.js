// The paused-step review panel.
//
// When a run pauses after a `diagnostics: true` step, the Run tab swaps its log
// console for this panel: the figures that step just produced, the attempts
// that came before them, and a form holding *only that step's* parameters — or,
// for an Apply QC step (which the runner splits so it pauses test by test),
// only the paused test's parameters. The
// form is bound to the same values object the builder card uses, so an edit
// here lands in the builder and the YAML immediately -- when you finally hit
// Continue, the config already holds the values you settled on.
//
// Cycle: tweak a parameter -> Re-run step (the pipeline re-executes just that
// step from its pre-step snapshot) -> compare against the previous attempt ->
// repeat until happy -> Continue.

const Review = {
  active: false,
  index: null,     // step index in the running pipeline
  name: null,      // step name, used to match figures and guard the re-run
  test: null,      // QC test, when the runner split this step test by test
  key: null,       // Run.unitKey(index, test) — what figures are grouped under
  showLog: false,  // Log toggle: show the console instead of the panel
  busy: false,     // true between "Re-run" and the next pause
  selected: null,  // attempt shown in the main view; null = the latest one

  host() { return document.getElementById('step-review'); },

  // ---- lifecycle ----
  show(index, name, test) {
    Review.active = true;
    Review.index = index;
    Review.name = name;
    Review.test = test || null;
    Review.key = Run.unitKey(index, test);
    Review.busy = false;
    Review.showLog = false;
    Review.selected = null;
    Review.build();
    Review.apply();
  },

  hide() {
    Review.active = false;
    Review.busy = false;
    Review.host().innerHTML = '';
    Review.apply();
  },

  // Show/hide the panel against the console. The pause banner itself stays up
  // on screen for as long as the run is paused, on every tab, so "paused after
  // step N" reads the same whichever view is showing; only its "Review step"
  // button — the way back to the panel — hides once the panel is already on
  // screen, where it would be redundant.
  apply() {
    const onRunTab = document.querySelector('.tab.active')?.dataset.tab === 'run';
    const panelVisible = Review.active && !Review.showLog;
    Review.host().classList.toggle('hidden', !panelVisible);
    document.getElementById('log-wrap').classList.toggle('hidden', panelVisible);
    document.getElementById('run-note').classList.toggle('hidden', panelVisible);
    document.getElementById('run-pause').classList.toggle('hidden', !Review.active);
    document.getElementById('btn-review').classList.toggle('hidden', panelVisible && onRunTab);
    document.getElementById('btn-log').classList.toggle('hidden', !panelVisible && onRunTab);
  },

  // The builder step this pause refers to. Indices line up with the running
  // config; if the user has since reordered/added steps, fall back to the first
  // step of the right name so the form still shows something sensible.
  item() {
    const items = STATE.pipeline.items;
    const byIndex = items[Review.index];
    if (byIndex && byIndex.name === Review.name) return byIndex;
    return items.find((i) => i.name === Review.name) || byIndex || null;
  },

  // ---- attempt selection ----
  // Which attempt the main view is showing. Purely a viewing choice — it has no
  // bearing on what Continue does. Defaults to the newest.
  selectedGroup() {
    const attempts = Run.groupsFor(Review.key);
    if (!attempts.length) return null;
    return attempts.includes(Review.selected)
      ? Review.selected : attempts[attempts.length - 1];
  },

  select(group) {
    Review.selected = group;
    Review.renderPlots();
  },

  // A QC unit's parameters are `{qc_settings: {<test>: …}}`; everywhere the
  // panel talks about "the parameters" it means that test's settings.
  unwrap(params) {
    if (!Review.test || !params) return params;
    return (params.qc_settings || {})[Review.test] || null;
  },

  // The QC test's values object, on the same reference the builder card edits.
  testValues() {
    const item = Review.item();
    if (!item || !Review.test) return null;
    const settings = item.values.qc_settings;
    return settings ? settings[Review.test] || null : null;
  },

  // Put an attempt's parameters back into the config (builder + YAML + form),
  // so what you are looking at is what the pipeline would run.
  applyParams(rawParams) {
    const item = Review.item();
    const params = Review.unwrap(rawParams);
    if (!item || !params) return;
    if (Review.test) {
      const values = Review.testValues();
      if (!values) return;
      const def = STATE.qcByName[Review.test];
      for (const spec of (def && def.parameters) || []) {
        values[spec.name] = spec.name in params
          ? Forms.clone(params[spec.name]) : Forms.defaultValue(spec);
      }
    } else {
      for (const spec of item.def.parameters || []) {
        item.values[spec.name] = spec.name in params
          ? Forms.clone(params[spec.name]) : Forms.defaultValue(spec);
      }
    }
    STATE.onChange();
    renderPipeline();
  },

  // ---- rendering ----
  // Built once per pause; the plots and the header status refresh on their own
  // so re-running never re-renders (and un-focuses) the parameter form.
  build() {
    const host = Review.host();
    host.innerHTML = '';

    // The step/test name, "Log"/"Review step" and "Re-run step" all live in
    // the pause banner above (it's on screen for as long as this panel is),
    // so the panel itself is just the plots and where to edit their params.
    const plots = document.createElement('div');
    plots.className = 'review-plots'; plots.id = 'review-plots';

    // The parameters live in the builder, which has just unlocked this step
    // (and, for a QC step, opened the paused test) — one editing surface, not
    // two copies of the same form.
    const where = document.createElement('div');
    where.className = 'hint review-where';
    where.textContent = Review.test
      ? `Edit '${Review.test}' in the pipeline builder on the left — it is the ` +
        'only part of the config unlocked while the run is paused.'
      : 'Edit this step in the pipeline builder on the left — it is the only ' +
        'part of the config unlocked while the run is paused.';
    const jump = document.createElement('button');
    jump.className = 'ghost review-jump';
    jump.textContent = 'Show me';
    jump.title = 'Scroll the builder to this step';
    jump.onclick = () => RunLock.pauseAt(Review.index, Review.test);
    where.appendChild(jump);

    host.appendChild(plots);
    host.appendChild(where);
    Review.renderPlots();
  },

  // Latest attempt large, earlier attempts as a comparison strip underneath.
  renderPlots() {
    const host = document.getElementById('review-plots');
    if (!host) return;
    host.innerHTML = '';
    const attempts = Run.groupsFor(Review.key);
    const current = Review.selectedGroup();
    // No open group for this step means the last re-run drew nothing, so what
    // is on screen is the previous attempt's figure — say so rather than
    // presenting a stale plot as the new result.
    const isLatest = current && current === attempts[attempts.length - 1];
    const stale = !Review.busy && isLatest && attempts.length > 0 &&
      (!Run.activeGroup || Run.activeGroup.key !== Review.key);

    if (!current || !current.figs.length) {
      const hint = document.createElement('div');
      hint.className = 'hint review-empty';
      hint.textContent = Review.busy
        ? 'Re-running — the new figure will appear here.'
        : 'This step produced no figure. Re-run it with diagnostics on, or Continue.';
      host.appendChild(hint);
    } else {
      const no = Run.attemptNo(current);
      const main = document.createElement('div');
      main.className = 'review-main';
      const label = document.createElement('div');
      label.className = 'review-attempt-label';
      label.textContent = stale
        ? `Attempt ${no} — the last re-run produced no new figure`
        : (attempts.length > 1
          ? `Attempt ${no}${isLatest ? ' (latest — this is what Continue carries forward)' : ''}`
          : 'Result');
      main.appendChild(label);
      // The parameters this attempt actually ran with. Spelled out rather than
      // implied, so accepting one is never a guess about what it contained.
      main.appendChild(Review.paramSummary(current.params));
      const cards = document.createElement('div');
      cards.className = 'review-main-cards';
      current.figs.forEach((_, i) =>
        cards.appendChild(Viewer.card(current.figs, i, { cls: 'big' })));
      main.appendChild(cards);
      host.appendChild(main);
    }

    if (attempts.length > 1) {
      const strip = document.createElement('div');
      strip.className = 'review-strip';
      const title = document.createElement('div');
      title.className = 'review-strip-title';
      title.textContent = 'Attempts — click to compare';
      strip.appendChild(title);
      const row = document.createElement('div');
      row.className = 'review-strip-row';
      // Newest first: the most recent comparison is the one you usually want.
      for (let k = attempts.length - 1; k >= 0; k--) {
        const g = attempts[k];
        const cell = document.createElement('div');
        cell.className = 'review-thumb' + (g === current ? ' selected' : '');
        if (g.figs.length && g.figs[0].isLog) {
          const pre = document.createElement('pre');
          pre.className = 'log-card-text log-card-text-compact' +
            (g.figs[0].isError ? ' error-card-text' : '');
          pre.textContent = g.figs[0].text;
          cell.appendChild(pre);
        } else if (g.figs.length) {
          const img = document.createElement('img');
          img.src = Viewer.src(g.figs[0]);
          img.alt = `attempt ${Run.attemptNo(g)}`;
          img.loading = 'lazy';
          cell.appendChild(img);
        }
        const cap = document.createElement('div');
        cap.className = 'review-thumb-cap';
        cap.textContent = `Attempt ${Run.attemptNo(g)}` + (g === current ? ' ·  shown' : '');
        cell.appendChild(cap);
        cell.appendChild(Review.paramSummary(g.params, { compact: true }));
        // Loading an old attempt's values only fills the form in — running with
        // them is still an explicit Re-run, so nothing happens behind your back.
        if (g.params && g !== attempts[attempts.length - 1]) {
          const use = document.createElement('button');
          use.className = 'ghost review-use';
          use.textContent = 'Use these';
          use.title = 'Load these parameters into the form (does not re-run)';
          use.onclick = (e) => { e.stopPropagation(); Review.applyParams(g.params); };
          cell.appendChild(use);
        }
        cell.onclick = () => Review.select(g);
        cell.title = 'Show this attempt';
        row.appendChild(cell);
      }
      strip.appendChild(row);
      host.appendChild(strip);
    }
  },

  // An attempt's parameters as chips. Only those that differ from the step's
  // schema defaults, so the summary stays readable on steps with many knobs.
  paramSummary(rawParams, { compact = false } = {}) {
    const params = Review.unwrap(rawParams);
    const wrap = document.createElement('div');
    wrap.className = 'review-diff' + (compact ? '' : ' review-diff-main');
    if (!params) {
      wrap.classList.add('hint');
      wrap.textContent = 'parameters not recorded';
      return wrap;
    }
    const keys = Object.keys(params);
    if (!keys.length) {
      wrap.classList.add('hint');
      wrap.textContent = 'step defaults';
      return wrap;
    }
    for (const k of keys.slice(0, compact ? 3 : 8)) {
      const chip = document.createElement('span');
      chip.className = 'review-chip';
      chip.textContent = `${k}: ${Review.short(params[k])}`;
      wrap.appendChild(chip);
    }
    if (keys.length > (compact ? 3 : 8)) {
      const more = document.createElement('span');
      more.className = 'review-chip';
      more.textContent = `+${keys.length - (compact ? 3 : 8)} more`;
      wrap.appendChild(more);
    }
    return wrap;
  },

  // "what changed after this attempt" — the parameters that differ between an
  // attempt and the one that followed it, so the strip explains itself.
  diffChips(before, after) {
    if (!before || !after) return null;
    const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])]
      .filter((k) => !Forms.equal(before[k], after[k]));
    if (!keys.length) return null;
    const wrap = document.createElement('div');
    wrap.className = 'review-diff';
    for (const k of keys.slice(0, 4)) {
      const chip = document.createElement('span');
      chip.className = 'review-chip';
      chip.textContent = `${k}: ${Review.short(before[k])} → ${Review.short(after[k])}`;
      wrap.appendChild(chip);
    }
    return wrap;
  },

  short(v) {
    if (v === undefined) return '—';
    const s = typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v);
    return s.length > 18 ? s.slice(0, 17) + '…' : s;
  },

  // Pause-banner status + button state while a re-run is in flight.
  setBusy(busy, text) {
    Review.busy = busy;
    const status = document.getElementById('run-pause-status');
    if (status) status.textContent = text || '';
    const rerun = document.getElementById('btn-rerun');
    if (rerun) rerun.disabled = busy;
  },
};
