# pelagos_py dashboard

A standalone web tool for **authoring, validating, and running** pelagos_py
pipeline configs. It is completely independent of the pipeline: it imports
`pelagos_py` only to *introspect* it. Nothing here is required to run the
pipeline operationally — a config authored in the dashboard is an ordinary YAML
file you can run any other way.

## What makes it "smart"

- **Auto-discovers steps.** The palette and every parameter form are generated
  from the live `STEP_CLASSES` / `QC_CLASSES` registries and each step's
  `describe_parameters()`. Add a new `@register_step` and it appears on the
  next server start — no dashboard changes.
- **Real validation.** `Validate` calls the pipeline's own
  `parameter_spec.resolve()` on the server, so what the dashboard accepts is
  exactly what the pipeline accepts (types, required params, `options`,
  unknown-key rejection). No duplicated validation logic to drift.
- **QC-aware.** Adding an *Apply QC* step gives you a picker of every registered
  QC test, each configured via its own schema.

## Running

```bash
pip install -r dashboard/requirements.txt   # fastapi, uvicorn, pyyaml
# (pelagos_py itself must be importable — installed, or run from the repo root)
python dashboard/app.py
```

Then open <http://localhost:8791>.

Steps can be grouped into **sections** (Pipeline → *Section*): a named,
contiguous run of steps that can be renamed, collapsed, dragged around as a
block, and dropped into. Sections are written to the YAML as `# ==== TITLE ====`
banner comments and read back from them, so hand-written configs like
`examples/configs/example_config_nelson.yaml` open with their sections intact.
The pipeline itself never sees them.

Configs you save live in `dashboard/configs/`. The pipeline runs as a
subprocess from the repo root, so relative paths in a config (e.g.
`examples/data/...`) resolve as they normally would.

## Tuning a step while it runs

**A running pipeline owns the config.** While it runs, the builder, the pipeline
settings and the YAML pane are all locked — the run is executing the YAML as it
was submitted, so an edit would leave the screen disagreeing with what is
actually running.

A step with `diagnostics: true` **pauses the run** once it has drawn its plot.
The Run tab then shows a review panel for that step alone: its figure (click for
a full-window viewer — arrow keys page through, click to zoom). That step's card
in the builder — and only that card — unlocks, expands and scrolls into view, so:

1. look at the plot,
2. adjust a parameter on the highlighted builder card,
3. **Re-run step** — the pipeline re-executes just that step from its pre-step
   snapshot and re-pauses, keeping the previous attempt as a thumbnail so you
   can compare,
4. repeat until it looks right, then **Continue** (which re-locks the config).

Reordering and removing steps stay locked even on the unlocked card: the runner
is working from the step indices it started with.

The parameters you settled on are already in the config, so saving keeps them.
Every figure of the run is also archived in the **Plots** tab, grouped by step
and attempt.

**Apply QC steps pause one test at a time.** A QC step runs every test it is
given in a single call, so pausing after the step would mean every plot at once
and a form covering every test. Instead the runner splits such a step into one
execution per test (exactly what the config could spell out by hand), and each
test gets its own pause, its own plot, and unlocks only that test's section of
the QC editor. Re-run replays that one test from the state just before it — the
tests already applied in the same step keep their flags. Splitting only happens
when the step would pause anyway, so an unattended run is unaffected.

## Interactive plots

Diagnostic figures are matplotlib, captured as PNGs — which cannot be zoomed
into. So alongside each PNG the runner also tries to write a **plot spec**
(`fig_spec.py`): the figure's actual x/y arrays plus its labels, limits and
legend. The viewer redraws that with plotly, giving box-zoom, pan, scroll-zoom,
hover readout and legend toggling on the real data. Panels a step drew with
`sharex=True` keep their x-ranges linked. **Image** in the viewer toolbar
switches back to the PNG at any point.

Nothing in `pelagos_py` changes: steps still just draw with matplotlib and call
`plt.show()`, and a run outside the dashboard never touches any of this.

**No step needs changing.** Steps keep drawing exactly as they do; whether a
figure becomes interactive depends only on what it is made of. Serialising is
all-or-nothing per figure: `fig_spec.py` understands line and marker plots
(including time-series drawn straight from `datetime64`), scatter points, and
`axhline`/`axvline` reference lines (range bounds, min/max limits). A figure
containing anything else — a histogram, `fill_between`, a `pcolormesh`, a map
projection — gets **no** spec and stays PNG-only, so a plot is never shown
half-drawn or subtly wrong.

The run log says which is which, per plot, and names what stopped it:

```
  · plot: TEMP Spike Test (zoomable)
  · plot: Profile summary (image only — patches)
  · plot: Track map (image only — projection:mercator)
```

Thumbnails of interactive figures also carry a blue rule. To make a PNG-only
plot interactive you either change the *step* to draw it with lines/scatter, or
teach `fig_spec.py` the artist named in the log — add a branch to
`_scatter_trace`/`_line_trace` and drop it from `_unsupported`.

Very large traces are thinned to a whole-figure budget of ~150k points before
being sent, using min/max-per-bucket decimation so single-sample spikes survive
(these are spike and stuck-value diagnostics — striding would hide exactly what
they exist to show). The viewer says so when it has thinned, and the PNG beside
it is always full resolution.

`static/vendor/plotly.min.js` is ~4.5 MB and is lazy-loaded the first time an
interactive plot is opened, so it costs nothing on page load.

## Layout

| File | Role |
|------|------|
| `app.py` | FastAPI backend: `/api/registry`, `/api/validate`, config CRUD, run + SSE log stream |
| `static/index.html` | Three-pane UI shell |
| `static/js/api.js` | Backend fetch wrappers |
| `static/js/forms.js` | Schema → form-field renderer (generic) |
| `static/js/builder.js` | Step palette, pipeline list, sections, QC editor |
| `static/js/config.js` | Builder ⇄ YAML, save/load |
| `static/js/run.js` | Run, streamed log console, captured-figure model |
| `static/js/review.js` | Paused-step panel: its plots + its parameters + re-run |
| `static/js/viewer.js` | Full-window figure viewer (lightbox) |
| `static/js/plot.js` | Plot spec → interactive plotly panels |
| `fig_spec.py` | matplotlib figure → plot spec (dashboard-only, best-effort) |
| `static/js/app.js` | Bootstrap and wiring |

## Later / offline

CodeMirror and js-yaml load from a CDN for now. To run fully offline (e.g.
wrapped as a local Tauri/pywebview app), vendor those into `static/vendor/` and
point `index.html` at the local copies.
