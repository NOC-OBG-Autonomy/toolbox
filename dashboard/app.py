# This file is part of pelagos_py.
#
# Copyright 2025-2026 National Oceanography Centre and The Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FastAPI backend for the pelagos_py config dashboard.

The dashboard is a *standalone* helper for authoring, validating and running
pipeline YAML configs. It never modifies the pipeline: it only introspects the
live step/QC registries (so newly-added steps appear automatically) and reuses
the pipeline's own ``parameter_spec`` validation, so what the dashboard accepts
is exactly what the pipeline accepts.

Run with::

    python dashboard/app.py            # then open http://localhost:8791
"""

from __future__ import annotations

import codecs
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Matches ANSI/VT100 control sequences (colour, cursor moves, clear-line) that
# tqdm and coloured loggers emit.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean_ansi(text: str) -> str:
    """Keep colour, drop terminal-only control codes.

    SGR sequences (``CSI … m``) are passed through for the browser console to
    render as styled spans; cursor moves and clear-line sequences mean nothing
    there, so they are dropped.
    """
    return _ANSI_RE.sub(lambda m: m.group(0) if m.group(0).endswith("m") else "", text)

import numpy as np
import xarray as xr
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Same-directory import as run_bootstrap.py's (script's own dir is on
# sys.path); reused here for the zoomed-in range-query decimation.
import fig_spec

# --- Locate the repo and make pelagos_py importable -------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
SRC_DIR = REPO_ROOT / "src"
STATIC_DIR = DASHBOARD_DIR / "static"
RUN_BOOTSTRAP = DASHBOARD_DIR / "run_bootstrap.py"
# Diagnostic figures captured from the current run are written here and served
# to the browser (see run_bootstrap.py). Cleared at the start of each run.
FIG_DIR = DASHBOARD_DIR / "_run_figures"
FIG_DIR.mkdir(exist_ok=True)
# Full-resolution trace arrays for zoomed-in range queries (see run_figdata),
# loaded from a figure's "_full.npz" lazily and kept only for this process's
# lifetime -- cleared on the next run, never written back to disk.
_FIGDATA_CACHE: dict[str, dict] = {}
# Configs authored in the dashboard live here by default.
CONFIG_DIR = DASHBOARD_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importing the package triggers discover_steps(), which imports every step/QC
# module and populates the registries. This is the single source of truth for
# "what steps exist" -- the dashboard derives everything from it.
from pelagos_py.steps import STEP_CLASSES, QC_CLASSES  # noqa: E402
from pelagos_py.utils import parameter_spec  # noqa: E402
from pelagos_py.utils.demo_data import DEMOS as DEMO_FILES, DEMO_DATA_DIR, MISSIONS  # noqa: E402


# Pipeline-level keys (the top ``pipeline:`` block) are not part of any step
# schema, so they are described here. Kept deliberately small and stable.
PIPELINE_FIELDS = [
    {"name": "name", "type": "str", "required": False, "default": "",
     "description": "A short name for the pipeline."},
    {"name": "description", "type": "str", "required": False, "default": "",
     "description": "Longer description of the pipeline's purpose."},
    {"name": "out_directory", "type": "str", "required": False, "default": "./",
     "description": "Output directory for generated files (logs, reports, figures)."},
    {"name": "log_file", "type": "str", "required": False, "default": None,
     "description": "Log file name. Leave blank/null for console-only logging."},
    {"name": "continue_on_step_fail", "type": "bool", "required": False, "default": True,
     "description": "Skip a step that fails and continue the pipeline (logged as a "
                     "severe warning) instead of stopping the whole run."},
]


def _category(cls) -> str:
    """Derive a step's category from its module path (processing / qc / io)."""
    module = getattr(cls, "__module__", "")
    if ".quality_control" in module:
        return "quality_control"
    if ".processing" in module:
        return "processing"
    if ".input_output" in module:
        return "input_output"
    return "other"


def _short_doc(cls) -> str:
    """First non-empty paragraph of a class docstring, whitespace-collapsed."""
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    # take up to the first blank line
    para = doc.split("\n\n", 1)[0]
    return " ".join(para.split())


def _describe_step(name: str, cls) -> dict:
    return {
        "name": name,
        "kind": "step",
        "category": _category(cls),
        "module": getattr(cls, "__module__", ""),
        "description": _short_doc(cls),
        # ``parameter_schema is None`` => not yet migrated to strict validation.
        "schema_declared": getattr(cls, "parameter_schema", None) is not None,
        "parameters": cls.describe_parameters(),
    }


def _describe_qc(name: str, cls) -> dict:
    return {
        "name": name,
        "kind": "qc",
        "category": "quality_control",
        "module": getattr(cls, "__module__", ""),
        "description": _short_doc(cls),
        "schema_declared": True,
        "parameters": cls.describe_parameters(),
        "required_variables": list(getattr(cls, "required_variables", []) or []),
        "qc_outputs": list(getattr(cls, "qc_outputs", []) or []),
    }


app = FastAPI(title="pelagos_py dashboard")


@app.middleware("http")
async def _no_cache(request, call_next):
    """Turn off caching for development
    
    """
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path == "/" or path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# =============================== Introspection ==============================
@app.get("/api/registry")
def registry():
    """Everything the frontend needs to render the step palette and forms.

    Reflects the live registries, so a newly ``@register_step``-ed class shows
    up here on the next server start with zero dashboard changes.
    """
    # Skip the blank template scaffolds -- they are registered so the machinery
    # is exercised, but they are not real, pickable pipeline steps.
    def _is_template(cls):
        return ".templates" in getattr(cls, "__module__", "")

    steps = [
        _describe_step(n, c) for n, c in sorted(STEP_CLASSES.items())
        if not _is_template(c)
    ]
    qc = [
        _describe_qc(n, c) for n, c in sorted(QC_CLASSES.items())
        if not _is_template(c)
    ]
    return {
        "steps": steps,
        "qc": qc,
        "pipeline_fields": PIPELINE_FIELDS,
    }


# ================================ Validation ================================
class ValidatePayload(BaseModel):
    yaml_content: str


@app.post("/api/validate")
def validate(payload: ValidatePayload):
    """Validate a whole config using the pipeline's real ``parameter_spec``.

    Returns structured, per-step issues rather than a single string, so the UI
    can point at the offending step. Uses the same ``resolve()`` the pipeline
    uses, so acceptance here == acceptance at run time.
    """
    try:
        config = yaml.safe_load(payload.yaml_content)
    except yaml.YAMLError as exc:
        return {"ok": False, "yaml_error": str(exc), "issues": []}

    if not isinstance(config, dict):
        return {"ok": False, "yaml_error": "Top-level config must be a mapping.",
                "issues": []}

    issues = []
    steps = config.get("steps") or []
    if not isinstance(steps, list):
        return {"ok": False, "yaml_error": "'steps' must be a list.", "issues": []}

    for index, step in enumerate(steps):
        if not isinstance(step, dict) or "name" not in step:
            issues.append({"index": index, "name": None,
                           "error": "Each step needs a 'name'."})
            continue
        name = step["name"]
        cls = STEP_CLASSES.get(name)
        if cls is None:
            # case-insensitive courtesy match
            lowered = {k.lower(): k for k in STEP_CLASSES}
            canonical = lowered.get(str(name).lower())
            if canonical is None:
                issues.append({"index": index, "name": name,
                               "error": f"Unknown step '{name}'."})
                continue
            cls = STEP_CLASSES[canonical]

        schema = getattr(cls, "parameter_schema", None)
        if schema is None:
            continue  # step opted out of strict validation
        params = step.get("parameters") or {}
        try:
            parameter_spec.resolve(
                schema, params, label=name,
                allowed_extra=getattr(cls, "framework_parameters", ()),
            )
        except ValueError as exc:
            issues.append({"index": index, "name": name, "error": str(exc)})

    return {"ok": not issues, "yaml_error": None, "issues": issues}


# ============================ Config management =============================
class SavePayload(BaseModel):
    name: str
    yaml_content: str


#: Demo configs, one per demo glider in pelagos_py.utils.demo_data.DEMOS. These
#: are virtual -- there is no demo_<key>.yaml on disk for each one -- their
#: YAML is synthesised on load by patching the glider's file path into one of
#: the two shipped templates (see _demo_yaml). Read-only for the same reason
#: as PROTECTED_CONFIGS below, and surfaced separately so the UI can group them.
DEMO_CONFIGS = {f"demo_{key}.yaml" for key in DEMO_FILES}

#: Reference configs shipped with the dashboard: the blank glider template and
#: the ALR template every demo config is patched from. They are read-only: the
#: UI forks an edited one to a new ``custom_run_N.yaml`` rather than
#: overwriting, and the API refuses to save or delete them, so a hand-crafted
#: request (or a stale browser tab) can't destroy them either.
PROTECTED_CONFIGS = {"default.yaml", "demo_alr.yaml"} | DEMO_CONFIGS


def _safe_config_path(name: str) -> Path:
    """Resolve ``name`` to a path inside CONFIG_DIR, rejecting traversal."""
    candidate = (CONFIG_DIR / name).resolve()
    if candidate.parent != CONFIG_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid config name.")
    if candidate.suffix not in (".yaml", ".yml"):
        candidate = candidate.with_suffix(".yaml")
    return candidate


def _demo_dest(config_name: str) -> Path | None:
    """The local path a demo config's NetCDF file lives (or would be downloaded
    to), or None if ``config_name`` isn't a demo config."""
    key = config_name[len("demo_"):-len(".yaml")]
    entry = DEMO_FILES.get(key)
    if entry is None:
        return None
    return REPO_ROOT / DEMO_DATA_DIR / entry.filename


@app.get("/api/configs")
def list_configs():
    files = sorted(
        p.name for p in CONFIG_DIR.iterdir()
        if p.is_file() and p.suffix in (".yaml", ".yml")
    )
    # Demo configs are virtual (see DEMO_CONFIGS above), so unlike the other
    # groups they're listed unconditionally rather than filtered by `files`.
    demo = sorted(DEMO_CONFIGS)
    return {
        "configs": sorted(set(files) | DEMO_CONFIGS),
        "protected": sorted(PROTECTED_CONFIGS),
        "demo": demo,
        # Demo config names grouped by deployment mission, in picker display
        # order, so the UI can show which glider belongs to which campaign.
        "missions": {
            mission: [f"demo_{key}.yaml" for key in keys]
            for mission, keys in MISSIONS.items()
        },
        # Display label per demo config name (glider names aren't unique --
        # across missions, e.g. "Churchill" and "Zephyr" each appear twice,
        # and within one glider, NRT vs Full is a separate entry).
        "labels": {f"demo_{key}.yaml": entry.display_label for key, entry in DEMO_FILES.items()},
        # Non-demo protected configs (default.yaml, demo_alr.yaml), shown as
        # their own "Default" group in the picker.
        "reference": sorted((PROTECTED_CONFIGS - DEMO_CONFIGS) & set(files)),
        # Which demo configs already have their NetCDF file on disk, so the
        # picker can show download status before the file is needed.
        "downloaded": sorted(name for name in demo if _demo_dest(name).exists()),
    }


@app.post("/api/configs/reveal")
def reveal_configs():
    """Open the configs folder in the OS file browser.

    Runs on the server, so it only shows a window when the dashboard is being
    viewed on the machine serving it (the normal 127.0.0.1 case).
    """
    if sys.platform == "darwin":
        cmd = ["open", str(CONFIG_DIR)]
    elif os.name == "nt":
        cmd = ["explorer", str(CONFIG_DIR)]
    else:
        cmd = ["xdg-open", str(CONFIG_DIR)]
    try:
        subprocess.Popen(cmd)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not open {CONFIG_DIR}: {exc}"
        ) from exc
    return {"status": "opened", "path": str(CONFIG_DIR)}


def _ensure_demo_file(config_name: str) -> None:
    """Download a demo config's input NetCDF file if it isn't there yet.

    Demo configs normally get their data from running
    ``examples/python/get_demo_file.py`` first, but a config picked straight
    from the dashboard shouldn't just fail to load if that step was skipped.
    Unlike that script, this doesn't trim churchill's window down to a
    demo-sized excerpt -- it's a fallback, not a replacement for running it.
    """
    dest = _demo_dest(config_name)
    if dest is None or dest.exists():
        return
    key = config_name[len("demo_"):-len(".yaml")]
    entry = DEMO_FILES[key]
    dest.parent.mkdir(parents=True, exist_ok=True)
    import requests

    # Files are 100s of MB; only bound the connect phase so a slow-but-alive
    # transfer of a large file isn't mistaken for a hang.
    response = requests.get(entry.url, stream=True, timeout=(15, None))
    response.raise_for_status()
    tmp = dest.with_name(dest.name + ".part")
    try:
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)  # left behind only if the download failed


_DEMO_FIELD_RE = {
    field: re.compile(rf"(?m)^(\s*{field}:).*$")
    for field in ("file_path", "output_path", "description")
}


def _demo_yaml(config_name: str) -> str:
    """Synthesise a demo config's YAML by patching its glider's file path,
    output path and description into the shared "default" or "alr" template
    (see DemoEntry.template) -- every demo glider reuses one of those two
    configs rather than shipping its own near-duplicate file.
    """
    key = config_name[len("demo_"):-len(".yaml")]
    entry = DEMO_FILES[key]
    template_name = "demo_alr.yaml" if entry.template == "alr" else "default.yaml"
    text = (CONFIG_DIR / template_name).read_text()
    rel_path = f"{DEMO_DATA_DIR}/{entry.filename}"
    stem = Path(entry.filename).stem
    text = _DEMO_FIELD_RE["file_path"].sub(
        rf"\1 {rel_path}  # Path to the input NetCDF file", text, count=1,
    )
    text = _DEMO_FIELD_RE["output_path"].sub(
        rf'\1 "{DEMO_DATA_DIR}/{stem}_Processed.nc"', text, count=1,
    )
    text = _DEMO_FIELD_RE["description"].sub(
        rf"\1 A demo pipeline using {entry.display_label} data.", text, count=1,
    )
    return text


@app.get("/api/configs/{name}")
def load_config(name: str):
    demo_name = name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"
    if demo_name in DEMO_CONFIGS:
        try:
            _ensure_demo_file(demo_name)
            return {"name": demo_name, "yaml_content": _demo_yaml(demo_name)}
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not download demo data: {exc}"
            ) from exc
    path = _safe_config_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Config not found.")
    return {"name": path.name, "yaml_content": path.read_text()}


@app.post("/api/configs")
def save_config(payload: SavePayload):
    path = _safe_config_path(payload.name)
    if path.name in PROTECTED_CONFIGS:
        raise HTTPException(
            status_code=403,
            detail=f"'{path.name}' is a locked reference config. "
                   "Save your changes under a different name.",
        )
    path.write_text(payload.yaml_content)
    return {"status": "saved", "name": path.name}


@app.delete("/api/configs/{name}")
def delete_config(name: str):
    path = _safe_config_path(name)
    if path.name in PROTECTED_CONFIGS:
        raise HTTPException(
            status_code=403,
            detail=f"'{path.name}' is a locked reference config and cannot be deleted.",
        )
    if path.exists():
        path.unlink()
    return {"status": "deleted", "name": path.name}


# ============================== Pipeline runner =============================
class _Run:
    """Holds the single active pipeline subprocess and its captured log lines.

    Only one run at a time -- the dashboard is a single-user local tool. Output
    is pumped off the subprocess's merged stdout/stderr by a background thread.

    Progress bars (tqdm) redraw one line with a carriage return (``\\r``) rather
    than emitting a new line each tick. Reading in binary preserves those ``\\r``
    boundaries so a whole bar collapses to a single, in-place-updating line in
    the console instead of thousands of spam lines.

    State is an append-only list of committed ``lines`` plus the latest transient
    progress redraw (``live``). The SSE endpoint reads these by index rather than
    draining a queue, so any number of clients -- including one reconnecting after
    a page refresh mid-run -- each replay the full log independently, with no
    duplicated or stolen events.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: list[str] = []  # committed lines, append-only (replayable)
        self.live: str | None = None  # latest transient progress redraw, if any
        self.live_after = 0  # index in `lines` the current `live` follows
        self.finished = False
        self.returncode: int | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, config_path: Path):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Output goes to a pipe, not a terminal, so the pipeline would otherwise
        # drop its colour and disable its tqdm bars (see utils/log_levels.py
        # _supports_color, which FORCE_COLOR overrides). The browser console
        # renders both, so ask for them. COLUMNS gives the bars a sane width,
        # since there is no terminal to measure.
        env["FORCE_COLOR"] = "1"
        env.pop("NO_COLOR", None)
        env["COLUMNS"] = "110"
        # Ensure the subprocess can import pelagos_py from src/.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_DIR), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        # Fresh figure dir per run so the Plots tab only shows this run's plots.
        # .json are the interactive plot specs, _full.npz the full-resolution
        # zoom captures, saved beside each .png.
        for pattern in ("*.png", "*.json", "*_full.npz"):
            for old in FIG_DIR.glob(pattern):
                old.unlink(missing_ok=True)
        _FIGDATA_CACHE.clear()
        # run_bootstrap.py redirects plt.show to save diagnostic figures into
        # FIG_DIR (and catches the Stop-button SIGINT) -- see that file.
        self.proc = subprocess.Popen(
            [sys.executable, str(RUN_BOOTSTRAP), str(config_path), str(FIG_DIR)],
            cwd=str(REPO_ROOT),  # relative paths in configs resolve from repo root
            stdin=subprocess.PIPE,  # control channel for pause/continue/rerun
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=-1,  # buffered binary: gives a BufferedReader (has read1);
            # \r is preserved either way, and read1 returns as soon as any
            # bytes arrive so progress-bar ticks still stream promptly.
        )
        self.lines = []
        self.live = None
        self.live_after = 0
        self.finished = False
        self.returncode = None
        threading.Thread(target=self._pump, daemon=True).start()

    def _commit(self, text: str):
        """A finished (newline-terminated) line: append it and clear live progress."""
        with self._lock:
            self.lines.append(text)
            self.live = None
            self.live_after = len(self.lines)

    def _progress(self, text: str):
        """A transient in-place redraw (bar tick): update live, don't accumulate."""
        with self._lock:
            self.live = text
            self.live_after = len(self.lines)

    def _pump(self):
        assert self.proc is not None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream = self.proc.stdout  # binary BufferedReader
        seg = ""  # chars seen since the last \r or \n
        pending_cr = False  # saw a \r; a \n next means it was a \r\n line ending
        while True:
            chunk = stream.read1(4096)  # type: ignore[union-attr]
            if not chunk:
                break
            for ch in decoder.decode(chunk):
                if ch == "\n":
                    # A \r right before this \n is a Windows (\r\n) line ending,
                    # not a redraw -- treat the whole pair as one newline.
                    pending_cr = False
                    self._commit(_clean_ansi(seg))
                    seg = ""
                    continue
                if pending_cr:
                    # The earlier \r had no \n after it: a real in-place redraw
                    # (tqdm bar tick). Emit what was drawn, then start fresh.
                    cleaned = _clean_ansi(seg)
                    if cleaned:
                        self._progress(cleaned)
                    seg = ""
                    pending_cr = False
                if ch == "\r":
                    pending_cr = True
                else:
                    seg += ch
        if seg:  # trailing text with no final newline
            if pending_cr:
                cleaned = _clean_ansi(seg)
                if cleaned:
                    self._progress(cleaned)
            else:
                self._commit(_clean_ansi(seg))
        self.returncode = self.proc.wait()
        self.finished = True

    def stop(self):
        if self.is_running():
            # SIGINT (not SIGTERM) so the child raises KeyboardInterrupt and runs
            # its Python cleanup -- atexit/finally, multiprocessing pool shutdown --
            # releasing pool semaphores instead of leaking them on abrupt exit.
            # Escalate to SIGTERM then SIGKILL if it doesn't stop promptly.
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

    def send(self, command: str):
        """Write a control line to the paused run's stdin.

        Drives the interactive pause protocol in run_bootstrap.py:
        ``continue`` / ``rerun <json>`` / ``stop``. No-op (swallowed) if the
        process has already exited or its stdin is gone.
        """
        if self.proc is not None and self.proc.stdin is not None and self.is_running():
            try:
                self.proc.stdin.write((command + "\n").encode())
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass


_run = _Run()


class RunPayload(BaseModel):
    yaml_content: str


@app.post("/api/run")
def run_pipeline(payload: RunPayload):
    if _run.is_running():
        raise HTTPException(status_code=409, detail="A pipeline is already running.")
    # Persist the exact YAML being run so the subprocess (and the user) can see it.
    run_path = CONFIG_DIR / "_last_run.yaml"
    run_path.write_text(payload.yaml_content)
    _run.start(run_path)
    return {"status": "started"}


@app.post("/api/run/stop")
def stop_pipeline():
    _run.stop()
    return {"status": "stopping"}


class RerunPayload(BaseModel):
    parameters: dict = {}


@app.post("/api/run/continue")
def continue_run():
    """Resume a run paused at a diagnostics step (interactive stepping)."""
    _run.send("continue")
    return {"status": "continued"}


@app.post("/api/run/rerun")
def rerun_step(payload: RerunPayload):
    """Re-run the currently paused step with edited parameters, then re-pause."""
    _run.send("rerun " + json.dumps(payload.parameters))
    return {"status": "rerunning"}


@app.get("/api/run/status")
def run_status():
    return {
        "running": _run.is_running(),
        "finished": _run.finished,
        "returncode": _run.returncode,
        "line_count": len(_run.lines),
    }


@app.get("/api/run/stream")
def stream_logs():
    """Server-Sent Events stream of the current run's log lines.

    Reads the run's append-only state by index, so it replays everything already
    captured (a client connecting late -- e.g. after a mid-run page refresh --
    sees the whole run) and then tails new output until the process exits. Each
    connection keeps its own cursors, so reconnecting never steals or duplicates
    events.
    """
    def frame(kind: str, text: str) -> str:
        # 'line' -> default SSE event (message); 'progress' -> named event.
        prefix = "" if kind == "line" else f"event: {kind}\n"
        return f"{prefix}data: {text}\n\n"

    def event_gen():
        cursor = 0  # next committed-line index to emit
        last_live = None  # last progress text emitted, to avoid repeats
        idle = 0.0
        while True:
            with _run._lock:
                new_lines = _run.lines[cursor:]
                cursor = len(_run.lines)
                live = _run.live
                finished = _run.finished
                returncode = _run.returncode
            for line in new_lines:
                last_live = None  # a committed line supersedes any live redraw
                yield frame("line", line)
            if live is not None and live != last_live:
                last_live = live
                yield frame("progress", live)
            if finished and cursor >= len(_run.lines):
                yield f"event: end\ndata: {returncode}\n\n"
                return
            if not new_lines:
                idle += 0.1
                if idle >= 15.0:  # periodic comment frame keeps the connection open
                    idle = 0.0
                    yield ": keep-alive\n\n"
            else:
                idle = 0.0
            time.sleep(0.1)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/run/figure/{name}")
def run_figure(name: str):
    """Serve a diagnostic figure captured from the current run by filename.

    The browser requests these after seeing a ``__PELAGOS_FIG__`` marker line in
    the log stream. ``Path(name).name`` strips any directory component so a
    crafted name can't escape FIG_DIR.
    """
    path = FIG_DIR / Path(name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/run/figspec/{name}")
def run_figspec(name: str):
    """Serve the interactive plot spec saved beside a captured figure.

    Written by run_bootstrap.py via fig_spec.py, and requested only for figures
    whose ``__PELAGOS_FIG__`` marker named one. Same traversal guard as above.
    """
    path = FIG_DIR / Path(name).name
    if path.suffix != ".json" or not path.is_file():
        raise HTTPException(status_code=404, detail="Plot spec not found.")
    return FileResponse(path, media_type="application/json")


def _load_fullres(stem: str) -> dict:
    """Full-resolution captures for a figure, loaded from its ``.npz`` once and
    cached in memory for the rest of this process's life (see _FIGDATA_CACHE).

    Maps ``"<panel>_<trace>"`` to ``{"x", "y", "color"}`` numpy arrays.
    """
    if stem in _FIGDATA_CACHE:
        return _FIGDATA_CACHE[stem]
    data: dict = {}
    path = FIG_DIR / (stem + "_full.npz")
    if path.is_file():
        with np.load(path) as npz:
            keys = {k.rsplit("_", 1)[0] for k in npz.files if k.endswith(("_x", "_y"))}
            for key in keys:
                data[key] = {
                    "x": npz[f"{key}_x"],
                    "y": npz[f"{key}_y"],
                    "color": npz[f"{key}_color"] if f"{key}_color" in npz.files else None,
                }
    _FIGDATA_CACHE[stem] = data
    return data


def _lttb_indices(x: "np.ndarray", y: "np.ndarray", cap: int):
    """Largest-Triangle-Three-Buckets indices thinning ``x``/``y`` to ``cap``
    points: ``(keep_idx, thinned)``, mirroring ``fig_spec._decimate_indices``.

    Zoom-endpoint only (see the user's choice to keep the original write-time
    spec on the plain min/max bucket decimator). That decimator picks each
    bucket's y-min *and* y-max by index-position, which on a repeating signal
    -- a profiling glider's depth sawtooth -- degenerates at a wide zoom into
    a clump of near-surface and near-bottom points with the dive/climb slopes
    between them dropped, since a slope's points are never a bucket's extreme.
    LTTB instead keeps, per bucket, whichever point forms the largest triangle
    with the previously-kept point and the next bucket's average -- it
    preserves the visual shape (including slopes) rather than only extremes.
    """
    n = len(x)
    if n <= cap:
        return np.arange(n), False
    if cap < 3:
        step = int(math.ceil(n / max(1, cap)))
        return np.arange(0, n, step)[:cap], True

    every = (n - 2) / (cap - 2)
    keep = np.empty(cap, dtype=np.int64)
    keep[0] = 0
    keep[-1] = n - 1
    a = 0
    for i in range(cap - 2):
        avg_lo = min(int((i + 1) * every) + 1, n - 1)
        avg_hi = min(int((i + 2) * every) + 1, n)
        avg_x = np.nanmean(x[avg_lo:avg_hi]) if avg_hi > avg_lo else x[avg_lo]
        avg_y = np.nanmean(y[avg_lo:avg_hi]) if avg_hi > avg_lo else y[avg_lo]

        lo = min(int(i * every) + 1, n - 1)
        hi = min(int((i + 1) * every) + 1, n)
        ys = y[lo:hi]
        valid = np.flatnonzero(np.isfinite(ys))
        if len(valid) == 0:
            idx = lo
        else:
            xs = x[lo:hi][valid]
            area = np.abs((x[a] - avg_x) * (ys[valid] - y[a]) - (x[a] - xs) * (avg_y - y[a]))
            idx = lo + int(valid[np.argmax(area)])
        keep[i + 1] = idx
        a = idx
    return keep, True


def _pack_binary(header: dict, arrays: list) -> bytes:
    """The figdata wire format: uint32 header length, JSON header, then each
    array's raw little-endian bytes back to back -- lets the browser slice
    straight into typed arrays instead of JSON-parsing huge point lists."""
    header_bytes = json.dumps(header).encode("utf-8")
    parts = [len(header_bytes).to_bytes(4, "little"), header_bytes]
    parts.extend(a.tobytes() for a in arrays)
    return b"".join(parts)


@app.get("/api/run/figdata/{name}")
def run_figdata(name: str, panel: int, trace: int, x_min: float, x_max: float,
                 cap: int = fig_spec.ZOOM_POINT_CAP):
    """Full-resolution points for one trace within ``[x_min, x_max]``.

    ``name`` is the same figure filename the browser got from figspec (e.g.
    ``fig_001.json``); only its stem is used, so the traversal guard is the
    same as the other figure routes. ``x_min``/``x_max`` are epoch
    milliseconds for a date axis, else the raw axis values -- whatever unit
    ``dashboard/fig_spec.py``'s ``_fullres`` stored. Response is packed
    binary (see ``_pack_binary``); ``complete`` in its header tells the
    browser this range needed no further thinning, so it can stop re-fetching
    as the user zooms further into it.
    """
    stem = Path(Path(name).name).stem
    entry = _load_fullres(stem).get(f"{panel}_{trace}")
    if entry is None:
        raise HTTPException(status_code=404, detail="No full-resolution data for this trace.")
    x, y, color = entry["x"], entry["y"], entry["color"]
    lo = int(np.searchsorted(x, x_min, side="left"))
    hi = int(np.searchsorted(x, x_max, side="right"))
    sx, sy = x[lo:hi], y[lo:hi]
    scolor = color[lo:hi] if color is not None else None

    cap = max(1, min(cap, fig_spec.ZOOM_POINT_CAP))
    keep, thinned = _lttb_indices(sx, sy, cap)
    rx, ry = sx[keep], sy[keep]
    rcolor = scolor[keep] if scolor is not None else None

    header = {
        "n": int(len(rx)), "complete": not thinned,
        "x_dtype": "float64", "y_dtype": "float64", "has_color": rcolor is not None,
    }
    arrays = [rx.astype("<f8"), ry.astype("<f8")]
    if rcolor is not None:
        arrays.append(np.ascontiguousarray(rcolor, dtype=np.uint8))
    return Response(content=_pack_binary(header, arrays), media_type="application/octet-stream")


@app.get("/api/run/report")
def run_report(path: str):
    """Serve a PDF report produced by the current run, for the Report tab.

    The browser passes the absolute path it saw in a ``__PELAGOS_REPORT__``
    marker. Restricted to existing ``.pdf`` files; this is a localhost
    single-user tool that already runs arbitrary configs, so there is no sandbox
    beyond that. ``inline`` so the browser previews it rather than downloading.
    """
    p = Path(path)
    if p.suffix.lower() != ".pdf" or not p.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        p, media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{p.name}"'},
    )


# ================================== Inspect ==================================
@app.get("/api/inspect")
def inspect_file(file_path: str):
    """Variables, global attributes and sensors of a NetCDF file, for the Inspect tab.

    ``file_path`` is read straight from the config's 'Load OG1' step, resolved
    against the repo root the same way the pipeline itself resolves it at run time.
    """
    if not file_path:
        raise HTTPException(status_code=400, detail="No file_path given.")
    path = Path(file_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        with xr.open_dataset(path) as ds:
            variables = [
                {
                    "name": name,
                    "units": var.attrs.get("units", ""),
                    "description": var.attrs.get("long_name") or var.attrs.get("comment") or "",
                    "dtype": str(var.dtype),
                }
                for name, var in ds.variables.items()
            ]
            global_attrs = {k: str(v) for k, v in ds.attrs.items()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open '{path.name}': {exc}")

    # Sensors: the 'instrument' global attribute, split into one entry per
    # instrument (handles a plain "a, b, c" string or a Python-list-style value).
    instr_key = next((k for k in global_attrs if k.lower() == "instrument"), None)
    raw = global_attrs.get(instr_key, "") if instr_key else ""
    sensors = [
        s.strip().strip("'\"")
        for s in raw.strip().lstrip("[").rstrip("]").split(",")
        if s.strip()
    ]

    return {
        "path": str(path),
        "variables": variables,
        "global_attributes": global_attrs,
        "sensors": sensors,
    }


# ================================ Static site ===============================
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    # Bind to the loopback IP but show the friendlier hostname in the URL.
    print("pelagos_py dashboard -> http://localhost:8791")
    uvicorn.run(app, host="127.0.0.1", port=8791, log_level="warning")
