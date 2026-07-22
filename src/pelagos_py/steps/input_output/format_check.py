# This file is part of the NOC Autonomy pelagos_py.
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

"""Checks the format of a file against OG1/CF standards and reports the result.

A short pass/fail summary is always logged to the console. Full detail is only
written to disk when the user asks for it (``output_type``) and an
``out_directory`` is configured.
"""

#### Mandatory imports ####
from pelagos_py.steps.base_step import BaseStep, register_step

#### Custom imports ####
from compliance_checker.runner import ComplianceChecker, CheckSuite, stdout_redirector
from pathlib import Path
import re

#: The compliance checker's strictness levels map to integer score limits;
#: "lenient" keeps every priority (1=high … 3=low) in the report.
_LENIENT_LIMIT = 3

#: A missing global attribute / variable reads e.g. "Global attribute X is
#: missing" or "Variable X is missing"; pull out the bare name for the summary.
_MISSING_RE = re.compile(r"\b(?:attribute|Variable)\s+([A-Za-z0-9_]+)\s+is missing")


def _is_named(entry, *keywords):
    """Whether a check entry's name contains all the given keywords."""
    name = entry.get("name", "").lower()
    return all(k in name for k in keywords)


def _named_check(priorities, *keywords):
    """Return the ``msgs`` of the first check whose name contains all keywords.

    Used to locate the OG1 "mandatory global attributes"/"mandatory variables"
    checks by substring. Returns ``None`` when no such check ran (e.g. a checker
    that does not define it), which the caller distinguishes from "ran, none
    missing" (an empty list).
    """
    for entry in priorities:
        if _is_named(entry, *keywords):
            return entry.get("msgs", [])
    return None


def _missing_names(msgs):
    """Pull the bare attribute/variable names out of "... is missing" messages."""
    names = []
    for msg in msgs or []:
        match = _MISSING_RE.search(msg)
        names.append(match.group(1) if match else msg)
    return names


def console_summary(checker_name, result, passed):
    """Build a compact console summary for one checker's result.

    Parameters
    ----------
    checker_name : str
        The checker that produced ``result`` (e.g. ``"og"``).
    result : dict
        A single checker's :meth:`CheckSuite.dict_output` dict.
    passed : bool
        Whether the dataset passed this checker at the chosen strictness.

    Returns
    -------
    list of str
        Lines to log. Kept deliberately short: a pass/fail header, the missing
        mandatory global attributes and variables, then a count of everything
        else so the console is not flooded with detail (that goes to the file).
    """
    priorities = result.get("all_priorities", [])
    scored = result.get("scored_points")
    possible = result.get("possible_points")

    lines = [
        f"{checker_name}: {'PASS' if passed else 'FAIL'} — score {scored}/{possible}"
    ]

    global_attrs = _named_check(priorities, "mandatory", "global attribute")
    variables = _named_check(priorities, "mandatory", "variable")

    if global_attrs:
        names = _missing_names(global_attrs)
        lines.append(
            f"  Mandatory global attributes missing ({len(names)}): {', '.join(names)}"
        )
    if variables:
        names = _missing_names(variables)
        lines.append(
            f"  Mandatory variables missing ({len(names)}): {', '.join(names)}"
        )

    # Everything else is summarised as a count only, to keep the console terse.
    other = sum(
        len(entry.get("msgs", []))
        for entry in priorities
        if not _is_named(entry, "mandatory", "global attribute")
        and not _is_named(entry, "mandatory", "variable")
    )
    if other:
        lines.append(f"  + {other} other issue(s) not shown")

    return lines


@register_step
class FormatCheck(BaseStep):
    """
    Run the IOOS file-format compliance checker and report the result.

    Does not run on the in-memory dataset; it re-reads the file from disk
    (its own loading routine). A short pass/fail summary is always printed to
    the console. JSON and/or RST report files are written only when requested
    via ``output_type`` and an ``out_directory`` is set.

    Parameters
    ----------
    src : path or str, optional
        File to check. If omitted, falls back to the file loaded by a preceding
        ``Load OG1`` step.
    standards : list of str
        Standards to check, e.g. ``['cf', 'og']`` (``og`` = OG1).
    output_type : str or list of str, optional
        Report file(s) to save *in addition to* the console summary: ``'json'``,
        ``'rst'``, or a list of both. Omit (default) for console only. Saving
        requires ``out_directory`` to be set in the pipeline config.
    proceed_on_fail : bool
        If False, halt the pipeline when the file fails the checks.
    """
    step_name = "Format Checker"

    parameter_schema = {
        "src": {
            "type": str,
            "default": None,
            "description": "File to check. If omitted, falls back to the file loaded by a preceding 'Load OG1' step.",
        },
        "standards": {
            "type": list,
            "default": ["cf", "og"],
            "description": "Standards to check, e.g. ['cf', 'og'].",
        },
        "output_type": {
            "type": [str, list],
            "default": ["console"],
            "options": ["console", "json", "rst"],
            "description": "Outputs to produce: 'console' for the in-log detail summary, "
            "'json'/'rst' to also save a report file. Default ['console']. Saving files requires out_directory.",
        },
        "proceed_on_fail": {
            "type": bool,
            "default": True,
            "description": "If False, halt the pipeline when the file fails the checks.",
        },
    }

    #: Recognised entries for ``output_type``.
    _OUTPUT_OPTIONS = ("console", "json", "rst")

    def _resolve_outputs(self):
        """Normalise ``output_type`` to a list drawn from {'console', 'json', 'rst'}.

        Accepts a single string or a list; unrecognised entries are dropped. The
        overall result header is always logged regardless of this selection.
        """
        raw = self.parameters.get("output_type")
        if not raw:
            return []
        values = [raw] if isinstance(raw, str) else list(raw)
        return [v.lower() for v in values if isinstance(v, str) and v.lower() in self._OUTPUT_OPTIONS]

    def run(self):
        check_suite = CheckSuite()
        check_suite.load_all_available_checkers()

        #   Fall back to the file loaded by a preceding Load OG1 step when no src is given.
        src = self.parameters.get("src") or self.context.get("global_parameters", {}).get("source_file")
        if not src:
            self.halt(
                "No file to check. Provide a 'src' path in the config, "
                "or place this step after a 'Load OG1' step so it can reuse that file."
            )

        cnames = self.parameters.get("standards")

        #   Each requested standard is served by a compliance-checker plugin; a name with
        #   no installed plugin would otherwise surface as an opaque library traceback.
        available = {name.split(":")[0] for name in check_suite.checkers}
        missing = [c for c in cnames if c not in available]
        if missing:
            self.halt(
                f"Compliance standard(s) {missing} are not installed. "
                f"Available: {', '.join(sorted(available)) or '(none)'}. "
                f"Install the matching plugin (e.g. 'pip install cc-plugin-og' for 'og')."
            )

        #   Resolve outputs: 'console' toggles the detail log; 'json'/'rst' save files.
        outputs = self._resolve_outputs()
        console_on = "console" in outputs
        save_formats = [fmt for fmt in outputs if fmt in ("json", "rst")]

        out_dir = self.context.get("global_parameters", {}).get("out_directory")
        if save_formats and not out_dir:
            self.log_warn(
                "No 'out_directory' set in the pipeline config — cannot save report file(s). "
                "Add 'out_directory', or remove 'json'/'rst' from output_type."
            )
            save_formats = []

        #   If run after loading data, the filename stem is saved in the global pipeline params.
        fname = self.context.get("global_parameters", {}).get("filename_core") or Path(src.strip("*.nc")).stem

        #   Run every requested checker once; reuse the results for the summary + files.
        ds = check_suite.load_dataset(src)
        score_groups = check_suite.run_all(ds, cnames)
        score_dict = {src: score_groups}

        #   Gather a short per-checker summary; track the overall pass/fail.
        overall_pass = True
        summary_lines = []
        cc_results = {}
        for checker_name, (groups, _errors) in score_groups.items():
            passed = check_suite.passtree(groups, _LENIENT_LIMIT)
            overall_pass = overall_pass and passed
            result = check_suite.dict_output(checker_name, groups, src, _LENIENT_LIMIT)
            summary_lines += console_summary(checker_name, result, passed)
            cc_results[checker_name] = result

        #   Stash the structured results so the data report can render a Format
        #   Checker section regardless of whether a report file was saved.
        self.context["cc_results"] = cc_results

        #   Write the detailed report file(s), if requested and possible.
        saved = self._write_reports(check_suite, score_dict, out_dir, fname, save_formats)

        #   --- Log 1: overall result (WARNING on fail, INFO on pass). Always emitted.
        header = (
            f"'{fname}' {'passed' if overall_pass else 'FAILED'} "
            f"format compliance check(s): {', '.join(cnames)}."
        )
        (self.log if overall_pass else self.log_warn)(header)

        #   --- Log 2: the detail summary, only when 'console' is selected.
        #   Each line is logged on its own so it carries the same time/step prefix
        #   as every other line (info lines are greyed by the console formatter).
        if console_on:
            detail = list(summary_lines)
            if ComplianceChecker.check_errors(score_groups, verbose=0):
                detail.append("! Errors occurred while running the checker — see a saved report.")
            for line in detail:
                self.log(line)

        #   --- Log 3: where the full detail lives, or how to save it.
        if saved:
            self.log(
                "  ".join(f"{fmt.upper()} report saved to: {path}" for fmt, path in saved.items())
            )
        else:
            self.log(
                "Add 'json' or 'rst' to output_type (with an out_directory) to save a full report."
            )

        if not overall_pass and self.parameters.get("proceed_on_fail") == False:
            self.halt(
                f"'{fname}' failed the format compliance checks and 'proceed_on_fail' is False."
            )

        return self.context

    def _write_reports(self, check_suite, score_dict, out_dir, fname, save_formats):
        """Write the requested report file(s) and record one for the report step.

        Returns a ``{format: path}`` dict of what was written (empty for
        console-only). When both are written, the JSON path is registered as the
        ``cc_file`` the data report embeds, since it is the richer source.
        """
        if not save_formats:
            return {}

        base = out_dir + fname + "_check"
        saved = {}

        if "json" in save_formats:
            json_path = base + ".json"
            ComplianceChecker.json_output(
                check_suite, score_dict, json_path, list(score_dict), _LENIENT_LIMIT
            )
            saved["json"] = json_path

        if "rst" in save_formats:
            rst_path = base + ".rst"
            with open(rst_path, "w", encoding="utf-8") as f:
                with stdout_redirector(f):
                    ComplianceChecker.stdout_output(
                        check_suite, score_dict, 1, _LENIENT_LIMIT
                    )
            saved["rst"] = rst_path

        #   Prefer JSON for the data report (structured); fall back to RST.
        self.context["global_parameters"]["cc_file"] = saved.get("json") or saved.get("rst")
        return saved
