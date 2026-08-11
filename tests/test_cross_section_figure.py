"""Tests cross_section_figure (src/pelagos_py/steps/input_output/write_report_python.py)."""

from pelagos_py.steps.input_output import write_report_python as wrp

import numpy as np
import xarray as xr

cross_section_figure = wrp.cross_section_figure


def make_dataset(n=200, extra=None):
    rng = np.random.default_rng(0)
    time = np.datetime64("2024-01-01") + (np.arange(n) * 60).astype("timedelta64[s]")
    ds = xr.Dataset(
        {
            "TIME": ("N_MEASUREMENTS", time),
            "PRES": ("N_MEASUREMENTS", np.linspace(0, 200, n)),
            "TEMP": ("N_MEASUREMENTS", rng.normal(10, 1, n)),
        }
    )
    if extra:
        for name, values in extra.items():
            ds[name] = ("N_MEASUREMENTS", np.asarray(values, dtype=float))
    return ds


def test_returns_none_when_no_panel_variable_has_data(tmp_path):
    ds = make_dataset()
    ds["TEMP"].values[:] = np.nan  # only candidate variable present, but empty

    img = cross_section_figure(ds, str(tmp_path) + "/")

    assert img is None


def _row_count_drawn(monkeypatch, ds, tmp_path, subdir):
    #   Spy on plt.close to read the gridspec row count before the figure is torn down.
    captured = {}
    orig_close = wrp.plt.close

    def spy_close(fig):
        captured["nrows"] = fig.axes[0].get_gridspec().nrows
        orig_close(fig)

    monkeypatch.setattr(wrp.plt, "close", spy_close)
    outdir = tmp_path / subdir
    outdir.mkdir()
    img = cross_section_figure(ds, str(outdir) + "/")
    assert img is not None
    return captured["nrows"]


def test_only_panels_with_data_are_drawn(tmp_path, monkeypatch):
    n = 200

    ds_two = make_dataset(n, extra={"PRAC_SALINITY": np.full(n, 35.0)})
    rows_two = _row_count_drawn(monkeypatch, ds_two, tmp_path, "two")

    ds_all = make_dataset(
        n,
        extra={
            "PRAC_SALINITY": np.full(n, 35.0),
            "DENSITY": np.full(n, 1025.0),
            "MOLAR_DOXY": np.full(n, 200.0),
            "CHLA": np.full(n, 0.5),
            "BBP700": np.full(n, 0.001),
            "DOWNWELLING_PAR": np.full(n, 100.0),
        },
    )
    rows_all = _row_count_drawn(monkeypatch, ds_all, tmp_path, "all")

    # Temperature + Salinity only, vs all seven panels having data.
    assert rows_two == 2
    assert rows_all == len(wrp._CROSS_SECTION_PANELS)


def test_oxygen_panel_prefers_fully_corrected_variable(tmp_path):
    n = 200
    ds = make_dataset(
        n,
        extra={
            "MOLAR_DOXY": np.full(n, 200.0),
            "MOLAR_DOXY_PSAL_PRES": np.full(n, 250.0),
        },
    )

    resolved = wrp._first_present(
        ds, next(p for p in wrp._CROSS_SECTION_PANELS if p["label"] == "Oxygen")["candidates"]
    )

    assert resolved == "MOLAR_DOXY_PSAL_PRES"
