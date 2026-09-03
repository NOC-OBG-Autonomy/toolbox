"""Fetch the demo dataset(s) used by the other example scripts.

Run this once before the other demos to download an OG1 NetCDF file into
examples/data/OG1. nelson/churchill/alr are hosted on the BODC deployment
catalogue (see https://noc.ac.uk/projects/bio-carbon for context);
voto_og_dm is hosted on VOTO's erddap instead.

Four demos are available:
  nelson      - Nelson (unit_397), Near-Real-Time deployment. Used as-is.
  churchill   - Churchill (unit_398), Recovered deployment. The full record
                spans May-Oct 2024; it is cut down to August 2024 to keep a
                demo-sized file, then the full download is deleted.
  alr         - ALR_4 (unit_399), Recovered deployment. Used as-is.
  voto_og_dm  - SEA063 (VOTO), cut down to 2024-07-25..2024-08-03.

Usage:
  python get_demo_file.py                 # all demos (default)
  python get_demo_file.py churchill       # a single demo
  python get_demo_file.py nelson alr      # several
  python get_demo_file.py all             # everything
"""

import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

# name -> (download URL, output filename, TIME window to keep or None for whole file)
DEMOS = {
    "nelson": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001/"
        "Nelson_20240528/Nelson_646_R.nc",
        "Nelson_646_R.nc",
        None,
    ),
    "churchill": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "Churchill_20240528/Churchill_647.nc",
        "Churchill_647.nc",
        ("2024-08-01", "2024-09-01"),
    ),
    "alr": (
        "https://linkedsystems.uk/erddap/files/Public_OG1_Data_001_Recovery/"
        "ALR_4_20240609/ALR_4_649.nc",
        "ALR_4_649.nc",
        None,
    ),
    "voto_og_dm": (
        "https://erddap.observations.voiceoftheocean.org/erddap/files/"
        "OG_complete_SEA063_M75/SEA063_20240724T0737_delayed.nc",
        "SEA063_20240724T0737_delayed.nc",
        ("2024-07-25", "2024-08-03"),
    ),
}

# Work from the repo root so the relative paths below resolve the same way no
# matter where the script was started from.
_config = "examples/configs/example_config_nelson.yaml"
if not Path(_config).exists() and Path("../..", _config).exists():
    import os

    os.chdir("../..")

INPUT_DIR = Path("examples/data/OG1")


def _download(url: str, dest: Path) -> bool:
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"  download failed (HTTP {response.status_code})")
        return False
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    return True


def _cut_to_window(path: Path, start: str, end: str) -> None:
    # Keep measurements in [start, end) whose TIME is valid and strictly increasing
    # (the pipeline requires monotonic, non-NaT time). NaT excludes itself since its
    # comparisons are False; the running-max test drops any out-of-order sample.
    with xr.open_dataset(path) as ds:
        t = ds["TIME"].values
        idx = np.flatnonzero((t >= np.datetime64(start)) & (t < np.datetime64(end)))
        if idx.size:
            tt = t[idx]
            increasing = np.concatenate(([True], tt[1:] > np.maximum.accumulate(tt)[:-1]))
            idx = idx[increasing]
        n_total = ds.sizes["N_MEASUREMENTS"]
        if idx.size == 0:
            print(f"  no measurements found in {start}..{end}; leaving file unchanged")
            return
        print(f"  cutting to {start}..{end}: {idx.size} of {n_total} measurements...")
        # Read the contiguous slice covering the window, then pick out the kept rows
        # in memory: a fancy-index isel straight off the compressed file stalls for
        # millions of points.
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        subset = ds.isel(N_MEASUREMENTS=slice(lo, hi)).load().isel(N_MEASUREMENTS=idx - lo)
    # Drop the source's chunk encoding (its chunksizes were sized for the full
    # dimension and stall the subset write); re-apply zlib to keep the file small.
    encoding = {}
    for v in subset.variables:
        subset[v].encoding = {}
        if subset[v].dtype.kind in "fiu":
            encoding[v] = {"zlib": True, "complevel": 4}
    tmp = path.with_suffix(".full.nc")
    path.rename(tmp)
    subset.to_netcdf(path, encoding=encoding)
    tmp.unlink()  # drop the full download, keep only the window
    print(f"  done ({idx.size} measurements kept)")


def fetch(name: str) -> None:
    url, filename, window = DEMOS[name]
    dest = INPUT_DIR / filename
    if dest.exists():
        print(f"{name}: already present at {dest.resolve()}")
        return
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not _download(url, dest):
        return
    if window is not None:
        _cut_to_window(dest, *window)
    print(f"{name}: written to {dest.resolve()}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    names = list(DEMOS) if "all" in args else args
    for name in names:
        if name not in DEMOS:
            print(f"Unknown demo '{name}'. Choose from: {', '.join(DEMOS)}, all")
            continue
        fetch(name)
