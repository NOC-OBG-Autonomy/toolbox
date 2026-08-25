"""Open the pelagos_py config dashboard in your web browser.

Run this to launch the dashboard -- the web tool for authoring, validating and
running pipeline YAML configs. It serves the dashboard on localhost and opens a
browser tab pointing at it.

    python examples/python/run_dashboard.py

(Or just hit Run in VS Code.) Nothing to configure -- it runs the same server
as ``python dashboard/app.py``, only wrapped so a single click opens the tab.
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path

# Work from the repo root so the dashboard's relative paths resolve the same way
# no matter where the script was started from (terminal, VS Code, double-click).
_marker = "dashboard/app.py"
if not Path(_marker).exists() and Path("../..", _marker).exists():
    os.chdir("../..")

# app.py imports sibling dashboard modules by bare name, so put dashboard/ on the
# path before importing it.
sys.path.insert(0, str(Path("dashboard").resolve()))

import uvicorn
from app import app

# Bind to the loopback IP (dodges the localhost->::1 IPv6 case) but show the
# friendlier hostname in the URL we print and open.
url = "http://localhost:8791"
print(f"pelagos_py dashboard -> {url}")
# Open the tab a moment after uvicorn.run() (below) has had time to bind.
threading.Timer(1.0, webbrowser.open, args=(url,)).start()
uvicorn.run(app, host="127.0.0.1", port=8791, log_level="warning")
