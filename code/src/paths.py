"""Every filesystem location the project uses, resolved in one place.

The package lives under `code/`, while `data/` and `results/` sit at the
repository root beside it. Deriving all of that from `__file__` in each module
means the whole tree breaks whenever a directory moves -- which is exactly what
happened when the package was moved into `code/`. Only this file knows how deep
the package sits; everything else imports from it.

Layout:
    <root>/code/src/          this package
    <root>/data/external/     bundled CPC18 aggregates
    <root>/data/processed/    generated arm caches (.npz), not in git
    <root>/results/           runs, tables, figures
"""

from __future__ import annotations

import os

# .../<root>/code/src/paths.py -> up 3 to <root>
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))

CODE = os.path.join(PROJECT_ROOT, "code")
DATA = os.path.join(PROJECT_ROOT, "data")
EXTERNAL = os.path.join(DATA, "external")
PROCESSED = os.path.join(DATA, "processed")
RESULTS = os.path.join(PROJECT_ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(RESULTS, "figures")
RUNS = os.path.join(RESULTS, "runs")
PAPER = os.path.join(PROJECT_ROOT, "paper_results")


def cpc18_raw_path():
    """Locate the full CPC18 raw release (Zenodo 2571510), which is not
    redistributable and so is not bundled.

    Checked in order: the CPC18_RAW environment variable, then the documented
    location inside the repo, then the layout the file lands in when the Zenodo
    archive is unzipped next to the repo. Returns None when absent -- callers
    skip the competition arms rather than failing, so the pipeline runs without
    it.
    """
    env = os.environ.get("CPC18_RAW")
    candidates = [env] if env else []
    name = "all CPC18 raw data.csv"
    candidates += [
        os.path.join(DATA, "cpc18_raw", name),
        os.path.join(DATA, "2571510", name),
        os.path.join(PROJECT_ROOT, "..", "data", "2571510", name),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None
