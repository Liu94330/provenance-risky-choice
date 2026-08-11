"""Cross-provenance transfer matrix (train row -> test column), with
noise-ceiling-normalised completeness."""
import _boot  # noqa
import json
import os

import pandas as pd

from src.analysis import transfer_table
from src.data import PROCESSED
from src.train import RESULTS

if __name__ == "__main__":
    runs = pd.read_csv(os.path.join(RESULTS, "runs_main.csv"))
    with open(os.path.join(PROCESSED, "meta.json")) as f:
        meta = json.load(f)
    tt = transfer_table(runs, meta)
    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    out = os.path.join(RESULTS, "tables", "transfer.csv")
    tt.to_csv(out, index=False)
    piv = tt.groupby(["train", "test"])["completeness"].mean().unstack()
    print("Completeness (mean over archs & seeds):")
    print(piv.round(3).to_string())
    print("wrote", out)
