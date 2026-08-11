"""Build every provenance arm and cache to data/processed.

Usage: python scripts/00_prepare_data.py --c13k-dir /path/to/choices13k-main
"""
import _boot  # noqa
import argparse
import json

from src.data import prepare_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--c13k-dir", required=True,
                    help="folder with c13k_selections.csv + c13k_problems.json")
    args = ap.parse_args()
    meta = prepare_all(args.c13k_dir)
    print(json.dumps(meta, indent=2))
    print("\nArm sizes / noise floors written to data/processed/meta.json")
