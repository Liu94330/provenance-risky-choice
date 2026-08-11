"""The lottery-expansion rules must reproduce the published distributions.

CPC15 and the CPC18 competition problems only give each option compactly as
(H, pH, L, LotShape, LotNum). Using them at all requires expanding that into an
explicit outcome distribution, and the expansion rules are not stated in a form
we could copy -- they were reverse-engineered. The CPC18 estimation set ships
with all 420 option distributions already expanded, so it is a ground truth to
check the rules against, and this test does exactly that.

If this test fails, every arm built from raw CPC files is suspect.

Usage: python tests/test_lottery_expansion.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))

from src.paths import EXTERNAL  # noqa: E402

import pandas as pd

from src.problems import expand_lottery

# The shipped CSV carries ~6 significant digits, so exact equality would fail
# on rounding alone (3.90625e-05 is stored as 3.91e-05). This tolerance is
# loose enough for that and far tighter than any rule error would be.
PROB_TOL = 2e-7


def published(row, side):
    agg = {}
    for k in range(1, 11):
        v, p = row[f"{side}v{k}"], row[f"{side}p{k}"]
        if pd.notna(v) and pd.notna(p) and float(p) > 0:
            agg[round(float(v), 6)] = agg.get(round(float(v), 6), 0.0) \
                + float(p)
    xs = sorted(agg)
    return xs, [agg[x] for x in xs]


def main():
    fd = pd.read_csv(os.path.join(EXTERNAL, "CPC18_EstSet_fullDist.csv"))
    checked = failures = 0
    for _, r in fd.iterrows():
        for side, H, pH, L, shape, num in [
                ("a", r["Ha"], r["pHa"], r["La"], r["LotShapeA"],
                 r["LotNumA"]),
                ("b", r["Hb"], r["pHb"], r["Lb"], r["LotShapeB"],
                 r["LotNumB"])]:
            want_x, want_p = published(r, side)
            if not want_x:
                continue
            checked += 1
            got_x, got_p = expand_lottery(H, pH, L, shape, num)
            same = (len(got_x) == len(want_x)
                    and all(abs(a - b) < 1e-6
                            for a, b in zip(got_x, want_x))
                    and all(abs(a - b) < PROB_TOL
                            for a, b in zip(got_p, want_p)))
            if not same:
                failures += 1
                if failures <= 5:
                    print(f"MISMATCH GameID={r['GameID']} side={side} "
                          f"shape={shape} num={num} H={H} pH={pH} L={L}")
                    print(f"  expanded : {list(zip(got_x, got_p))}")
                    print(f"  published: {list(zip(want_x, want_p))}")

    print(f"lottery expansion: {checked - failures}/{checked} option "
          f"distributions match the published CPC18 set")
    if failures:
        raise SystemExit(f"FAILED: {failures} mismatches")
    print("Lottery expansion test passed.")


if __name__ == "__main__":
    main()
