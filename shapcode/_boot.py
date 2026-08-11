"""Make both this package and the main `code/` tree importable.

shapcode sits beside code/ rather than inside it: the attribution readout is a
self-contained analysis with its own dependency (shap's semantics, though not
the library's plotting) and its own correctness test, and keeping it separate
means the main pipeline neither imports it nor breaks when it changes. It does
depend on code/ for the data arms, models and shared figure styling, so both
directories go on the path.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (os.path.join(_ROOT, "code"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
