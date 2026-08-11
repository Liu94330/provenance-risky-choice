"""The numpy CPT (used for theory fitting and the synthetic agent) and the
torch CPT (the trainable CPT_NET) must be the same function. Run:
    python tests/test_cpt_consistency.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..")))
import numpy as np
import torch

from src.models import CPTNet
from src.problems import SCALE, cpt_rate, problem_arrays
from src.probes import build_probe_battery


def random_problems(n=200, seed=0):
    rng = np.random.default_rng(seed)
    plist = []
    for i in range(n):
        ka, kb = rng.integers(1, 6), rng.integers(1, 10)
        for _ in range(3):
            pa = rng.dirichlet(np.ones(ka))
            pb = rng.dirichlet(np.ones(kb))
            if pa.min() > 1e-3 and pb.min() > 1e-3:
                break
        xa = rng.uniform(-50, 256, ka).round(1)
        xb = rng.uniform(-50, 256, kb).round(1)
        plist.append(dict(A=(xa, pa), B=(xb, pb), corr=0, pid=i))
    return problem_arrays(plist)


def torch_rate(arrs, theta):
    net = CPTNet()
    with torch.no_grad():
        for k, (lo, hi) in net.BOUNDS.items():
            t = (theta[k] - lo) / (hi - lo)
            net.raw[k].copy_(torch.logit(torch.tensor(t)))
    from src.problems import set_features
    slots, mask, corr = set_features(arrs)
    with torch.no_grad():
        logit = net(torch.as_tensor(slots), torch.as_tensor(mask),
                    torch.as_tensor(corr))
    return torch.sigmoid(logit).numpy()


if __name__ == "__main__":
    theta = dict(alpha=0.88, beta=0.92, lam=2.25, gamma_pos=0.65,
                 gamma_neg=0.7, tau=6.0)
    vec = (theta["alpha"], theta["beta"], theta["lam"],
           theta["gamma_pos"], theta["gamma_neg"], theta["tau"])

    for name, arrs in [("random", random_problems()),
                       ("probes", build_probe_battery()[0])]:
        r_np = cpt_rate(arrs, vec)
        r_t = torch_rate(arrs, theta)
        err = np.max(np.abs(r_np - r_t))
        assert err < 1e-4, f"{name}: numpy vs torch CPT differ by {err}"
        print(f"{name}: max |numpy - torch| = {err:.2e}  OK "
              f"(n={len(r_np)}, mean rate {r_np.mean():.3f})")
    print("CPT consistency test passed.")
