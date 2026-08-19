"""Property self-tests for the metrics package (no MATLAB available on cluster).

Validates documented invariants rather than bit-matching MATLAB:
  * SSIM/MS-SSIM of identical images == 1
  * entropy of a constant image == 0
  * MI of identical images == entropy
  * CC of identical images == 1
  * correct higher/lower-is-better direction (a clean fuse beats a noisy fuse)
  * Qabf / VIF in sane ranges
Run:  venv/bin/python selftest_metrics.py
"""
import numpy as np
import metrics as M


def _rng(seed=0):
    return np.random.default_rng(seed)


def main():
    rng = _rng(0)
    H = W = 128
    A = rng.uniform(0, 255, (H, W))
    B = gaussian = rng.uniform(0, 255, (H, W))
    F_good = (A + B) / 2                       # clean average fuse
    F_noisy = np.clip(F_good + rng.normal(0, 40, (H, W)), 0, 255)

    checks = []

    def chk(name, cond, extra=""):
        checks.append((name, bool(cond), extra))

    # identity properties
    chk("SSIM(X,X)==1", abs(M.ssim_pair(A, A, A) - 1.0) < 1e-6,
        f"{M.ssim_pair(A, A, A):.6f}")
    chk("MS_SSIM(X,X)==1", abs(M.ms_ssim_pair(A, A, A) - 1.0) < 1e-4,
        f"{M.ms_ssim_pair(A, A, A):.6f}")
    chk("CC(X,X)==1", abs(M.cc(A, A, A) - 1.0) < 1e-6, f"{M.cc(A, A, A):.6f}")

    const = np.full((H, W), 100.0)
    chk("EN(const)==0", abs(M.en(const)) < 1e-9, f"{M.en(const):.6e}")
    chk("SD(const)==0", abs(M.sd(const)) < 1e-9, f"{M.sd(const):.6e}")
    chk("SF(const)==0", abs(M.sf(const)) < 1e-9, f"{M.sf(const):.6e}")
    chk("AG(const)==0", abs(M.ag(const)) < 1e-9, f"{M.ag(const):.6e}")

    # MI of identical image equals its entropy (per-source), so fusion MI = 2*EN
    chk("MI(X,X,X)==2*EN(X)", abs(M.mi(A, A, A) - 2 * M.en(A)) < 1e-6,
        f"{M.mi(A, A, A):.4f} vs {2*M.en(A):.4f}")

    # direction: clean fuse should beat noisy fuse on quality metrics
    chk("Qabf good>noisy", M.qabf(A, B, F_good) > M.qabf(A, B, F_noisy),
        f"{M.qabf(A,B,F_good):.4f} > {M.qabf(A,B,F_noisy):.4f}")
    chk("SSIM good>noisy", M.ssim_pair(A, B, F_good) > M.ssim_pair(A, B, F_noisy))
    chk("Nabf noisy>good (lower=better)", M.nabf(A, B, F_noisy) > M.nabf(A, B, F_good),
        f"noisy {M.nabf(A,B,F_noisy):.4f} > good {M.nabf(A,B,F_good):.4f}")
    chk("VIF good>noisy", M.vif_pair(A, B, F_good) > M.vif_pair(A, B, F_noisy),
        f"{M.vif_pair(A,B,F_good):.4f} > {M.vif_pair(A,B,F_noisy):.4f}")

    # range sanity
    q = M.qabf(A, B, F_good)
    chk("Qabf in [0,1]", 0 <= q <= 1, f"{q:.4f}")
    chk("VIF >= 0", M.vif_pair(A, B, F_good) >= 0)

    # compute_all returns finite for all keys on a normal fuse
    allm = M.compute_all(A, B, F_good)
    finite = all(np.isfinite(v) for v in allm.values())
    chk("compute_all finite", finite, str({k: round(v, 3) for k, v in allm.items()}))

    print("\n=== metric self-test ===")
    npass = 0
    for name, ok, extra in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({extra})" if extra else ""))
        npass += ok
    print(f"\n{npass}/{len(checks)} checks passed")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
