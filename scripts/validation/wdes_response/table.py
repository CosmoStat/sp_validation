import numpy as np
import yaml

W = "/n23data1/mkilbing/imsim_wdes_response/grid_%d/%s/results/m_bias_results.yaml"
O = "/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow/grid_%d/results/m_bias_results.yaml"


def get(y, meth, w):
    return y["weights"][w] if meth == "split" else y["joint"]["weights"][w]


# check iv reproduces original
for g in (1, 2, 3, 4):
    a = yaml.safe_load(open(W % (g, "iv")))
    b = yaml.safe_load(open(O % g))
    d = max(
        abs(get(a, m, "w_iv")[k] - get(b, m, "w_iv")[k])
        for m in ("split", "joint")
        for k in ("m1", "m2", "m1_err_jk", "m2_err_jk")
    )
    print(f"iv check grid_{g}: max diff {d:.1e}")
rows = [
    ("w_iv, R w_iv (existing)", "iv", "w_iv"),
    ("w_iv, R w_iv per branch (fit)", "iv_d", "w_iv"),
    ("w_iv, R w_iv per branch (boxes)", "iv_box", "w_iv"),
    ("w_des, R w_iv (existing)", "iv", "w_des"),
    ("w_des, R w_des fixed (c)", "des_c", "w_des"),
    ("w_des, R w_des per branch (d)", "des_d", "w_des"),
]
for meth in ("split", "joint"):
    print(f"\n== {'R_shear + R_sel' if meth == 'split' else 'R_joint'}")
    for lab, s, w in rows:
        ys = [yaml.safe_load(open(W % (g, s))) for g in (1, 2, 3, 4)]
        print(f"  {lab}")
        for k in ("m1", "m2"):
            x = np.array([get(y, meth, w)[k] for y in ys])
            e = np.array([get(y, meth, w)[k + "_err_jk"] for y in ys])
            iv = 1 / e**2
            mu = (iv * x).sum() / iv.sum()
            s_ = iv.sum() ** -0.5
            ch = ((x - mu) ** 2 * iv).sum()
            per = "  ".join(f"{v:+.4f}+/-{u:.4f}" for v, u in zip(x, e))
            print(f"    {k}  {per}   mean {mu:+.4f} +/- {s_:.4f}  chi2 {ch:.1f}/3")

# differences with their own jackknife error (diff_jk.py)
D = "/n23data1/mkilbing/imsim_wdes_response/grid_%d/diff_%s_%s.yaml"
for b, a, w in (
    ("iv_d", "iv", "w_iv"),
    ("iv_box", "iv", "w_iv"),
    ("des_d", "des_c", "w_des"),
):
    print(f"\n== {b} - {a}, weight {w}")
    for meth in ("default", "joint"):
        ys = [yaml.safe_load(open(D % (g, b, a)))[meth][w] for g in (1, 2, 3, 4)]
        for k in ("dm1", "dm2"):
            x = np.array([y[k] for y in ys])
            e = np.array([y[k + "_err_jk"] for y in ys])
            iv = 1 / e**2
            mu = (iv * x).sum() / iv.sum()
            s_ = iv.sum() ** -0.5
            ch = ((x - mu) ** 2 * iv).sum()
            per = "  ".join(f"{v:+.4f}+/-{u:.4f}" for v, u in zip(x, e))
            print(
                f"  {meth:7s} {k}  {per}   mean {mu:+.4f} +/- {s_:.4f}  chi2 {ch:.1f}/3"
            )
