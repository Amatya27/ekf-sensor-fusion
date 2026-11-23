"""Does outlier gating actually help, and when?

Full factorial experiment: GPS outlier rate (3 levels) x gating (off/on).
Each seed gives one simulated run, and both gating settings are run on that
same data, so the comparison is paired. A paired t-test on position RMSE
checks whether the difference is real at each outlier rate.

python -m fusion.doe --seeds 30
"""
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import stats
from .sim import simulate, SensorSpec
from .evaluate import run_filter, metrics

OUTLIER_RATES = (0.0, 0.05, 0.15)


def main(seeds=30, out="results"):
    rows, table = [], {}
    for p in OUTLIER_RATES:
        spec = replace(SensorSpec(), gps_outlier_p=p)
        rmse = {False: [], True: []}
        for seed in range(seeds):
            run = simulate(seed=1000 + seed, spec=spec)
            for gate in (False, True):
                est, *_ = run_filter(run, spec, use_gyro=True, gps=True, lidar=False, gate=gate)
                r = metrics(run, est)["pos_rmse_m"]
                rmse[gate].append(r)
                rows.append({"outlier_rate": p, "gate": gate, "seed": seed, "pos_rmse_m": r})
        off, on = np.array(rmse[False]), np.array(rmse[True])
        t, pval = stats.ttest_rel(off, on)
        diff = off - on
        table[str(p)] = {"rmse_no_gate_m": float(off.mean()), "rmse_gate_m": float(on.mean()),
                         "mean_improvement_m": float(diff.mean()),
                         "improvement_pct": float(100 * diff.mean() / off.mean()),
                         "t_stat": float(t), "p_value": float(pval),
                         "seeds_where_gate_helped": int((diff > 0).sum())}
    # interaction: does the benefit of gating grow with the outlier rate?
    gains = [np.array([r["pos_rmse_m"] for r in rows if r["outlier_rate"] == p and not r["gate"]]) -
             np.array([r["pos_rmse_m"] for r in rows if r["outlier_rate"] == p and r["gate"]]) for p in OUTLIER_RATES]
    f, p_inter = stats.f_oneway(*gains)
    result = {"seeds_per_cell": seeds, "factors": {"gps_outlier_rate": list(OUTLIER_RATES), "gating": [False, True]},
              "by_outlier_rate": table, "interaction_anova": {"F": float(f), "p_value": float(p_inter)}}
    Path(out).mkdir(exist_ok=True)
    Path(out, "doe.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    r = main(a.seeds, a.out)
    print(f"{'outlier rate':>12} {'no gate':>9} {'gate':>9} {'better':>8} {'p':>10}  gate helped")
    for p, m in r["by_outlier_rate"].items():
        print(f"{float(p):>12.2f} {m['rmse_no_gate_m']:8.2f}m {m['rmse_gate_m']:8.2f}m {m['improvement_pct']:7.1f}% "
              f"{m['p_value']:10.1e}  {m['seeds_where_gate_helped']}/{r['seeds_per_cell']}")
    print(f"interaction (does the gain grow with outliers): F={r['interaction_anova']['F']:.1f}, "
          f"p={r['interaction_anova']['p_value']:.1e}")
