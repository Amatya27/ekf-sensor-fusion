import argparse, json, time
from pathlib import Path
import numpy as np
from .sim import simulate, SensorSpec, wrap
from .ekf import EKF

SETUPS = {
    "dead_reckoning": dict(use_gyro=False, gps=False, lidar=False, gate=False),
    "odom+gyro": dict(use_gyro=True, gps=False, lidar=False, gate=False),
    "odom+gyro+gps (no gating)": dict(use_gyro=True, gps=True, lidar=False, gate=False),
    "odom+gyro+gps": dict(use_gyro=True, gps=True, lidar=False, gate=True),
    "odom+gyro+gps+lidar": dict(use_gyro=True, gps=True, lidar=True, gate=True),
}


def run_filter(run, spec, use_gyro, gps, lidar, gate):
    P0 = np.diag([0.5, 0.5, 0.05]) ** 2
    f = EKF(run.truth[0] + np.array([0.3, -0.3, 0.02]), P0, spec, use_gyro=use_gyro, gate=gate)
    est, nees, t_upd = np.zeros_like(run.truth), [], []
    for k in range(len(run.t)):
        f.predict(run.odom[k, 0], run.odom[k, 1], run.gyro[k])
        t0 = time.perf_counter()
        if gps and k in run.gps:
            f.update_gps(run.gps[k])
        if lidar and k in run.lidar:
            for i, r, b in run.lidar[k]:
                f.update_landmark(run.landmarks[i], r, b)
        if (gps and k in run.gps) or (lidar and k in run.lidar):
            t_upd.append(time.perf_counter() - t0)
        est[k] = f.x
        e = run.truth[k] - f.x; e[2] = wrap(e[2])
        nees.append(float(e @ np.linalg.solve(f.P, e)))
    return est, np.array(nees), f.rejected, t_upd


def metrics(run, est):
    pos = np.linalg.norm(run.truth[:, :2] - est[:, :2], axis=1)
    hd = np.abs(wrap(run.truth[:, 2] - est[:, 2]))
    return {"pos_rmse_m": float(np.sqrt(np.mean(pos ** 2))), "final_pos_err_m": float(pos[-1]),
            "max_pos_err_m": float(pos.max()), "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(hd ** 2))))}


def main(runs=50, out="results"):
    spec = SensorSpec(); out = Path(out); out.mkdir(exist_ok=True)
    agg = {name: [] for name in SETUPS}; extra = {name: {"nees": [], "rejected": 0, "upd_us": []} for name in SETUPS}
    keep = None
    for seed in range(runs):
        run = simulate(seed=seed, spec=spec)
        for name, cfg in SETUPS.items():
            est, nees, rej, t_upd = run_filter(run, spec, **cfg)
            agg[name].append(metrics(run, est))
            extra[name]["nees"].append(float(nees.mean())); extra[name]["rejected"] += rej
            extra[name]["upd_us"] += [1e6 * x for x in t_upd]
            if seed == 0:
                keep = keep or {}; keep[name] = est
        if seed == 0:
            run0 = run
    summary = {}
    for name in SETUPS:
        m = {k: float(np.mean([r[k] for r in agg[name]])) for k in agg[name][0]}
        m["mean_nees"] = float(np.mean(extra[name]["nees"]))
        m["outliers_rejected_total"] = extra[name]["rejected"]
        if extra[name]["upd_us"]:
            m["update_median_us"] = float(np.median(extra[name]["upd_us"]))
        summary[name] = m
    result = {"runs": runs, "duration_s": 120.0, "setups": summary,
              "note": "NEES for a consistent 3-state filter averages about 3."}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(run0.truth[:, 0], run0.truth[:, 1], "k", lw=2, label="truth")
        for name in ("dead_reckoning", "odom+gyro+gps", "odom+gyro+gps+lidar"):
            ax.plot(keep[name][:, 0], keep[name][:, 1], lw=1, label=name)
        ax.scatter(*run0.landmarks.T, marker="^", c="gray", label="landmarks")
        g = np.array(list(run0.gps.values())); ax.scatter(g[:, 0], g[:, 1], s=4, c="orange", alpha=.5, label="GPS fixes")
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("Seed 0 trajectory")
        fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=120)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        for name in ("dead_reckoning", "odom+gyro", "odom+gyro+gps", "odom+gyro+gps+lidar"):
            ax.plot(run0.t, np.linalg.norm(run0.truth[:, :2] - keep[name][:, :2], axis=1), label=name)
        ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)"); ax.set_yscale("log"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "error.png", dpi=120)
    except Exception as e:
        print("couldn't save plots:", e)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=50); ap.add_argument("--out", default="results")
    r = main(ap.parse_args().runs, ap.parse_args().out)
    for name, m in r["setups"].items():
        print(f"{name:28} pos RMSE {m['pos_rmse_m']:6.2f} m  heading RMSE {m['heading_rmse_deg']:5.2f} deg  "
              f"NEES {m['mean_nees']:7.2f}  rejected {m['outliers_rejected_total']}")
