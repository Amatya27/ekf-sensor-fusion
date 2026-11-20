import numpy as np
from fusion.sim import simulate, SensorSpec, wrap
from fusion.ekf import EKF
from fusion.evaluate import run_filter, metrics

SPEC = SensorSpec()


def test_landmark_jacobian_matches_numeric():
    x, lm, eps = np.array([1.0, -2.0, 0.4]), np.array([6.0, 3.0]), 1e-6
    _, H = EKF.landmark_model(x, lm)
    num = np.zeros((2, 3))
    for i in range(3):
        d = np.zeros(3); d[i] = eps
        hp, _ = EKF.landmark_model(x + d, lm); hm, _ = EKF.landmark_model(x - d, lm)
        diff = hp - hm; diff[1] = wrap(diff[1])
        num[:, i] = diff / (2 * eps)
    assert np.allclose(H, num, atol=1e-5)


def test_fusion_beats_dead_reckoning():
    run = simulate(seed=3, duration=60.0)
    dr, *_ = run_filter(run, SPEC, use_gyro=False, gps=False, lidar=False, gate=False)
    full, *_ = run_filter(run, SPEC, use_gyro=True, gps=True, lidar=True, gate=True)
    assert metrics(run, full)["pos_rmse_m"] < 0.2 * metrics(run, dr)["pos_rmse_m"]


def test_full_filter_is_consistent():
    # NEES should be around 3 for 3 states, a bit higher is fine since biases aren't modeled
    nees = [run_filter(simulate(seed=s, duration=60.0), SPEC, True, True, True, True)[1].mean() for s in range(5)]
    assert 1.0 < np.mean(nees) < 6.0


def test_gate_rejects_far_gps_fix():
    f = EKF([0, 0, 0], np.diag([0.5, 0.5, 0.05]) ** 2, SPEC)
    assert f.update_gps([0.5, -0.4]) is True
    assert f.update_gps([14.0, 0.0]) is False and f.rejected == 1


def test_covariance_stays_symmetric_positive_definite():
    run = simulate(seed=1, duration=30.0)
    f = EKF(run.truth[0], np.eye(3) * 0.1, SPEC)
    for k in range(len(run.t)):
        f.predict(run.odom[k, 0], run.odom[k, 1], run.gyro[k])
        if k in run.lidar:
            for i, r, b in run.lidar[k]:
                f.update_landmark(run.landmarks[i], r, b)
    assert np.allclose(f.P, f.P.T) and np.all(np.linalg.eigvalsh(f.P) > 0)
