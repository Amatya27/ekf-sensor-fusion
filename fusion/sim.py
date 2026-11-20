from dataclasses import dataclass, field
import numpy as np

DT = 0.02


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class SensorSpec:
    odom_v_std: float = 0.05
    odom_w_std: float = 0.05
    odom_w_bias: float = 0.02  # wheel slip
    gyro_std: float = 0.01
    gyro_bias: float = 0.003
    gps_std: float = 1.5
    gps_rate: float = 2.0
    gps_outlier_p: float = 0.05  # multipath
    lidar_range_std: float = 0.10
    lidar_bearing_std: float = 0.02
    lidar_rate: float = 5.0
    lidar_max_range: float = 12.0


@dataclass
class Run:
    t: np.ndarray
    truth: np.ndarray
    odom: np.ndarray
    gyro: np.ndarray
    gps: dict = field(default_factory=dict)
    lidar: dict = field(default_factory=dict)
    landmarks: np.ndarray = None


def commands(t):
    v = 1.0 + 0.3 * np.sin(0.05 * t)
    w = 0.25 * np.sin(0.1 * t) + 0.1 * np.sign(np.sin(0.02 * t))
    return v, w


def simulate(seed=0, duration=120.0, spec=SensorSpec(), landmarks=None):
    rng = np.random.default_rng(seed)
    if landmarks is None:
        landmarks = np.array([[x, y] for x in (-20, 0, 20, 40) for y in (-20, 0, 20, 40)], float)
    n = int(duration / DT)
    t = np.arange(n) * DT
    truth = np.zeros((n, 3)); odom = np.zeros((n, 2)); gyro = np.zeros(n)
    run = Run(t, truth, odom, gyro, landmarks=landmarks)
    gps_every, lidar_every = int(1 / (spec.gps_rate * DT)), int(1 / (spec.lidar_rate * DT))
    for k in range(n):
        v, w = commands(t[k])
        if k:
            x, y, th = truth[k - 1]
            truth[k] = [x + v * np.cos(th) * DT, y + v * np.sin(th) * DT, wrap(th + w * DT)]
        odom[k] = [v + rng.normal(0, spec.odom_v_std), w + spec.odom_w_bias + rng.normal(0, spec.odom_w_std)]
        gyro[k] = w + spec.gyro_bias + rng.normal(0, spec.gyro_std)
        if k % gps_every == 0 and k:
            fix = truth[k, :2] + rng.normal(0, spec.gps_std, 2)
            if rng.random() < spec.gps_outlier_p:
                ang = rng.uniform(0, 2 * np.pi)
                fix = truth[k, :2] + rng.uniform(8, 15) * np.array([np.cos(ang), np.sin(ang)])
            run.gps[k] = fix
        if k % lidar_every == 0 and k:
            obs = []
            for i, (lx, ly) in enumerate(landmarks):
                dx, dy = lx - truth[k, 0], ly - truth[k, 1]
                r = np.hypot(dx, dy)
                if r < spec.lidar_max_range:
                    b = wrap(np.arctan2(dy, dx) - truth[k, 2])
                    obs.append((i, r + rng.normal(0, spec.lidar_range_std), wrap(b + rng.normal(0, spec.lidar_bearing_std))))
            run.lidar[k] = obs
    return run
