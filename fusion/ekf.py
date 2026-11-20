import numpy as np
from .sim import wrap, DT

CHI2_2DOF_99 = 9.21


class EKF:
    def __init__(self, x0, P0, spec, use_gyro=True, gate=True):
        self.x, self.P, self.spec = np.array(x0, float), np.array(P0, float), spec
        self.use_gyro, self.gate = use_gyro, gate
        self.rejected = 0

    def turn_rate(self, w_odom, w_gyro):
        if not self.use_gyro:
            return w_odom, self.spec.odom_w_std ** 2
        a, b = 1 / self.spec.odom_w_std ** 2, 1 / self.spec.gyro_std ** 2
        return (a * w_odom + b * w_gyro) / (a + b), 1 / (a + b)

    def predict(self, v, w_odom, w_gyro, dt=DT):
        w, var_w = self.turn_rate(w_odom, w_gyro)
        x, y, th = self.x
        self.x = np.array([x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, wrap(th + w * dt)])
        F = np.array([[1, 0, -v * np.sin(th) * dt], [0, 1, v * np.cos(th) * dt], [0, 0, 1]])
        G = np.array([[np.cos(th) * dt, 0], [np.sin(th) * dt, 0], [0, dt]])
        # small extra noise on the turn rate since the biases aren't modeled
        Q = G @ np.diag([self.spec.odom_v_std ** 2, var_w + 0.01 ** 2]) @ G.T + np.diag([1e-4, 1e-4, 1e-6])
        self.P = F @ self.P @ F.T + Q

    def _update(self, z, h, H, R, angle_idx=None):
        y = z - h
        if angle_idx is not None:
            y[angle_idx] = wrap(y[angle_idx])
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        if self.gate and d2 > CHI2_2DOF_99:
            self.rejected += 1
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form
        return True

    def update_gps(self, z):
        H = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        return self._update(np.asarray(z, float), self.x[:2], H, np.eye(2) * self.spec.gps_std ** 2)

    @staticmethod
    def landmark_model(x, lm):
        dx, dy = lm[0] - x[0], lm[1] - x[1]
        q = dx * dx + dy * dy; r = np.sqrt(q)
        h = np.array([r, wrap(np.arctan2(dy, dx) - x[2])])
        H = np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])
        return h, H

    def update_landmark(self, lm, rng, bearing):
        h, H = self.landmark_model(self.x, lm)
        R = np.diag([self.spec.lidar_range_std ** 2, self.spec.lidar_bearing_std ** 2])
        return self._update(np.array([rng, bearing]), h, H, R, angle_idx=1)
