# EKF Sensor Fusion for a Ground Robot

I built this to work through state estimation end to end: how a robot keeps track of where it is when every sensor it has is wrong in a different way.

The robot is a simulated differential-drive base. It has four sensors:

| Sensor | Rate | What goes wrong |
|---|---|---|
| Wheel odometry | 50 Hz | noisy, and a small turn-rate bias (wheel slip) |
| Gyro | 50 Hz | less noisy than the wheels, but it has its own bias |
| GPS | 2 Hz | 1.5 m noise, and about 5% of fixes are multipath outliers 8-15 m off |
| LiDAR | 5 Hz | range and bearing to known landmarks, only within 12 m |

An Extended Kalman Filter (EKF) tracks the pose `[x, y, theta]`:

- **Predict:** unicycle motion model. Wheel speed drives it, and the turn rate is a mix of gyro and wheel readings weighted by how much I trust each one (inverse variance).
- **Update:** GPS position (linear) and LiDAR range/bearing to landmarks (nonlinear, so it uses the Jacobian).
- **Outlier gating:** every update is checked against its Mahalanobis distance first. If it's outside the 99% bound, it gets dropped instead of pulling the estimate off. This is what handles the bad GPS fixes.
- Covariance updates use the Joseph form so the matrix stays symmetric and positive definite over long runs.

## Results

50 Monte Carlo runs, 120 s each, same sensor noise, different random seeds (`results/results.json`):

| Setup | Position RMSE | Heading RMSE | Mean NEES |
|---|---|---|---|
| Wheel odometry only (dead reckoning) | 30.24 m | 81.1 deg | 495.5 |
| Odometry + gyro | 6.30 m | 15.5 deg | 13.1 |
| + GPS, no outlier gating | 0.73 m | 5.4 deg | 8.5 |
| + GPS, with gating | 0.62 m | 5.4 deg | 6.8 |
| + GPS + LiDAR landmarks | **0.30 m** | **2.3 deg** | **3.7** |

What I took from it:

- Fusing everything cut position error about 100x compared with wheel odometry alone.
- Gating the GPS fixes alone lowered error from 0.73 m to 0.62 m. The ungated filter trusts every multipath jump.
- NEES checks whether the filter's own uncertainty is honest. For 3 states it should average around 3. The full filter sits at 3.7, so it's close to consistent. The setups without absolute measurements are overconfident because the sensor biases aren't in the model.
- A measurement update takes about 30 microseconds (median), so this is nowhere near a bottleneck at these rates.

![trajectory](results/trajectory.png)
![error over time](results/error.png)

## Run it

```bash
pip install -r requirements.txt
python -m fusion.evaluate --runs 50   # prints the table, writes results/ and plots
python -m pytest -q                    # 5 tests
```

The tests check that the landmark Jacobian matches a numerical derivative, that fusion beats dead reckoning by a wide margin, that the full filter stays consistent (NEES band), that the gate drops a far-off GPS fix, and that the covariance stays symmetric positive definite.

## Limits (on purpose, for now)

- Everything is simulated in 2D. No real robot or recorded dataset yet.
- Landmark positions are known and so is which landmark each LiDAR return belongs to (no data association problem).
- Sensor biases are not estimated as states; they're absorbed as extra process noise.

Next steps I'd like to try: estimate the gyro bias as a state, run it on a public dataset with real IMU and GPS logs, and wrap it as a ROS 2 node that publishes the pose.

## Layout

```
fusion/sim.py       robot + sensor simulator
fusion/ekf.py       the filter (predict, GPS update, landmark update, gating)
fusion/evaluate.py  Monte Carlo comparison, results.json, plots
tests/test_ekf.py   correctness and consistency tests
```
