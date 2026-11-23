# FMEA: what can go wrong with the pose estimate

I went through each sensor and the filter itself and asked: how can this fail, what does it do to the pose, and how would I notice. Ratings are 1-10 (S = severity, O = how often it shows up in the simulator, D = how hard it is to detect, 10 = hardest). RPN = S x O x D.

| # | Failure mode | Effect on the estimate | S | O | D | RPN | What handles it now | How it's checked |
|---|---|---|---|---|---|---|---|---|
| 1 | GPS multipath fix 8-15 m off | estimate jumps toward a bad fix | 7 | 5 | 4 | 140 | Mahalanobis gate drops updates outside the 99% bound | `test_gate_rejects_far_gps_fix`, `fusion/doe.py` |
| 2 | Wheel slip (turn-rate bias) | heading drifts, position error grows without bound | 8 | 10 | 6 | 480 | gyro blended in by inverse variance; GPS and LiDAR pull it back | dead reckoning vs fused RMSE in `evaluate.py` |
| 3 | Gyro bias | slow heading drift between absolute updates | 6 | 10 | 7 | 420 | absorbed as extra process noise, not estimated | NEES of the odom+gyro setup (13.1, overconfident) |
| 4 | Filter overconfident (P too small) | real error outside the reported uncertainty, gate starts rejecting good data | 8 | 4 | 8 | 256 | Joseph-form update, small extra process noise | NEES band test on the full filter |
| 5 | Covariance loses symmetry / positive definiteness | filter blows up numerically | 9 | 2 | 3 | 54 | Joseph form | `test_covariance_stays_symmetric_positive_definite` |
| 6 | Gate rejects good GPS fixes | small accuracy loss when there are no outliers | 3 | 6 | 7 | 126 | accepted trade-off | `doe.py`: 0.56 m vs 0.58 m at 0% outliers |
| 7 | No landmark within LiDAR range | no correction from LiDAR, relies on GPS | 5 | 4 | 2 | 40 | GPS keeps running | covered by the GPS-only setup |
| 8 | LiDAR return matched to the wrong landmark | large wrong correction | 8 | 1 | 6 | 48 | not handled: the simulator gives the true landmark ID | open |
| 9 | GPS outage | drift until it comes back | 6 | 1 | 2 | 12 | not simulated yet | open |

## What the numbers point to

The two biggest RPNs are the biases (rows 2 and 3). They're always there, and nothing in the filter estimates them. That's why the next change I'd make is adding the gyro bias as a fourth state.

Row 1 was the one I could test properly. `fusion/doe.py` runs GPS outlier rate (0%, 5%, 15%) against gating off/on, 30 seeds per cell, same data for both gate settings so the comparison is paired:

| Outlier rate | RMSE, no gate | RMSE, gate | Change | paired t-test p |
|---|---|---|---|---|
| 0% | 0.56 m | 0.58 m | 3% worse | 2e-4 |
| 5% | 0.66 m | 0.57 m | 14% better | 2e-5 |
| 15% | 0.89 m | 0.60 m | 33% better | 5e-12 |

So the gate isn't free. With clean GPS it throws away a few good fixes and costs a little accuracy (row 6). Once outliers show up, it pays for itself, and the gain grows with the outlier rate (one-way ANOVA on the per-seed gains, p < 1e-18). With gating on, the error barely moves no matter how bad the GPS gets.
