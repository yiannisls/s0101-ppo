# PPO baseline: 2026-10-23

## Protocol

- Environment: SO100 end-effector tracking, six position commands, 19 observations.
- Training seed: 42; evaluation seed: 42 (global Python/NumPy/PyTorch seeding).
- Configuration: supplied PPO defaults, retained in `exercises/ex3_ppo_config.py`
  and `results/baseline/training-config.json`.
- Training: 500 iterations × 2,048 interactions = 1,024,000 environment steps.
- Device: NVIDIA GeForce RTX 4060 Laptop GPU.
- Retained checkpoint: `26_10_23_15_26_57_model/iter_500.pt`.
- Selection: final checkpoint, without a checkpoint sweep.
- Evaluation: deterministic actor mean, 100 randomized episodes of 30 steps.

The environment reset helpers use global NumPy randomness. Reproducible runs
use `set_seed`; passing a seed to `env.reset` alone does not seed those helpers.
Reproduction across platforms and GPU library versions is not guaranteed.

## Validation and observations

A three-iteration smoke run completed with finite training losses and evaluation
metrics. A separate robot reset/step check returned a 19-dimensional observation.
Tests verify Gaussian KL, clipping for positive and negative advantages, value
clipping, entropy sign, action shapes, and a complete synthetic PPO update.

The baseline curves show a rapid initial rise in return and fall in tracking
error, followed by slower improvement and evaluation fluctuations. Each training
iteration evaluates five random episodes. These curves are diagnostic monitoring
data, rather than an independent final test set.

GUI playback was observed by the learner: fast approach, some approach
oscillation, and settling near the target. This is qualitative evidence; neither
settling time nor oscillation amplitude was measured.

## Recorded evaluation

| Metric | Value |
|---|---:|
| Episodes | 100 |
| Mean return | 53.080 |
| Return standard deviation | 10.928 |
| Median return | 52.337 |
| Minimum / maximum return | 18.168 / 68.983 |
| Mean episode-averaged tracking error | 0.017480 m |
| Episode-error standard deviation | 0.012824 m |
| Minimum / maximum episode-averaged error | 0.004514 / 0.085964 m |

The first 20 episodes were evaluated earlier and are included in this run.
All standard deviations describe episode variation, not variation between
independent training runs. The errors include approach motion and do not measure
steady-state accuracy alone. No generalization or pick-and-place claim is made.

The published terminal output is preserved apart from removal of a local absolute
checkpoint-path prefix. Episode data and scalar curves are exported as JSON/CSV.
Raw event files and model checkpoints remain local.

## Next experiments

Compare selected saved checkpoints under a fixed screening protocol, followed by
fresh-seed confirmation. Measure transient versus final-step errors separately
before attempting to reduce oscillation. Pick-and-place requires a separate
environment, reward, and training experiment.
