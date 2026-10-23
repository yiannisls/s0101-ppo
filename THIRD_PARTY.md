# Sources and attribution

## Course infrastructure

This independent learning project is based on the
[ETH Zürich Robot Learning course repository](https://github.com/tenvinc/ethz-course-2026),
specifically [Homework 4](https://github.com/tenvinc/ethz-course-2026/tree/main/hw4_reinforcement_learning).
It is not an official ETH project.

The supplied material includes the MuJoCo environment, robot assets, neural
network definitions, rollout buffers, configuration, and training/evaluation
templates. The original assignment is preserved in `docs/course-assignment.md`;
it describes all four exercises, while this repository packages the PPO exercise.
Some submission filenames in the assignment differ from the supplied checkout.

The work completed in this study comprises the PPO action-selection, KL,
surrogate/value/entropy losses and optimizer-update TODOs, numerical validation,
baseline training, evaluation, and result documentation. The KL statistics
accumulation was also changed to use a scalar value for logging.

Source files were copied from the learner's local course checkout. That parent
repository had no commits or remote configured at packaging time, so no upstream
commit is asserted for this snapshot. SHA-256 hashes of the copied source files
are recorded in `results/baseline/summary.json`. Packaging normalizes trailing
whitespace and line endings; original and packaged hashes are both retained.

No top-level course license file was available in the local material. No blanket
license is applied to the course code in this project. Original authorship and
any applicable upstream terms remain in effect.

## Robot assets

The Standard Open Arm-100 model is provided under Apache-2.0. Its supplied
README and license are retained at:

- `assets/mujoco/trs_so_arm100/README.md`
- `assets/mujoco/trs_so_arm100/LICENSE`

The model credits [The Robot Studio's SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).

## Algorithm references

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), 2017.
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), 2015.

This follows the course's PPO variant: clipped Gaussian actions, clipped value
loss, entropy regularization, and a KL-based adaptive learning rate.
