"""Render three frames from the first seeded evaluation episode."""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ppo-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.so100_rl_env import SO100RLEnv
from exercises.ex3_ppo import PPOAgent
from exercises.ex3_ppo_config import PPO_PARAMETERS
from rl.common import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    env = SO100RLEnv(ROOT / 'assets/mujoco/so100_pos_ctrl.xml')
    agent = PPOAgent(19, 6, PPO_PARAMETERS['hidden_sizes'])
    agent.load(args.model_path)
    agent.eval_mode()
    obs, _ = env.reset()
    mujoco.mj_forward(env.model, env.data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.15, 0, 0.18]
    camera.distance = 0.85
    camera.azimuth = 135
    camera.elevation = -22
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    out = ROOT / 'media'
    out.mkdir(exist_ok=True)
    frames, records = [], []
    try:
        for step in range(env.max_episode_length + 1):
            if step in (0, 5, env.max_episode_length):
                renderer.update_scene(env.data, camera=camera)
                frame = renderer.render().copy()
                frames.append(frame)
                error = float(np.linalg.norm(env.data.site('ee_site').xpos - env.data.mocap_pos[0]))
                records.append({'step': step, 'time_s': step * env.ctrl_timestep, 'error_m': error})
                plt.imsave(out / f'reaching-step-{step:02d}.png', frame)
            if step == env.max_episode_length:
                break
            with torch.inference_mode():
                action = agent.predict_action(torch.as_tensor(obs).unsqueeze(0)).squeeze(0).numpy()
            obs, _, _, _, _ = env.step(action)
    finally:
        renderer.close()
        env.close()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, frame, record, label in zip(axes, frames, records, ['Initial state', 'Approach', 'End of episode']):
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f"{label} | {record['time_s']:.1f} s\nEE distance: {record['error_m'] * 100:.2f} cm", fontsize=12)
    fig.savefig(out / 'reaching-sequence.png', dpi=160)
    plt.close(fig)
    (out / 'reaching-sequence.json').write_text(json.dumps({
        'checkpoint': args.model_path.name, 'seed': args.seed, 'episode': 1,
        'device': 'cpu', 'policy': 'deterministic Gaussian mean, clipped',
        'frames': records, 'note': 'First episode for this seed; illustrative rollout, not aggregate evaluation.'
    }, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
