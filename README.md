# 🤖 RL Navigation Environment — Panda3D

A Gymnasium-compatible 3D navigation environment built on Panda3D for training reinforcement learning agents. Agents navigate procedurally generated obstacle fields to reach sequential waypoints using continuous control.

![Panda3D](https://img.shields.io/badge/Panda3D-1.10-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-purple)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Features

### Environment (`env/nav_env.py`)
- **Gymnasium v0.29+ compatible** — Standard `reset()` / `step()` API with proper termination/truncation
- **24-dim observation space** — Agent state (position, velocity, heading), 16-ray lidar, target direction + distance
- **4-dim continuous action space** — Forward/back, strafe, turn, jump
- **Procedural generation** — Random obstacle placement and waypoint selection each episode
- **Reward shaping** — Distance reduction bonus, waypoint completion reward, collision penalty, time pressure
- **Lidar system** — Ray-circle intersection for efficient obstacle detection without physics engine overhead
- **Panda3D rendering** — Optional visual mode for debugging and demonstration

### Training (`training/train.py`)
- **PPO and SAC** — Two algorithm options via Stable-Baselines3
- **Evaluation callbacks** — Periodic eval with best-model checkpointing
- **TensorBoard logging** — Full training metrics visualization
- **CLI interface** — `--algo`, `--timesteps`, `--render`, `--eval-only`, `--seed`

---

## 📂 Project Structure

```
panda3d-rl-environment/
├── env/
│   ├── __init__.py
│   └── nav_env.py           # Gymnasium environment (24-obs, 4-act)
├── training/
│   └── train.py             # SB3 training with PPO/SAC
├── configs/                  # Hyperparameter configs
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train with PPO (default)
python -m training.train --algo ppo --timesteps 500000

# Train with SAC
python -m training.train --algo sac --timesteps 500000

# Visual mode
python -m training.train --algo ppo --timesteps 100000 --render

# Evaluate a saved model
python -m training.train --algo ppo --eval-only runs/ppo_*/final_model

# Monitor training
tensorboard --logdir runs/
```

---

## 🧪 Environment Details

### Observation Space (24-dim)
| Index | Feature | Range |
|-------|---------|-------|
| 0-2 | Agent position (normalized) | [-1, 1] |
| 3-5 | Agent velocity | [-1, 1] |
| 6-7 | Heading (sin, cos) | [-1, 1] |
| 8-23 | 16-ray lidar distances | [0, 1] |
| 24-26 | Target direction | [-1, 1] |
| 27 | Target distance (normalized) | [0, 1] |

### Action Space (4-dim continuous)
| Index | Action | Range |
|-------|--------|-------|
| 0 | Forward/backward | [-1, 1] |
| 1 | Strafe left/right | [-1, 1] |
| 2 | Turn | [-1, 1] |
| 3 | Jump (>0.5 triggers) | [-1, 1] |

### Reward Structure
| Event | Reward |
|-------|--------|
| Per timestep | -1.0 |
| Distance reduction | +10.0 × Δd |
| Waypoint reached | +100.0 |
| All waypoints cleared | +500.0 |
| Obstacle collision | -50.0 |
| Fall off platform | -100.0 |

---

## 🔬 AI Research Applications

- **Spatial reasoning** — 3D navigation with continuous control in procedural environments
- **Sim-to-real** — Lidar observation model mirrors real-world sensor modalities
- **Curriculum learning** — Increase `num_obstacles` / `num_waypoints` / `arena_size` progressively
- **Multi-agent extension** — Environment architecture supports multiple simultaneous agents
- **Reward engineering** — Modular reward components for ablation studies

---

## 📄 License

MIT

---

## Quick Start

    pip install -r requirements.txt
    python training/train.py

To run with visual rendering:

    python -c "from env.nav_env import NavigationEnv; e = NavigationEnv(render_mode='human'); e.reset(); [e.step(e.action_space.sample()) for _ in range(500)]"

---

## Observation Space

| Indices | Meaning | Range |
|---------|---------|-------|
| 0-2 | Agent position (normalized) | [-1, 1] |
| 3-5 | Agent velocity | [-1, 1] |
| 6-7 | Heading (sin, cos) | [-1, 1] |
| 8-23 | Lidar rays (16) | [0, 1] |
| 24-26 | Target direction + altitude | [-1, 1] |
| 27 | Distance to target (normalized) | [0, 1] |

---

## Training

Uses **PPO** from Stable-Baselines3 by default:

    from stable_baselines3 import PPO
    from env.nav_env import NavigationEnv

    env = NavigationEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=500_000)
    model.save("panda3d_nav_ppo")

<!-- env version: 1.1.0 | last updated 2026-04-28 -->
