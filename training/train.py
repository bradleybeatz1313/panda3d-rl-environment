"""
train.py
Training script for the Panda3D Navigation RL environment.
Uses Stable-Baselines3 PPO with configurable hyperparameters.

Usage:
    python -m training.train --algo ppo --timesteps 1000000
    python -m training.train --algo sac --timesteps 500000 --render
"""

import argparse
import os
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure env is registered
import env.nav_env  # noqa: F401


def make_env(render: bool = False, seed: int = 42, config: dict = None):
    """Factory function for creating the navigation environment."""
    import gymnasium as gym
    
    config = config or {}
    env = gym.make(
        "Panda3DNav-v0",
        render_mode="human" if render else None,
        arena_size=config.get("arena_size", 50.0),
        num_obstacles=config.get("num_obstacles", 15),
        num_waypoints=config.get("num_waypoints", 5),
        max_steps=config.get("max_steps", 2000),
        lidar_rays=config.get("lidar_rays", 16),
        seed=seed,
    )
    return env


def train_ppo(env, total_timesteps: int, log_dir: str):
    """Train using PPO (Proximal Policy Optimization)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    # Callbacks
    eval_env = make_env(render=False, seed=123)
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best"),
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=50000,
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="ppo_nav",
        ),
    ]
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(os.path.join(log_dir, "final_model"))
    
    eval_env.close()
    return model


def train_sac(env, total_timesteps: int, log_dir: str):
    """Train using SAC (Soft Actor-Critic)."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        policy_kwargs=dict(
            net_arch=[256, 256]
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    eval_env = make_env(render=False, seed=123)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best"),
        eval_freq=10000,
        n_eval_episodes=5,
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(os.path.join(log_dir, "final_model"))
    
    eval_env.close()
    return model


def evaluate(model, env, n_episodes: int = 10):
    """Evaluate a trained model."""
    episode_rewards = []
    episode_waypoints = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 1000)
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_waypoints.append(info.get("waypoints_reached", 0))
    
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_waypoints": float(np.mean(episode_waypoints)),
        "completion_rate": float(
            np.mean([w >= 5 for w in episode_waypoints])
        ),
    }
    
    print("\n=== Evaluation Results ===")
    print(f"  Mean Reward:     {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Mean Waypoints:  {results['mean_waypoints']:.1f} / 5")
    print(f"  Completion Rate: {results['completion_rate'] * 100:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train RL agent in Panda3D Nav")
    parser.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--eval-only", type=str, default=None,
                        help="Path to saved model for evaluation only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/{args.algo}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    env = make_env(render=args.render, seed=args.seed)
    
    if args.eval_only:
        from stable_baselines3 import PPO, SAC
        ModelClass = PPO if args.algo == "ppo" else SAC
        model = ModelClass.load(args.eval_only, env=env)
        results = evaluate(model, env)
        
        with open(os.path.join(log_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        print(f"Training {args.algo.upper()} for {args.timesteps} steps...")
        train_fn = train_ppo if args.algo == "ppo" else train_sac
        model = train_fn(env, args.timesteps, log_dir)
        
        print("\nRunning evaluation...")
        results = evaluate(model, env)
        
        with open(os.path.join(log_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    env.close()


if __name__ == "__main__":
    main()


# ─── Curriculum Helper ────────────────────────────────────────────────

def make_curriculum_env(stage: int):
    """Create env with difficulty scaled to curriculum stage."""
    obstacles = max(5, 5 + stage * 2)
    waypoints = max(2, 2 + stage)
    env = NavigationEnv(num_obstacles=obstacles, num_waypoints=waypoints)
    return env


def log_episode(writer, ep_num: int, total_reward: float, steps: int, wp_reached: int) -> None:
    """Log episode stats to TensorBoard."""
    writer.add_scalar("episode/reward", total_reward, ep_num)
    writer.add_scalar("episode/steps", steps, ep_num)
    writer.add_scalar("episode/waypoints_reached", wp_reached, ep_num)
