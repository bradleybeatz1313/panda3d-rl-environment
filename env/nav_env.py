"""
nav_env.py
OpenAI Gymnasium-compatible 3D navigation environment built on Panda3D.

An agent must navigate through procedurally generated obstacle fields
to reach target waypoints. Designed as a training environment for
reinforcement learning research in 3D spatial reasoning.

Observation space: 24-dim continuous (agent state + lidar + target)
Action space: 4-dim continuous (forward/back, strafe, turn, jump)

Compatible with: Stable-Baselines3, CleanRL, RLlib
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from panda3d.core import (
    Vec3, Point3, LColor, BitMask32,
    CollisionTraverser, CollisionNode, CollisionSphere,
    CollisionRay, CollisionHandlerQueue, CollisionHandlerPusher,
    AmbientLight, DirectionalLight, NodePath,
)
from direct.showbase.ShowBase import ShowBase


class NavigationEnv(gym.Env):
    """
    3D Navigation Environment for Reinforcement Learning.
    
    The agent spawns in a procedurally generated arena with obstacles
    and must reach a series of waypoints as quickly as possible.
    
    Rewards:
        - Distance reduction to current waypoint (shaped reward)
        - +100 for reaching a waypoint
        - -1 per timestep (time pressure)
        - -50 for collision with obstacle
        - -100 for falling off the platform
    
    Episode ends when:
        - All waypoints reached (success)
        - Max timesteps exceeded
        - Agent falls off platform
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        arena_size: float = 50.0,
        num_obstacles: int = 15,
        num_waypoints: int = 5,
        max_steps: int = 2000,
        lidar_rays: int = 16,
        lidar_range: float = 20.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.arena_size = arena_size
        self.num_obstacles = num_obstacles
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.lidar_rays = lidar_rays
        self.lidar_range = lidar_range
        self.render_mode = render_mode
        
        # Observation: [agent_pos(3), agent_vel(3), agent_heading(2),
        #               lidar(16), target_dir(3), target_dist(1)]
        # Total: 3 + 3 + 2 + lidar_rays + 3 + 1 = 12 + lidar_rays
        obs_dim = 12 + lidar_rays
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: [forward/back, strafe, turn, jump]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )
        
        # Internal state
        self._agent_pos = np.zeros(3)
        self._agent_vel = np.zeros(3)
        self._agent_heading = 0.0  # Radians
        self._waypoints = []
        self._current_waypoint_idx = 0
        self._obstacles = []
        self._step_count = 0
        self._prev_dist = 0.0
        self._rng = np.random.default_rng(seed)
        
        # Panda3D setup (only if rendering)
        self._app = None
        self._agent_node = None
        self._setup_panda3d = render_mode == "human"
        
        if self._setup_panda3d:
            self._init_renderer()
    
    def _init_renderer(self):
        """Initialize Panda3D for visual rendering."""
        self._app = ShowBase()
        self._app.disableMouse()
        
        # Camera
        self._app.camera.setPos(0, -80, 60)
        self._app.camera.lookAt(0, 0, 0)
        
        # Lighting
        ambient = AmbientLight("ambient")
        ambient.setColor(LColor(0.3, 0.3, 0.35, 1))
        self._app.render.setLight(self._app.render.attachNewNode(ambient))
        
        sun = DirectionalLight("sun")
        sun.setColor(LColor(0.9, 0.85, 0.8, 1))
        sun_np = self._app.render.attachNewNode(sun)
        sun_np.setHpr(45, -60, 0)
        self._app.render.setLight(sun_np)
        
        # Ground plane
        from panda3d.core import CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-self.arena_size, self.arena_size,
                     -self.arena_size, self.arena_size)
        ground = self._app.render.attachNewNode(cm.generate())
        ground.setP(-90)
        ground.setColor(0.3, 0.5, 0.3, 1)
        
        # Collision system
        self._cTrav = CollisionTraverser()
        self._cHandler = CollisionHandlerQueue()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        self._step_count = 0
        self._current_waypoint_idx = 0
        
        # Spawn agent at center
        self._agent_pos = np.array([0.0, 0.0, 1.0])
        self._agent_vel = np.zeros(3)
        self._agent_heading = 0.0
        
        # Generate obstacles
        self._obstacles = self._generate_obstacles()
        
        # Generate waypoints (not inside obstacles)
        self._waypoints = self._generate_waypoints()
        
        # Calculate initial distance to first waypoint
        self._prev_dist = self._distance_to_current_waypoint()
        
        if self._setup_panda3d:
            self._render_scene()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        self._step_count += 1
        action = np.clip(action, -1.0, 1.0)
        
        # Apply action
        forward = action[0] * 0.5    # Forward/backward speed
        strafe = action[1] * 0.3     # Strafe speed
        turn = action[2] * 0.1       # Turn rate (radians)
        jump = action[3]             # Jump impulse
        
        # Update heading
        self._agent_heading += turn
        
        # Calculate movement in world space
        cos_h = math.cos(self._agent_heading)
        sin_h = math.sin(self._agent_heading)
        
        dx = forward * cos_h - strafe * sin_h
        dy = forward * sin_h + strafe * cos_h
        
        # Apply velocity
        self._agent_vel[0] = dx * 8.0
        self._agent_vel[1] = dy * 8.0
        
        # Jump (only if on ground)
        if jump > 0.5 and self._agent_pos[2] <= 1.1:
            self._agent_vel[2] = 5.0
        
        # Gravity
        self._agent_vel[2] -= 9.81 * (1.0 / 60.0)
        
        # Integrate position
        dt = 1.0 / 60.0
        self._agent_pos += self._agent_vel * dt
        
        # Ground collision
        if self._agent_pos[2] < 1.0:
            self._agent_pos[2] = 1.0
            self._agent_vel[2] = 0.0
        
        # --- Reward Calculation ---
        reward = -1.0  # Time penalty
        terminated = False
        truncated = False
        
        # Distance-based shaping reward
        curr_dist = self._distance_to_current_waypoint()
        reward += (self._prev_dist - curr_dist) * 10.0  # Approach bonus
        self._prev_dist = curr_dist
        
        # Waypoint reached
        if curr_dist < 3.0:
            reward += 100.0
            self._current_waypoint_idx += 1
            
            if self._current_waypoint_idx >= len(self._waypoints):
                terminated = True  # All waypoints reached — success
                reward += 500.0
            else:
                self._prev_dist = self._distance_to_current_waypoint()
        
        # Obstacle collision
        if self._check_obstacle_collision():
            reward -= 50.0
            # Push agent back
            self._agent_pos -= self._agent_vel * dt * 2
            self._agent_vel *= -0.3
        
        # Fall off platform
        if (abs(self._agent_pos[0]) > self.arena_size or
            abs(self._agent_pos[1]) > self.arena_size):
            reward -= 100.0
            terminated = True
        
        # Max steps
        if self._step_count >= self.max_steps:
            truncated = True
        
        # Update renderer
        if self._setup_panda3d:
            self._update_render()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    # ============================================================
    # Observation
    # ============================================================
    
    def _get_observation(self) -> np.ndarray:
        """Build the observation vector."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Agent position (normalized)
        obs[0:3] = self._agent_pos / self.arena_size
        
        # Agent velocity
        obs[3:6] = self._agent_vel / 10.0
        
        # Heading as sin/cos
        obs[6] = math.sin(self._agent_heading)
        obs[7] = math.cos(self._agent_heading)
        
        # Lidar
        lidar = self._cast_lidar()
        obs[8:8+self.lidar_rays] = lidar
        
        # Target direction and distance
        if self._current_waypoint_idx < len(self._waypoints):
            target = self._waypoints[self._current_waypoint_idx]
            direction = target - self._agent_pos[:2]
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            obs[-4] = direction[0]
            obs[-3] = direction[1]
            obs[-2] = 0.0  # Target is on ground plane
            obs[-1] = min(dist / self.arena_size, 1.0)
        
        return obs
    
    def _cast_lidar(self) -> np.ndarray:
        """Cast lidar rays around the agent for obstacle detection."""
        readings = np.ones(self.lidar_rays, dtype=np.float32)
        
        for i in range(self.lidar_rays):
            angle = self._agent_heading + (2 * math.pi * i / self.lidar_rays)
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            
            min_dist = self.lidar_range
            
            for obs_pos, obs_radius in self._obstacles:
                # Ray-circle intersection
                to_obs = obs_pos - self._agent_pos[:2]
                proj = np.dot(to_obs, ray_dir)
                
                if proj < 0:
                    continue
                
                closest = self._agent_pos[:2] + ray_dir * proj
                dist_to_center = np.linalg.norm(obs_pos - closest)
                
                if dist_to_center < obs_radius:
                    hit_dist = proj - math.sqrt(
                        obs_radius**2 - dist_to_center**2
                    )
                    if 0 < hit_dist < min_dist:
                        min_dist = hit_dist
            
            # Arena boundary
            for axis in [0, 1]:
                if ray_dir[axis] != 0:
                    for boundary in [-self.arena_size, self.arena_size]:
                        t = (boundary - self._agent_pos[axis]) / ray_dir[axis]
                        if 0 < t < min_dist:
                            min_dist = t
            
            readings[i] = min_dist / self.lidar_range
        
        return readings
    
    # ============================================================
    # World Generation
    # ============================================================
    
    def _generate_obstacles(self):
        """Generate random cylindrical obstacles."""
        obstacles = []
        for _ in range(self.num_obstacles):
            for attempt in range(50):
                pos = self._rng.uniform(
                    -self.arena_size * 0.8,
                    self.arena_size * 0.8,
                    size=2
                )
                radius = self._rng.uniform(1.5, 4.0)
                
                # Don't spawn on agent
                if np.linalg.norm(pos) < radius + 5.0:
                    continue
                
                # Don't overlap other obstacles
                valid = True
                for other_pos, other_r in obstacles:
                    if np.linalg.norm(pos - other_pos) < radius + other_r + 2.0:
                        valid = False
                        break
                
                if valid:
                    obstacles.append((pos, radius))
                    break
        
        return obstacles
    
    def _generate_waypoints(self):
        """Generate waypoints in free space."""
        waypoints = []
        for _ in range(self.num_waypoints):
            for attempt in range(100):
                pos = self._rng.uniform(
                    -self.arena_size * 0.7,
                    self.arena_size * 0.7,
                    size=2
                )
                
                valid = True
                for obs_pos, obs_r in self._obstacles:
                    if np.linalg.norm(pos - obs_pos) < obs_r + 3.0:
                        valid = False
                        break
                
                if valid:
                    waypoints.append(pos)
                    break
        
        return waypoints
    
    # ============================================================
    # Collision & Distance
    # ============================================================
    
    def _check_obstacle_collision(self) -> bool:
        agent_radius = 0.5
        for obs_pos, obs_r in self._obstacles:
            dist = np.linalg.norm(self._agent_pos[:2] - obs_pos)
            if dist < obs_r + agent_radius:
                return True
        return False
    
    def _distance_to_current_waypoint(self) -> float:
        if self._current_waypoint_idx >= len(self._waypoints):
            return 0.0
        target = self._waypoints[self._current_waypoint_idx]
        return float(np.linalg.norm(self._agent_pos[:2] - target))
    
    # ============================================================
    # Info
    # ============================================================
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._step_count,
            "waypoints_reached": self._current_waypoint_idx,
            "total_waypoints": len(self._waypoints),
            "distance_to_target": self._distance_to_current_waypoint(),
            "agent_position": self._agent_pos.tolist(),
        }
    
    # ============================================================
    # Rendering (Panda3D)
    # ============================================================
    
    def _render_scene(self):
        """Full scene rebuild after reset."""
        if not self._app:
            return
        
        # Clear previous scene objects
        for child in self._app.render.getChildren():
            if child.getName() in ("obstacle", "waypoint", "agent"):
                child.removeNode()
        
        # Obstacles
        for pos, radius in self._obstacles:
            from panda3d.core import CardMaker
            obstacle = self._app.loader.loadModel("models/misc/sphere")
            obstacle.setScale(radius)
            obstacle.setPos(pos[0], pos[1], radius)
            obstacle.setColor(0.7, 0.3, 0.3, 1)
            obstacle.setName("obstacle")
            obstacle.reparentTo(self._app.render)
        
        # Waypoints
        for i, wp in enumerate(self._waypoints):
            marker = self._app.loader.loadModel("models/misc/sphere")
            marker.setScale(1.5)
            marker.setPos(wp[0], wp[1], 1.5)
            color = LColor(0.2, 0.9, 0.3, 0.7) if i == 0 else LColor(0.3, 0.3, 0.9, 0.5)
            marker.setColor(color)
            marker.setName("waypoint")
            marker.reparentTo(self._app.render)
        
        # Agent
        self._agent_node = self._app.loader.loadModel("models/misc/sphere")
        self._agent_node.setScale(0.8)
        self._agent_node.setColor(0.2, 0.7, 1.0, 1)
        self._agent_node.setName("agent")
        self._agent_node.reparentTo(self._app.render)
    
    def _update_render(self):
        """Update agent position in renderer."""
        if self._agent_node:
            self._agent_node.setPos(
                self._agent_pos[0],
                self._agent_pos[1],
                self._agent_pos[2]
            )
            self._agent_node.setH(math.degrees(self._agent_heading))
        
        if self._app:
            self._app.taskMgr.step()
    
    def render(self):
        if self.render_mode == "human" and self._app:
            self._app.taskMgr.step()
    
    def close(self):
        if self._app:
            self._app.destroy()
            self._app = None


# ============================================================
# Gymnasium Registration
# ============================================================

gym.register(
    id="Panda3DNav-v0",
    entry_point="env.nav_env:NavigationEnv",
    max_episode_steps=2000,
)


    def get_episode_stats(self) -> dict:
        """Return stats about the current episode progress."""
        return {
            "steps": self._step_count,
            "waypoints_reached": self._current_waypoint_idx,
            "total_waypoints": len(self._waypoints),
            "completion_pct": self._current_waypoint_idx / max(len(self._waypoints), 1),
        }


    def set_difficulty(self, num_obstacles: int, num_waypoints: int) -> None:
        """Adjust difficulty parameters (takes effect on next reset)."""
        self.num_obstacles = num_obstacles
        self.num_waypoints = num_waypoints


    def _is_on_ground(self) -> bool:
        """Returns True if agent is at or near ground level."""
        return self._agent_pos[2] <= 1.1

    def _get_heading_vector(self) -> tuple:
        """Returns (sin, cos) of current heading angle."""
        import math
        return math.sin(self._agent_heading), math.cos(self._agent_heading)


    def seed(self, seed: int) -> None:
        """Re-seed the environment RNG (Gymnasium legacy API)."""
        import numpy as np
        self._rng = np.random.default_rng(seed)


    def get_waypoint_positions(self) -> list:
        """Return remaining waypoint positions as a list of [x, y] arrays."""
        return [wp.tolist() for wp in self._waypoints[self._current_waypoint_idx:]]


    def compute_reward_components(self, action) -> dict:
        """Return a breakdown of reward components for analysis."""
        curr_dist = self._distance_to_current_waypoint()
        approach = (self._prev_dist - curr_dist) * 10.0
        time_pen = -1.0
        collision_pen = -50.0 if self._check_obstacle_collision() else 0.0
        return {"approach": approach, "time": time_pen, "collision": collision_pen}


    def get_obstacle_positions(self) -> list:
        """Return list of (center_x, center_y, radius) for all obstacles."""
        return [(float(pos[0]), float(pos[1]), float(r)) for pos, r in self._obstacles]


    def apply_action_noise(self, action, noise_std: float = 0.05):
        """Add Gaussian noise to actions for domain randomization."""
        import numpy as np
        noisy = action + self._rng.normal(0, noise_std, size=action.shape)
        return np.clip(noisy, -1.0, 1.0).astype(np.float32)


    def export_trajectory(self) -> list:
        """Export agent trajectory as list of position snapshots."""
        return [pos.tolist() for pos in self._trajectory_log] if hasattr(self, "_trajectory_log") else []


    @property
    def current_waypoint(self):
        """Return the current target waypoint position, or None if all reached."""
        if self._current_waypoint_idx < len(self._waypoints):
            return self._waypoints[self._current_waypoint_idx]
        return None

    def randomize_arena(self) -> None:
        """Randomize obstacle layout without full reset (keeps agent position)."""
        self._obstacles = self._generate_obstacles()
        if self._setup_panda3d:
            self._render_scene()

    def get_normalized_obs(self) -> 'np.ndarray':
        """Return observation clipped and scaled to [-1, 1]."""
        obs = self._get_observation()
        return obs.clip(-1.0, 1.0)

    def get_agent_position(self) -> list:
        """Return agent world position as [x, y, z]."""
        return self._agent_pos.tolist()

    def get_agent_heading_deg(self) -> float:
        """Return agent heading in degrees."""
        import math
        return math.degrees(self._agent_heading) % 360
