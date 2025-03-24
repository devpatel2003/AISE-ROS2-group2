import os
import pickle
import numpy as np
import math
import pybullet as p
from imitation.data.types import TrajectoryWithRew
from IRL_maze_gym import CrowdAvoidanceEnv

# ---- CONFIG ----
NUM_TRAJECTORIES = 5
STEPS_PER_WAYPOINT = 1000
SAVE_DIR = "expert_trajectories1"
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_action_to_point(current_pos, current_yaw, target_point, speed_scale=1.0):
    dx = target_point[0] - current_pos[0]
    dy = target_point[1] - current_pos[1]
    target_angle = math.atan2(dy, dx)
    angle_diff = (target_angle - current_yaw + math.pi) % (2 * math.pi) - math.pi

    # Slow on turn
    angle_scale = max(0.0, 1.0 - abs(angle_diff) / math.pi)
    v = speed_scale * angle_scale ** 3  # forward motion (scaled)
    w = angle_diff * 2.0  # proportional control for rotation

    v = np.clip(v, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    return np.array([v, w], dtype=np.float32)

# ---- Main Loop ----
for traj_id in range(NUM_TRAJECTORIES):
    print(f"Generating expert trajectory {traj_id+1}/{NUM_TRAJECTORIES}")

    # Randomize speed for this trajectory
    speed_scale = np.random.uniform(0.5, 1.5)
    print(f"Using speed scale: {speed_scale:.2f}")

    env = CrowdAvoidanceEnv(use_gui=False)
    obs, _ = env.reset()

    astar_world_path = [env.grid_to_world(x, y) for (y, x) in env.astar_path]

    obs_list = []
    act_list = []
    info_list = []
    step = 0
    done = False

    # Save initial observation before the first action
    obs_list.append(obs.copy())

    for wp in astar_world_path:
        for _ in range(STEPS_PER_WAYPOINT):
            pos, ori = p.getBasePositionAndOrientation(env.robot)
            yaw = p.getEulerFromQuaternion(ori)[2]

            action = compute_action_to_point(pos, yaw, wp, speed_scale=speed_scale)
            obs, reward, done, _, _ = env.step(action)

            # Append action
            act_list.append(action.copy())
            # Append new obs after action
            obs_list.append(obs.copy())

            # Save info (Optional, used for diagnostics but not required for AIRL)
            robot_xy = np.array(pos[:2])
            goal_xy = np.array(env.goal_position[:2])
            goal_dist = np.linalg.norm(goal_xy - robot_xy)

            info = {
                "step": step,
                "goal_distance": goal_dist,
                "is_success": goal_dist < 0.25,
                "min_lidar": float(np.min(obs[2:])),  # Assuming lidar data starts at index 2
                "expert_id": traj_id,
                "speed_scale": speed_scale
            }
            info_list.append(info)

            step += 1

            if done or goal_dist < 0.25:
                break

        if done:
            break

    # === Save in AIRL-compatible format ===
    expert_traj = TrajectoryWithRew(
        obs=np.array(obs_list),
        acts=np.array(act_list),
        rews=np.zeros(len(act_list), dtype=np.float32),  # Rewards can be zero for expert trajectories
        infos=info_list,  # Reintroduced the 'infos' argument
        terminal=True
    )

    with open(os.path.join(SAVE_DIR, f"expert_{traj_id:03}.pkl"), "wb") as f:
        pickle.dump(expert_traj, f)

    env.close()
