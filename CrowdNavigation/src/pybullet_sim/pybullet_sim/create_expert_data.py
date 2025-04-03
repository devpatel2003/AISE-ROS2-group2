import os
import pickle
import numpy as np
import math
import pybullet as p
from imitation.data.types import TrajectoryWithRew
from IRL_maze_gym import CrowdAvoidanceEnv

# ---- CONFIG ----
NUM_TRAJECTORIES = 200
STEPS_PER_WAYPOINT = 1500
WAYPOINT_RADIUS = 0.15
SAVE_DIR = "expert_trajectories3"
collision_happened = False

os.makedirs(SAVE_DIR, exist_ok=True)

class HeadingController:
    def __init__(self, kp=0.8, kd=0.15, dt=1/240):
        self.kp = kp
        self.kd = kd
        self.dt = dt
        self.prev_error = 0.0

    def compute_turn_speed(self, angle_diff):
        # Hard threshold for close angles
        if abs(angle_diff) < 0.05:
            angle_diff = 0.0
        d_error = (angle_diff - self.prev_error) / self.dt
        turn = self.kp * angle_diff  + self.kd * d_error
        self.prev_error = angle_diff
        return turn


def compute_action_to_point(current_pos, current_yaw, target_point, heading_ctrl: HeadingController, speed_scale=1.0):
    dx = target_point[0] - current_pos[0]
    dy = target_point[1] - current_pos[1]
    target_angle = math.atan2(dy, dx)
    angle_diff = (target_angle - current_yaw + math.pi) % (2*math.pi) - math.pi

    turn_speed = heading_ctrl.compute_turn_speed(angle_diff)
    # scale forward velocity if angle is large
    angle_scale = max(0.0, 1.0 - abs(angle_diff)/math.pi)
    forward_speed =  speed_scale * (angle_scale**3)

    # clip
    w = np.clip(turn_speed, -1.0, 1.0)
    v = np.clip(forward_speed, -1.0, 1.0)

    return np.array([v, w], dtype=np.float32)

# ---- Main Loop ----
for traj_id in range(NUM_TRAJECTORIES):
    print(f"Generating expert trajectory {traj_id+1}/{NUM_TRAJECTORIES}")

    speed_scale = 1 #np.random.uniform(0.5, 1.5)
    print(f"Using speed scale: {speed_scale:.2f}")

    env = CrowdAvoidanceEnv(use_gui=False)
    obs = env.reset()

    astar_world_path = [env.grid_to_world(x, y) for (y, x) in env.astar_path]

    obs_list = []
    act_list = []
    info_list = []
    step = 0
    done = False

    obs_list.append(obs.copy())

    final_goal_distance = float("inf") 

    pid = HeadingController(kp=0.7, kd=0.2, dt=1/240) 

    for wp in astar_world_path:
        for _ in range(STEPS_PER_WAYPOINT):
            pos, ori = p.getBasePositionAndOrientation(env.robot)
            yaw = p.getEulerFromQuaternion(ori)[2]

            action = compute_action_to_point(pos, yaw, wp, speed_scale=speed_scale, heading_ctrl=pid)
            obs, reward, done, _ = env.step(action)

            act_list.append(action.copy())
            obs_list.append(obs.copy())

            waypoint_xy = np.array(wp)         # The current waypoint's XY
            robot_xy   = np.array(pos[:2])     # Robot’s XY
            wp_dist    = np.linalg.norm(waypoint_xy - robot_xy)
            goal_xy = np.array(env.goal_position[:2])
            goal_dist = np.linalg.norm(goal_xy - robot_xy)

            info = {
                "step": step,
                "goal_distance": goal_dist,
                "is_success": goal_dist < 0.21,
                "min_lidar": float(np.min(obs[2:])),
                "expert_id": traj_id,
                "speed_scale": speed_scale
            }
            info_list.append(info)

            step += 1
            final_goal_distance = goal_dist  
            collision_happened = env.collision_happened

            if wp_dist < WAYPOINT_RADIUS or goal_dist < 0.2:
                # We are close enough to the current waypoint
                break


        if done:
            break

    # === Only save successful runs ===
    if final_goal_distance < 0.21 and (not collision_happened):

        final_info = info_list[-1]
        final_info["grid_map"] = env.grid_map.copy()
        final_info["start"] = env.robot_start_pos
        final_info["goal"] = env.goal_position[:2]

        expert_traj = TrajectoryWithRew(
            obs=np.array(obs_list),
            acts=np.array(act_list),
            rews=np.zeros(len(act_list), dtype=np.float32),
            infos=info_list,
            terminal=True
        )

        with open(os.path.join(SAVE_DIR, f"expert_{traj_id:03}.pkl"), "wb") as f:
            pickle.dump(expert_traj, f)

        print("Trajectory saved: success achieved!\n")
    else:
        print("Trajectory discarded: did not reach goal.\n")

    env.close()
