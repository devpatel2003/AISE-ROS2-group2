import gym 
from gym import Env
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
import time
from astar import astar


# ----- TUNABLE PARAMETERS FOR ENVIORMENT (not hyperparams) -----
ENV_BOUNDARY = 2.0          # Robot operates within -2..2 in x,y
MIN_DISTANCE = ENV_BOUNDARY * 0.8 # Spawn the robot and goal this distance apart
EP_LENGTH = 15_000           # Max steps per episode
TIME_STEP = 1.0 / 240
WHEEL_BASE = 0.14           # Distance between left & right wheels
MAX_LINEAR_SPEED = 0.3       # m/s
MAX_ANGULAR_SPEED = 1     # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

NUM_MOVING_OBSTACLES = 10
OBSTACLE_SPEED = 0.1

WP_DISTANCE = 0.8
RADIUS_COLLISION = 0.3
CLUTTER_PROB = 0.2

# LiDAR specs
NUM_READINGS = 72                # 360° / 0.8° = 450 <- LIDAR ON REAL WORLD, 360° / 5° = 72 <- SIM  
MAX_LIDAR_RANGE = 1            # Up to ~12 m for black objects
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.2         # Slightly above ground/robot’s base

# Reward & penalty weights        #-0.2, 500, 10, 3, 10, 1
REWARD_GOAL_BONUS = 5
REWARD_WP = 200
REWARD_DTG_POSITIVE = 1.2    # Reward for reducing distance to goal, after * dist_improve number becomes very small
REWARD_HTG_POSITIVE = 1    # Reward for facing goal
REWARD_ACTION_HIGH = 1  # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 0.5      # Reward for forward & rotating
ASTAR_GAIN = 1
PENALTY_COLLISION = -10
PENALTY_NEAR_COLLISION = -10
PENALTY_TURN = 0
PENALTY_TIME = -2



class CrowdAvoidanceEnv(Env):
    """Gym environment for a robot navigating a PyBullet scene."""

    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False, preset_grid=None, preset_start=None, preset_goal=None):
        super().__init__()
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSubSteps=5)

        self.past_observations = np.zeros((5, NUM_READINGS))  # Store last 5 LiDAR frames (framestacking)

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([0, -math.pi, 0], dtype=np.float32),
                np.full(NUM_READINGS * 5, 0, dtype=np.float32)
            )),
            high=np.concatenate((
                np.array([5, math.pi, 1], dtype=np.float32),
                np.full(NUM_READINGS * 5, 1, dtype=np.float32)
            )),
            dtype=np.float32
        )

        self.robot = None
        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
        self.collision_happened = False

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False
        self.prev_wp_dist = None

        self.grid_size = 10  # Create 20x20 grid
        self.cell_size = (ENV_BOUNDARY * 2) / self.grid_size
        self.grid_map = None
        self.astar_path = None
        self.plane_id = None

        # view expert trajectories
        self.preset_grid = preset_grid
        self.preset_start = preset_start
        self.preset_goal = preset_goal

        self.reset()

    def world_to_grid(self, x, y):
        gx = int((x + ENV_BOUNDARY) / self.cell_size)
        gy = int((y + ENV_BOUNDARY) / self.cell_size)
        return (gy, gx)

    def grid_to_world(self, gx, gy):
        x = gx * self.cell_size - ENV_BOUNDARY + self.cell_size / 2
        y = gy * self.cell_size - ENV_BOUNDARY + self.cell_size / 2
        return x, y
    


    def reset(self):


        if hasattr(self, "debug_lines"):
            for line_id in self.debug_lines:
                p.removeUserDebugItem(line_id)

        self.previous_linear_speed = 0.0
        self.previous_angular_speed = 0.0
        self.previous_goal_distance = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False
        self.prev_wp_dist = None
        self.collision_happened = False

        wall_half_extents = [self.cell_size * 0.5] * 3

        # Use preset grid, start, goal if provided
        if self.preset_grid is not None and self.preset_start is not None and self.preset_goal is not None:
            self.grid_map = self.preset_grid.copy()
            self.astar_path = astar(self.grid_map, self.world_to_grid(*self.preset_start), self.world_to_grid(*self.preset_goal))
            start_x, start_y = self.preset_start
            goal_x, goal_y = self.preset_goal
        else:
            # Try generating a valid grid + A* path before touching the simulation
            while True:
                grid_map = np.zeros((self.grid_size, self.grid_size), dtype=int)

                # Add random internal obstacles
                for y in range(self.grid_size):
                    for x in range(self.grid_size):
                        if random.random() < CLUTTER_PROB:
                            grid_map[y, x] = 1

                # Try placing robot and goal
                for _ in range(100):  # Cap attempts
                    start_x = random.uniform(-ENV_BOUNDARY + 0.5, -0.5)
                    start_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
                    goal_x = random.uniform(0.5, ENV_BOUNDARY - 0.5)
                    goal_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

                    distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                    if distance < MIN_DISTANCE:
                        continue

                    start_idx = self.world_to_grid(start_x, start_y)
                    goal_idx = self.world_to_grid(goal_x, goal_y)
                    
                    # Ensure robot and goal are not in obstacles
                    if grid_map[start_idx[0], start_idx[1]] == 1:
                        continue
                    if grid_map[goal_idx[0], goal_idx[1]] == 1:
                        continue

                    path = astar(grid_map, start_idx, goal_idx)
                    if path:
                        # Optional: forcibly clear spawn cells in grid map
                        grid_map[start_idx[0], start_idx[1]] = 0
                        grid_map[goal_idx[0], goal_idx[1]] = 0

                        self.grid_map = grid_map
                        self.astar_path = path
                        break

                if self.astar_path:
                    break  # Exit the world generation loop

        # After valid map found → setup physics world
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        self.plane_id = p.loadURDF("plane.urdf")
        self.obstacle_ids = []

        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=wall_half_extents, rgbaColor=[0.4, 0.4, 0.4, 1])

        # Populate the world with internal obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid_map[y, x] == 1:
                    wx, wy = self.grid_to_world(x, y)
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=wall_collision,
                        baseVisualShapeIndex=wall_visual,
                        basePosition=[wx, wy, wall_half_extents[2]]
                    )
                    self.obstacle_ids.append(body_id)

        # Add border walls around the grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1:
                    self.grid_map[y, x] = 1  # Mark as wall
                    wx, wy = self.grid_to_world(x, y)
                    body_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=wall_collision,
                        baseVisualShapeIndex=wall_visual,
                        basePosition=[wx, wy, wall_half_extents[2]]
                    )
                    self.obstacle_ids.append(body_id)

        self.goal_position = [goal_x, goal_y, 0.05]
        self.goal_marker = p.loadURDF("sphere_small.urdf", basePosition=self.goal_position, globalScaling=2)
        p.changeVisualShape(self.goal_marker, -1, rgbaColor=[0, 1, 0, 1])

        # Draw A* path
        '''self.debug_lines = []
        for i in range(len(self.astar_path) - 1):
            (gy1, gx1) = self.astar_path[i]
            (gy2, gx2) = self.astar_path[i + 1]
            x1, y1 = self.grid_to_world(gx1, gy1)
            x2, y2 = self.grid_to_world(gx2, gy2)
            line_id = p.addUserDebugLine(
                lineFromXYZ=[x1, y1, 0.05],
                lineToXYZ=[x2, y2, 0.05],
                lineColorRGB=[0, 0, 1],
                lineWidth=2.0,
                lifeTime=0
            )
            self.debug_lines.append(line_id)'''

        self.robot = p.loadURDF(
            "./urdf/MicroROS.urdf",
            basePosition=[start_x, start_y, 0.05], 
            useFixedBase=False
        )

        self.robot_start_pos = (start_x, start_y)


        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.changeDynamics(self.robot, joint, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, ccdSweptSphereRadius=0.05)
        p.setPhysicsEngineParameter(enableConeFriction=True)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        p.setCollisionFilterPair(self.robot, self.plane_id, -1, -1, enableCollision=False)

        self.previous_goal_distance = 0
        
        return self._get_observation()



    def step(self, action):
        self.current_step += 1

        # Scale speed
        lin_speed = action[0] * MAX_LINEAR_SPEED
        ang_speed = action[1] * MAX_ANGULAR_SPEED


        # Find current position and calculate next positon
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        _, _, yaw = p.getEulerFromQuaternion(orn)

        new_yaw = yaw + ang_speed * TIME_STEP
        new_x = pos[0] + lin_speed * math.cos(new_yaw) * TIME_STEP
        new_y = pos[1] + lin_speed * math.sin(new_yaw) * TIME_STEP

        p.resetBasePositionAndOrientation(
            self.robot,
            [new_x, new_y, pos[2]],
            p.getQuaternionFromEuler([0, 0, new_yaw])
        )

        p.stepSimulation()




        obs = self._get_observation()
        reward, done = self._compute_reward(action)

        self.episode_reward += reward
        self.episode_length += 1

        if self.current_step >= self.max_steps:
            done = True
        
        info = {}

        if done:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length
            }

        
        return obs, reward, done, info

    def get_goal_angle(self):
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        rx, ry, _ = robot_pos
        gx, gy, _ = self.goal_position
        abs_angle = math.atan2(gy - ry, gx - rx)
        _, _, yaw = p.getEulerFromQuaternion(robot_ori)

        angle = abs_angle - yaw
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return angle
    
    def compute_waypoint(self, robot_pos, goal_pos, R=WP_DISTANCE):
        dx = goal_pos[0] - robot_pos[0]
        dy = goal_pos[1] - robot_pos[1]
        dist_rg = np.sqrt(dx*dx + dy*dy)

        if dist_rg <= R:
            # Already near goal, so the “waypoint” is just the goal
            return goal_pos
        else:
            # Lambda = R / dist_rg
            lam = R / dist_rg
            wx = robot_pos[0] + lam * dx
            wy = robot_pos[1] + lam * dy
            return (wx, wy)
    
    def _compute_reward(self, action):
        lin_vel, ang_vel = action
        base_reward, penalty, astar_reward  = 0, 0, 0

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(robot_pos[:2])
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)

        base_reward = 0.5 * (lin_vel - abs(ang_vel))

        # 1) Distance to goal
        prev_dist = getattr(self, "previous_goal_distance", goal_dist)
        dist_improvement = prev_dist - goal_dist
        self.previous_goal_distance = goal_dist
        #dtg_r = REWARD_DTG_POSITIVE * max(dist_improvement*1000, 0) #*1000 because dist improvement is 0.00002
        if dist_improvement > 0.0001:
            base_reward += 0.5
       

        # 2) Goal reached
        if goal_dist < GOAL_REACHED_DIST:
            bonus = REWARD_GOAL_BONUS + (self.max_steps - self.current_step) * 0.02
            return bonus, True
        
        # 2) Collision Penalty
        lidar_scan = self._perform_lidar_scan()
        min_distance = np.min(lidar_scan)
        
   
        if min_distance < 0.125:
            print("🚨 Robot hit a wall or obstacle!")
            self.collision_happened = True
            return PENALTY_COLLISION, True
            collision_p = PENALTY_COLLISION # 5 BIG BOOMS for collision
        elif min_distance < 0.20:
            penalty = -1
        elif min_distance < 0.25:
            penalty = -0.25
        
        contacts = p.getContactPoints(self.robot)
        for contact in contacts:
            if contact[2] not in [self.robot, self.plane_id, self.goal_marker] and contact[2] in self.obstacle_ids:
                self.collision_happened = True
                print("🚨 Robot hit a wall or obstacle!")
                return PENALTY_COLLISION, True
            
        self.astar_world_path = [self.grid_to_world(x, y) for (y, x) in self.astar_path]
        min_astar_dist = min(
            np.linalg.norm(robot_xy - np.array([px, py]))
            for (px, py) in self.astar_world_path
        )
        if lin_vel > 0.1:
            if min_astar_dist < 0.:
               astar_reward = ASTAR_GAIN
            else:
               penalty += -0.5
        if ang_vel > 0.1:
            penalty += -0.25
        
        # 7) Final reward
        reward =  base_reward + penalty + astar_reward
        #print(dtg_r , htg_r , collision_p , time_p, wp_r)

      
        return reward, False
    
    def _perform_lidar_scan(self):
        """Perform a simulated 360° LiDAR sweep with accurate angles and no drift."""

        # Get robot position and precise orientation
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos

        # Compute the exact robot yaw using the full rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(robot_ori)).reshape(3, 3)
        forward_vector = np.array([1, 0, 0])  # Assuming robot's x-axis is forward
        transformed_vector = rot_matrix @ forward_vector
        robot_yaw = np.arctan2(transformed_vector[1], transformed_vector[0])  # More stable yaw calculation

        # LiDAR start position (slightly above robot's base to prevent ground interference)
        start_pos = [robot_x, robot_y, robot_z + LIDAR_HEIGHT_OFFSET]

        # Generate angles correctly (no incremental accumulation)
        angles = np.linspace(-np.pi, np.pi, NUM_READINGS, endpoint=False)
        rotated_angles = angles + robot_yaw  # Adjust based on the robot’s actual orientation

        # Compute LiDAR ray end positions
        end_positions = np.array([
            [robot_x + np.cos(angle) * MAX_LIDAR_RANGE,
            robot_y + np.sin(angle) * MAX_LIDAR_RANGE,
            robot_z + LIDAR_HEIGHT_OFFSET]
            for angle in rotated_angles
        ])

        # Perform batched raycasting
        results = p.rayTestBatch([start_pos] * NUM_READINGS, end_positions.tolist())

        # Compute accurate distances
        ranges = np.array([
            max(res[2] * MAX_LIDAR_RANGE if res[0] != -1 else MAX_LIDAR_RANGE, MIN_LIDAR_RANGE)
            for res in results
        ], dtype=np.float32)

        return ranges

    def _get_observation(self):

        # LiDAR data 
        lidar_scan = self._perform_lidar_scan()
        self.past_observations = np.roll(self.past_observations, shift=-1, axis=0) # Shift new observation into list
        self.past_observations[-1] = lidar_scan
        flattened_lidar = self.past_observations.flatten()
        norm_lidar = (flattened_lidar - MIN_LIDAR_RANGE) / (MAX_LIDAR_RANGE - MIN_LIDAR_RANGE) # Min Max normalization
        

        # Compute goal distance, angle, IMU 
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos
        goal_dx = self.goal_position[0] - robot_x
        goal_dy = self.goal_position[1] - robot_y
        goal_distance = np.sqrt(goal_dx**2 + goal_dy**2)
        goal_angle = self.get_goal_angle()
        
        obs = np.concatenate((
                np.array([goal_distance, goal_angle, self.current_step / self.max_steps], dtype=np.float32),
                norm_lidar
            ), axis=0)
        
        #obs = (obs - obs.mean()) / (obs.std() + 1e-8)


        return obs
    
    

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

