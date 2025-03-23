import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
import time

# ----- TUNABLE PARAMETERS FOR ENVIORMENT (not hyperparams) -----
ENV_BOUNDARY = 2.0          # Robot operates within -2..2 in x,y
MIN_DISTANCE = ENV_BOUNDARY * 0.8 # Spawn the robot and goal this distance apart
EP_LENGTH = 10_000           # Max steps per episode
TIME_STEP = 1.0 / 240
WHEEL_BASE = 0.14           # Distance between left & right wheels
MAX_LINEAR_SPEED = 0.2       # m/s
MAX_ANGULAR_SPEED = 1     # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

NUM_MOVING_OBSTACLES = 10
OBSTACLE_SPEED = 0.1

WP_DISTANCE = 0.8
RADIUS_COLLISION = 0.3

# LiDAR specs
NUM_READINGS = 72                # 360° / 0.8° = 450 <- LIDAR ON REAL WORLD, 360° / 5° = 72 <- SIM  
MAX_LIDAR_RANGE = 1            # Up to ~12 m for black objects
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.5         # Slightly above ground/robot’s base

# Reward & penalty weights        #-0.2, 500, 10, 3, 10, 1
REWARD_GOAL_BONUS = 100_000
REWARD_WP = 200
REWARD_DTG_POSITIVE = 1.2    # Reward for reducing distance to goal, after * dist_improve number becomes very small
REWARD_HTG_POSITIVE = 1    # Reward for facing goal
REWARD_ACTION_HIGH = 1  # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 0.5      # Reward for forward & rotating
PENALTY_COLLISION = -200
PENALTY_NEAR_COLLISION = -10
PENALTY_TURN = 0
PENALTY_TIME = -2
 


class CrowdAvoidanceEnv(gym.Env):
    """Gym environment for a robot navigating a PyBullet scene."""

    metadata = {'render.modes': ['human']}

    def __init__(self, use_gui=False):
        super().__init__()
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSubSteps=5)

        self.past_observations = np.zeros((5, NUM_READINGS))  # Store last 5 LiDAR frames (framestacking)


        # [v, w]: linear & angular velocity commands, matches ros2 outputs, low is lowest value each command can be, high is the highest value the command can be
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]),
            dtype=np.float32
        )

        # Define observation space: [goal_dist, goal_angle, ax, ay, wx, wy, wz] + LiDAR readings (including past 5)
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([0, -math.pi, -5, -5, -5, -5, -5], dtype=np.float32),
                np.full(NUM_READINGS * 5, 0, dtype=np.float32)  # Create (num_reading * past observation) sized array filled with zeros for the lowest number a lidar scan can be
            )),
            high=np.concatenate((
                np.array([5, math.pi, 5, 5, 5, 5, 5], dtype=np.float32),
                np.full(NUM_READINGS * 5, MAX_LIDAR_RANGE, dtype=np.float32)  # Same as low, but the max lidar value (12)
            )),
            dtype=np.float32
        )

        self.robot = None
        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.obstacle = None
        self.current_step = 0

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False  # track if we got +100 already
        self.prev_wp_dist = None

        

        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.previous_linear_speed = 0.0
        self.previous_angular_speed = 0.0
        self.previous_goal_distance = None
        self.current_step = 0

        self.current_wp = None
        self.num_steps_since_wp = 0
        self.wp_reached = False  # track if we got +100 already
        self.prev_wp_dist = None

        plane_id = p.loadURDF("plane.urdf")

        # Find a random staring configuration that satisfies min distance condition
        while True:
                # Robot on left half
                start_x = random.uniform(-ENV_BOUNDARY + 0.5, -0.5)
                start_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

                # Ball (goal) on right half
                goal_x = random.uniform(0.5, ENV_BOUNDARY - 0.5)
                goal_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

                distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
                if distance >= MIN_DISTANCE:
                    break
        
        # Spawn goal
        self.goal_position = [goal_x, goal_y, 0.05]
        self.goal_marker = p.loadURDF("sphere_small.urdf", basePosition=self.goal_position, globalScaling=1)
        p.changeVisualShape(self.goal_marker, -1, rgbaColor=[0, 1, 0, 1])

        self.robot = p.loadURDF(
            "C:/Users/Dev/Documents/Personal/Projects/CrowdNavigation/src/pybullet_sim/urdf/MicroROS.urdf",
            basePosition=[start_x, start_y, 0.05], 
            useFixedBase=False
        )

        # Prevents robot from sliding like its on ice
        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.changeDynamics(self.robot, joint, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, ccdSweptSphereRadius=0.05)
        p.setPhysicsEngineParameter(enableConeFriction=True)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        p.setCollisionFilterPair(self.robot, plane_id, -1, -1, enableCollision=False)


      
        # Spawn a single obstacle at a random position between the robot and goal
        obstacle_x = (start_x + goal_x) / 2 + random.uniform(-0.2, 0.2)
        obstacle_y = (start_y + goal_y) / 2 + random.uniform(-0.2, 0.2)
        self.obstacle = p.loadURDF(
            "C:/Users/Dev/Documents/Personal/Projects/AISE-ROS2-group2/CrowdNavigation/src/pybullet_sim/urdf/obstacle.urdf",
            basePosition=[obstacle_x, obstacle_y, 0.3], 
            globalScaling=1,
            useFixedBase = True)

        # Disable collisions between the goal_marker and obstacle
        p.setCollisionFilterPair(self.obstacle, self.goal_marker, -1, -1, enableCollision=False)
        p.setCollisionFilterPair(self.obstacle, self.robot, -1, -1, enableCollision=True)


        
        self.previous_goal_distance = 0
        return self._get_observation(), {}

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

        if self.current_step >= self.max_steps:
            done = True

        return obs, reward, done, False, {}

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
        dtg_r, htg_r, wp_r, time_p, collision_p  = 0, 0, 0, 0, 0

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = (robot_pos[0], robot_pos[1])
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)

        # 1) Distance to goal
        # Checks if distance to the goal has decreased 
        prev_dist = getattr(self, "previous_goal_distance", goal_dist)
        dist_improvement = prev_dist - goal_dist
        self.previous_goal_distance = goal_dist
        #dtg_r = REWARD_DTG_POSITIVE * max(dist_improvement*1000, 0) #*1000 because dist improvement is 0.00002
        if dist_improvement > 0.0001:
            dtg_r = REWARD_DTG_POSITIVE
       
        # 2) Heading to goal reward
        goal_angle = self.get_goal_angle()  # Between -pi and pi
        previous_goal_angle = getattr(self, "previous_goal_angle", goal_angle)
        angle_improvement = (abs(previous_goal_angle) - abs(goal_angle))  # Increase reward for reducing error
        self.previous_goal_angle = goal_angle 
        htg_r = REWARD_HTG_POSITIVE* np.cos(goal_angle) # facing goal (0) = +1, facing away (180) = -1 
        if angle_improvement > 0.0001:
            htg_r += 0
 
    
        # 3) Collision Penalty
        lidar_scan = self._perform_lidar_scan()
        min_distance = np.min(lidar_scan)  # Closest object detected by LiDAR
        
        if min_distance < 0.1:
            return PENALTY_COLLISION, True
            collision_p = PENALTY_COLLISION # 5 BIG BOOMS for collision
        elif min_distance < RADIUS_COLLISION:
            collision_p = PENALTY_NEAR_COLLISION * (RADIUS_COLLISION - min_distance) 

        contacts = p.getContactPoints(self.robot)
        if contacts:
            contact_id = contacts[0][2]
            if contact_id != 0:
                self.last_collision = contact_id
                #print(f"Collision with object {contact_id} at {contacts[0][6:9]}")
                return PENALTY_COLLISION, True
            
        
        # 4) Waypoint logic
        # (A) If no current_wp, compute one now
        if self.current_wp is None:
            self.current_wp = self.compute_waypoint(robot_pos, (gx, gy), R=WP_DISTANCE)
            self.num_steps_since_wp = 0
            self.wp_reached = False
            self.prev_wp_dist = math.dist(robot_xy, self.current_wp)

        # (B) Check if we've hit 1000 steps since last WP
        self.num_steps_since_wp += 1
        if self.num_steps_since_wp > 1000:
            # Force a new WP
            self.current_wp = self.compute_waypoint(robot_pos, (gx, gy), R=WP_DISTANCE)
            self.num_steps_since_wp = 0
            self.wp_reached = False
            self.prev_wp_dist = math.dist(robot_xy, self.current_wp)

        # (C) Check distance to current_wp
        wp_dist = math.dist(robot_xy, self.current_wp)
        
        # If < 0.1 and not previously reached => big reward
        if wp_dist < 0.1 and not self.wp_reached:
            wp_r = REWARD_WP
            self.wp_reached = True  # mark so we don't keep awarding it
            # Compute a new WP immediately
            self.current_wp = self.compute_waypoint(robot_pos, (gx, gy), R=WP_DISTANCE)
            self.num_steps_since_wp = 0
            self.wp_reached = False
            self.prev_wp_dist = math.dist(robot_xy, self.current_wp)
        
        # 5) Time penalty
        #time_penalty = -10 * (self.current_step / EP_LENGTH)  #  Larger time penalty over time
        time_p = PENALTY_TIME

        # 6) Goal reached
        if goal_dist < GOAL_REACHED_DIST:
            return REWARD_GOAL_BONUS, True
        
        # 7) Final reward
        reward =  dtg_r + htg_r  + time_p + collision_p + wp_r
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

        # IMU
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        ax, ay, _ = linear_vel
        wx, wy, wz = angular_vel
        
        obs = np.concatenate((
                np.array([goal_distance, goal_angle, ax, ay, wx, wy, wz], dtype=np.float32),
                norm_lidar
            ), axis=0)
        
        #obs = (obs - obs.mean()) / (obs.std() + 1e-8)


        return obs
    
    

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()

