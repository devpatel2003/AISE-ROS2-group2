import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os

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

# LiDAR specs
NUM_READINGS = 450                # 360° / 0.8° ≈ 450
MAX_LIDAR_RANGE = 3.0            # Up to ~12 m for black objects
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.1         # Slightly above ground/robot’s base

# Reward & penalty weights        #-0.2, 500, 10, 3, 10, 1
REWARD_GOAL_BONUS = 100_000
REWARD_DTG_POSITIVE = 1    # Reward for reducing distance to goal, after * dist_improve number becomes very small
REWARD_HTG_POSITIVE = 1    # Reward for facing goal
REWARD_ACTION_HIGH = 1  # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 1       # Reward for forward & rotating
PENALTY_COLLISION = -100 
PENALTY_NEAR_COLLISION = -2  
PENALTY_TURN = 0
PENALTY_TIME = 0 #-1.125   
 

class MovingObstacle:
    """A small moving obstacle with random direction."""

    def __init__(self, position, robot):
        self.body = p.loadURDF("cube.urdf", position, globalScaling=0.2)
        self.robot = robot
        p.setCollisionFilterPair(self.body, self.robot, -1, -1, enableCollision=True)
        direction = np.array([
            random.uniform(-1, 1), # x direction
            random.uniform(-1, 1), # y direction
            0 # z direction
        ])

        self.direction = direction * OBSTACLE_SPEED * TIME_STEP # scale speed 


    def move(self):
        pos, _ = p.getBasePositionAndOrientation(self.body)
        new_pos = np.array(pos) + self.direction
        if abs(new_pos[0]) > ENV_BOUNDARY or abs(new_pos[1]) > ENV_BOUNDARY:
            self.direction *= -1
        p.resetBasePositionAndOrientation(self.body, new_pos, [0, 0, 0, 1])

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

        self.robot = p.loadURDF(
            "C:/Users/Dev/Documents/Personal/Projects/CrowdNavigation/src/pybullet_sim/urdf/MicroROS.urdf",
            basePosition=[0, 0, 0.05],
            useFixedBase=False
        )
        p.changeDynamics(self.robot, -1, ccdSweptSphereRadius=0.05)


        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.current_step = 0

        self.moving_obstacles = []

        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.previous_linear_speed = 0.0
        self.previous_angular_speed = 0.0
        self.previous_goal_distance = None
        self.current_step = 0

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

        p.setPhysicsEngineParameter(enableConeFriction=True)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        p.setCollisionFilterPair(self.robot, plane_id, -1, -1, enableCollision=False)


        # Create moving obstacles
        self.moving_obstacles.clear()
        for _ in range(NUM_MOVING_OBSTACLES):
            ox = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            oy = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            obstacle = MovingObstacle(position=[ox, oy, 0.03], robot=self.robot)
            self.moving_obstacles.append(obstacle)
        
        # Disable collisions between the goal_marker and each obstacle
        for obs in self.moving_obstacles:
            p.setCollisionFilterPair(obs.body, self.goal_marker, -1, -1, enableCollision=False)
            p.setCollisionFilterPair(obs.body, self.robot, -1, -1, enableCollision=True)


        
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

        # Move each obstacle
        for obs in self.moving_obstacles:
            obs.move()

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
    
    def _compute_reward(self, action):
        lin_vel, ang_vel = action
        dtg_r, htg_r, action_r, time_penalty, min_distance_p  = 0, 0, 0, 0, 0

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)

        

        # Checks if distance to the goal has decreased 
        prev_dist = getattr(self, "previous_goal_distance", goal_dist)
        dist_improvement = prev_dist - goal_dist
        self.previous_goal_distance = goal_dist
        dtg_r = REWARD_DTG_POSITIVE * max(dist_improvement*1000, 0) #*1000 because dist improvement is 0.00002
       


        # Heading to goal reward
        goal_angle = self.get_goal_angle()  # Between -pi and pi
        previous_goal_angle = getattr(self, "previous_goal_angle", goal_angle)
        angle_improvement = (abs(previous_goal_angle) - abs(goal_angle))  # Increase reward for reducing error
        self.previous_goal_angle = goal_angle 
        htg_r = REWARD_HTG_POSITIVE* np.cos(goal_angle) # facing goal (0) = +1, facing away (180) = -1 
        
        # Action reward
        if lin_vel >= 0 and abs(goal_angle) < np.deg2rad(5):
            # if linear_speed >= 0 and angular_speed == 0:
            action_r = REWARD_ACTION_HIGH * lin_vel
        elif lin_vel >= 0:
            action_r = REWARD_ACTION_MED * lin_vel

        
        lidar_scan = self._perform_lidar_scan()
        min_distance = np.min(lidar_scan)  # Closest object detected by LiDAR

        if min_distance < 0.2:
            min_distance_p = PENALTY_NEAR_COLLISION
        elif min_distance < 0.3:
            min_distance_p = PENALTY_NEAR_COLLISION * (0.3 - min_distance)
       

        contacts = p.getContactPoints(self.robot)

        if contacts:
            contact_id = contacts[0][2]
            if contact_id != 0:
                self.last_collision = contact_id
                #print(f"Collision with object {contact_id} at {contacts[0][6:9]}")
                return PENALTY_COLLISION, True
            
        #time_penalty = -10 * (self.current_step / EP_LENGTH)  #  Larger time penalty over time
        time_penalty = PENALTY_TIME

        if goal_dist < GOAL_REACHED_DIST:
            return REWARD_GOAL_BONUS + time_penalty, True
        reward = action_r + dtg_r + htg_r + action_r + min_distance_p + time_penalty
        #print(dtg_r , htg_r , min_distance_p , time_penalty)
        return reward, False
    
    def _perform_lidar_scan(self):
        """ Perform a simulated 360° LiDAR sweep with ~0.8° resolution and up to 12 m range.
        Returns an array of distances (one per angle).
        """

        # Get robot position and orientation
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos
        _, _, robot_yaw = p.getEulerFromQuaternion(robot_ori)  # Extract yaw angle

        start_pos = [robot_x, robot_y, robot_z + LIDAR_HEIGHT_OFFSET]
        angles = np.linspace(-np.pi, np.pi, NUM_READINGS, endpoint=False)

        # Rotate scan angles based on robot orientation
        rotated_angles = angles + robot_yaw  

        # Compute end positions with the rotated angles
        end_positions = [
            [robot_x + np.cos(angle) * MAX_LIDAR_RANGE,
            robot_y + np.sin(angle) * MAX_LIDAR_RANGE,
            robot_z + LIDAR_HEIGHT_OFFSET]
            for angle in rotated_angles
        ]

        # Perform batch raycast
        results = p.rayTestBatch([start_pos] * NUM_READINGS, end_positions)

        ranges = []
        for res in results:
            hit_object_id = res[0]
            hit_fraction = res[2]
            distance = hit_fraction * MAX_LIDAR_RANGE if hit_object_id != -1 else MAX_LIDAR_RANGE
            distance = max(distance, MIN_LIDAR_RANGE)
            ranges.append(distance)

        return np.array(ranges, dtype=np.float32)

    def _get_observation(self):
        # LiDAR scan 
        lidar_scan = self._perform_lidar_scan()

        self.past_observations = np.roll(self.past_observations, shift=-1, axis=0)
        self.past_observations[-1] = lidar_scan

        # Compute goal distance, angle, IMU, etc. as before
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
                self.past_observations.flatten()
            ), axis=0)
        
        #obs = (obs - obs.mean()) / (obs.std() + 1e-8)


        return obs
    
    

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
