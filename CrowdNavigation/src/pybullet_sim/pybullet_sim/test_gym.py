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
MIN_DISTANCE = ENV_BOUNDARY * 0.3 # Spawn the robot and goal this distance apart
EP_LENGTH = 10000           # Max steps per episode
TIME_STEP = 1.0 / 240.0
WHEEL_BASE = 0.14           # Distance between left & right wheels
MAX_LINEAR_SPEED = 0.2      # m/s
MAX_ANGULAR_SPEED = 1.0     # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

PENALTY_COLLISION = -10
REWARD_GOAL_BONUS = 100
REWARD_DTG_POSITIVE = 1     # Reward for reducing distance to goal
REWARD_HTG_POSITIVE = 1     # Reward for improving heading
REWARD_ACTION_HIGH = 2      # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 1       # Reward for forward & rotating

class MovingObstacle:
    """Create moving obstacle with random direction."""

    def __init__(self, position, robot):
        self.body = p.loadURDF("cube.urdf", position, globalScaling=0.2)
        self.robot = robot
        p.setCollisionFilterPair(self.body, self.robot, -1, -1, enableCollision=True)
        self.direction = np.array([
            random.uniform(-0.02, 0.02), #x direction
            random.uniform(-0.02, 0.02),   #y direction
            0 # z direction
        ])

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

        # [v, w]: linear & angular velocity commands, matches ros2 outputs
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]),
            dtype=np.float32
        )

        # [goal_dist, goal_angle, ax, ay, wx, wy, wz]
        self.observation_space = spaces.Box(
            low=np.array([0, -math.pi, -5, -5, -5, -5, -5]),
            high=np.array([5,  math.pi,  5,  5,  5,  5,  5]),
            dtype=np.float32
        )

        self.robot = p.loadURDF(
            "C:/Users/Dev/Documents/Personal/Projects/CrowdNavigation/src/pybullet_sim/urdf/MicroROS.urdf",
            basePosition=[0, 0, 0.05],
            useFixedBase=False
        )

        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.current_step = 0

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
            start_x = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            start_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            goal_x = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
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
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

        self.previous_goal_distance = distance
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

    def _compute_reward(self, action):
        lin_vel, ang_vel = action
        dtg_r, htg_r, action_r = 0, 0, 0

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)

        # Checks if distance to the goal has decreased 
        prev_dist = getattr(self, "previous_goal_distance", goal_dist)
        dist_improvement = prev_dist - goal_dist
        self.previous_goal_distance = goal_dist

        if dist_improvement > 0:
            dtg_r = REWARD_DTG_POSITIVE

        goal_angle = self.get_goal_angle()
        prev_angle = getattr(self, "previous_goal_angle", goal_angle)
        angle_diff = abs(prev_angle) - abs(goal_angle)
        self.previous_goal_angle = goal_angle

        if angle_diff > 0:
            htg_r = REWARD_HTG_POSITIVE # Encourages facing towards to goal

        if lin_vel >= 0 and -2/32 <= ang_vel <= 2/32:
            action_r = REWARD_ACTION_HIGH # Encorages going forward & near-zero rotation
        elif lin_vel >= 0:
            action_r = REWARD_ACTION_MED

        collision_pen = 0
        contacts = p.getContactPoints(self.robot)
        if contacts:
            contact_id = contacts[0][2]
            if contact_id != 0 and contact_id != getattr(self, "last_collision", None):
                self.last_collision = contact_id
                collision_pen = PENALTY_COLLISION
                print(f"Collision with object {contact_id} at {contacts[0][6:9]}")

        if goal_dist < GOAL_REACHED_DIST:
            return REWARD_GOAL_BONUS, True

        reward = dtg_r + htg_r + action_r + collision_pen
        return reward, False

    def _get_observation(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - pos[0])**2 + (gy - pos[1])**2)
        goal_angle = self.get_goal_angle()

        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        ax, ay, _ = lin_vel
        wx, wy, wz = ang_vel

        # Observations that the RL model uses 
        return np.array([
            round(goal_dist, 3),
            goal_angle,
            ax, ay,
            wx, wy, wz
        ], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
