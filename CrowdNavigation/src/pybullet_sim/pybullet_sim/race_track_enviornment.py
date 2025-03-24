import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import os
from collections import deque
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ----- TUNABLE PARAMETERS FOR ENVIORMENT (not hyperparams) -----
ENV_BOUNDARY = 3.0          # Robot operates within -3..3 in x,y
MIN_DISTANCE = ENV_BOUNDARY * 0.5 # Spawn the robot and goal this distance apart (50% of the map apart)
EP_LENGTH = 100_000           # Max steps per episode
TIME_STEP = 1.0 / 240        # 240 Hz physics update rate
MAX_LINEAR_SPEED = 0.2 * 10      # m/s
MAX_ANGULAR_SPEED = 1  * 10    # rad/s
GOAL_REACHED_DIST = 0.2     # Robot is "at" goal if closer than this

# LiDAR specs
NUM_READINGS = 450                # 360° / 0.8° ≈ 450
MAX_LIDAR_RANGE = 12.0            # Up to ~12 m for black objects
MIN_LIDAR_RANGE = 0.03            # Minimum measurable distance
LIDAR_HEIGHT_OFFSET = 0.2         # Slightly above ground/robot’s base

# PENALTY & REWARD VALUES
PENALTY_COLLISION = -10     
PENALTY_TIME =0          
REWARD_GOAL_BONUS = 500
REWARD_DTG_POSITIVE = 10     # Reward for reducing distance to goal
REWARD_HTG_POSITIVE = 3    # Reward for facing goal
REWARD_ACTION_HIGH = 10   # Reward for forward & near-zero rotation
REWARD_ACTION_MED = 1       # Reward for forward & rotating

def is_traversable(grid):
    rows, cols = grid.shape


    # Find start (2) and goal (3) positions
    start, goal = None, None
    for y in range(rows):
        for x in range(cols):
            if grid[y, x] == 2:
                start = (y, x)
            elif grid[y, x] == 3:
                goal = (y, x)


    if not start or not goal:
        return False  # Missing start or goal


    # BFS setup
    queue = deque([start])
    visited = set([start])


    # Movement directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


    while queue:
        y, x = queue.popleft()
    
        # If we reached the goal, the grid is traversable
        if (y, x) == goal:
            return True


        # Explore neighbors
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols and (ny, nx) not in visited:
                if grid[ny, nx] in {0, 3}:  # Can move through empty space or goal
                    queue.append((ny, nx))
                    visited.add((ny, nx))

    return False  # If BFS completes without finding goal


def generate_random_grid(rows=10, cols=10, wall_prob=0.3, min_wall_ratio=0.8):
    """
    Generate a random grid with:
    - A solid wall border
    - A guaranteed path from a random start (2) to a random goal (3)
    - A minimum wall-to-open space ratio


    :param rows: Number of rows in the grid
    :param cols: Number of columns in the grid
    :param wall_prob: Probability of placing a wall (1) in the interior
    :param min_wall_ratio: Minimum fraction of walls in the grid
    :return: A 2D numpy array representing the grid
    """
    while True:  # Keep generating grids until a valid one is found
        # Initialize grid with walls
        grid = np.ones((rows, cols), dtype=int)


        # Randomize start and goal positions (inside the border)
        start = (random.randint(1, rows - 2), random.randint(1, cols - 2))
        goal = (random.randint(1, rows - 2), random.randint(1, cols - 2))


        while goal == start:  # Ensure start and goal are different
            goal = (random.randint(1, rows - 2), random.randint(1, cols - 2))


        grid[start] = 2  # Place start
        grid[goal] = 3  # Place goal


        # Randomly place walls while keeping start and goal free
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if (y, x) not in [start, goal] and random.random() < wall_prob:
                    grid[y, x] = 1  # Place wall


        # Ensure connectivity using DFS
        stack = [start]
        visited = set([start])


        while stack:
            y, x = stack.pop()
            if (y, x) == goal:
                break  # Stop when reaching the goal


            # Shuffle movement directions
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)


            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 1 <= ny < rows - 1 and 1 <= nx < cols - 1 and (ny, nx) not in visited:
                    grid[ny, nx] = 0  # Clear path
                    stack.append((ny, nx))
                    visited.add((ny, nx))


        # **Fix:** Ensure goal is not overwritten
        grid[goal] = 3  


        # **New Feature:** Check minimum wall fill
        total_cells = (rows - 2) * (cols - 2)  # Excluding border
        wall_count = np.count_nonzero(grid == 1)
        wall_ratio = wall_count / total_cells


        # If the wall ratio is too low, regenerate the grid
        if wall_ratio < min_wall_ratio:
            continue  # Retry with a new random grid


        # Check if the grid is traversable
        if is_traversable(grid):
            return grid  # Return only a valid grid
    

def print_grid(grid):
    
    
    """
    Print the grid in a readable format.
    """
    symbols = {0: ' ', 1: '█', 2: 'S', 3: 'G'}  # Empty, Wall, Start, Goal
    for row in grid:
        print("".join(symbols[cell] for cell in row))
    print("\n")  # Add newline
# Generate and print a random grid

 

class CrowdAvoidanceEnv(gym.Env):
    """Gym environment for a robot navigating a PyBullet scene."""

    metadata = {'render.modes': ['human']}

    # Dont worry about this code, it only runs once when the class is created
    def __init__(self, use_gui=False):
        super().__init__()
        self.physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.setTimeStep(TIME_STEP)
        p.setPhysicsEngineParameter(numSubSteps=5)

        self.past_observations = np.zeros((5, NUM_READINGS))  # Store last 5 LiDAR frames


        # [v, w]: linear & angular velocity commands, matches ros2 outputs
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]),
            dtype=np.float32
        )

        # Define observation space: [goal_dist, goal_angle, ax, ay, wx, wy, wz] + LiDAR readings
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([0, -math.pi, -5, -5, -5, -5, -5], dtype=np.float32),
                np.full(NUM_READINGS * 5, 0, dtype=np.float32)  # Includes 5 past LiDAR frames
            )),
            high=np.concatenate((
                np.array([5, math.pi, 5, 5, 5, 5, 5], dtype=np.float32),
                np.full(NUM_READINGS * 5, MAX_LIDAR_RANGE, dtype=np.float32)
            )),
            dtype=np.float32
        )

        self.robot = p.loadURDF(
            "../urdf/MicroROS.urdf",
            basePosition=[0, 0, 0.05],
            useFixedBase=False
        )
        p.changeDynamics(self.robot, -1, ccdSweptSphereRadius=0.05)


        self.goal_position = [0, 0, 0.05]
        self.max_steps = EP_LENGTH
        self.current_step = 0

        self.moving_obstacles = []

        self.reset()
    # Everytime the environment is reset, this code runs. You will mostly be modifying this code
    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        self.previous_linear_speed = 0.0
        self.previous_angular_speed = 0.0
        self.previous_goal_distance = 0
        self.current_step = 0

        plane_id = p.loadURDF("plane.urdf")  # Spawn ground plane

        # Generate a new environment grid
        grid_size = 50  # Define the size of the grid
        random_grid = generate_random_grid(grid_size, grid_size, wall_prob=0.4, min_wall_ratio=0.2)
        print_grid(random_grid)

        # Identify start (S) and goal (G) positions
        start_x, start_y, goal_x, goal_y = None, None, None, None
        for y in range(grid_size):
            for x in range(grid_size):
                if random_grid[y, x] == 2:  # Start position (S)
                    start_x, start_y = x, y
                elif random_grid[y, x] == 3:  # Goal position (G)
                    goal_x, goal_y = x, y

        if start_x is None or goal_x is None:
            raise ValueError("Grid generation failed to include start or goal position.")

        # Convert grid coordinates to world space (adjust scaling)
        cell_size = ENV_BOUNDARY * 2 / grid_size  # Map grid to world coordinates
        start_x = start_x * cell_size - ENV_BOUNDARY
        start_y = start_y * cell_size - ENV_BOUNDARY
        goal_x = goal_x * cell_size - ENV_BOUNDARY
        goal_y = goal_y * cell_size - ENV_BOUNDARY

        # Spawn goal marker
        self.goal_position = [goal_x, goal_y, 0.05]
        self.goal_marker = p.loadURDF("sphere_small.urdf", basePosition=self.goal_position, globalScaling=1)
        p.changeVisualShape(self.goal_marker, -1, rgbaColor=[0, 1, 0, 1])  # Green for goal

        # Spawn robot at the start position
        self.robot = p.loadURDF(
            "../urdf/MicroROS.urdf",
            basePosition=[start_x, start_y, 0.05],
            useFixedBase=False
        )

        # Prevents robot from sliding like it's on ice
        num_joints = p.getNumJoints(self.robot)
        for joint in range(num_joints):
            p.changeDynamics(self.robot, joint, lateralFriction=0.7)
        p.changeDynamics(self.robot, -1, lateralFriction=0.7)

        p.setPhysicsEngineParameter(enableConeFriction=True)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.0001)
        p.setCollisionFilterPair(self.robot, plane_id, -1, -1, enableCollision=False)

        # Spawn walls (static obstacles) with a larger size
        wall_z = 0.3  # Make walls taller
        wall_scale = 12.0 / 5  # Make walls larger in width & depth
        for y in range(grid_size):
            for x in range(grid_size):
                if random_grid[y, x] == 1:  # Wall
                    world_x = x * cell_size - ENV_BOUNDARY
                    world_y = y * cell_size - ENV_BOUNDARY
                    p.loadURDF("cube_small.urdf", basePosition=[world_x, world_y, wall_z], globalScaling=wall_scale)

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
        dtg_r, htg_r, action_r, time_penalty = 0, 0, 0, 0

        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        gx, gy, _ = self.goal_position
        goal_dist = math.sqrt((gx - robot_pos[0])**2 + (gy - robot_pos[1])**2)

        

        # Checks if distance to the goal has decreased 
        prev_dist = getattr(self, "previous_goal_distance", goal_dist)
        dist_improvement = prev_dist - goal_dist
        self.previous_goal_distance = goal_dist
        dtg_r = REWARD_DTG_POSITIVE * max(dist_improvement, 0) # ensures only positive improvements count
   

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

        if lin_vel > 0:
            action_r = 1
        
        # Min Distance Penalty
        lidar_scan = self._perform_lidar_scan()
        min_distance = np.min(lidar_scan)  # Closest object detected by LiDAR
        if min_distance < 0.5:
            action_r += -10
       

        # End episode if collision
        contacts = p.getContactPoints(self.robot)
        if contacts:
            contact_id = contacts[0][2]
            if contact_id != 0:
                self.last_collision = contact_id
                #print(f"Collision with object {contact_id} at {contacts[0][6:9]}")
                return PENALTY_COLLISION, False
            
        time_penalty = PENALTY_TIME

        # Give bonus for reaching goal and end
        if goal_dist < GOAL_REACHED_DIST:
            return REWARD_GOAL_BONUS + time_penalty, True

        reward = time_penalty + dtg_r + htg_r + action_r
        return reward, False
    

    
    def _perform_lidar_scan(self):
        """ Perform a simulated 360° LiDAR sweep with ~0.8° resolution and up to 12 m range.
        Returns an array of distances (one per angle).
        """

        # Get robot pose
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos

        # Store distances
        ranges = [] 

        # Sweep from -π to +π with ~0.8° steps
        for angle in np.linspace(-np.pi, np.pi, NUM_READINGS, endpoint=False):
            dx = np.cos(angle)
            dy = np.sin(angle)
            start_pos = [robot_x, robot_y, robot_z + LIDAR_HEIGHT_OFFSET]
            end_pos = [
                robot_x + dx * MAX_LIDAR_RANGE,
                robot_y + dy * MAX_LIDAR_RANGE,
                robot_z + LIDAR_HEIGHT_OFFSET
            ]

            # Ray test
            hit_result = p.rayTest(start_pos, end_pos)[0]
            hit_object_id = hit_result[0]
            hit_fraction = hit_result[2]

            # Compute distance
            distance = hit_fraction * MAX_LIDAR_RANGE if hit_object_id != -1 else MAX_LIDAR_RANGE
            distance = max(distance, MIN_LIDAR_RANGE)

            # Normalize so MIN_LIDAR_RANGE → 1 and MAX_LIDAR_RANGE → 0
            normalized_distance = 1 - (distance - MIN_LIDAR_RANGE) / (MAX_LIDAR_RANGE - MIN_LIDAR_RANGE)
            ranges.append(normalized_distance)

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
