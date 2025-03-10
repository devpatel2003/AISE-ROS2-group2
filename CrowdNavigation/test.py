import pybullet as p
import pybullet_data
import numpy as np

# Start PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Spawn a test cube
cube = p.loadURDF("cube_small.urdf", [0, 1, 0.1])

# Define a raycast test
robot_pos = [0, 0, 0.2]
direction = [0, 1, 0]  # Forward ray
end_pos = [robot_pos[0] + direction[0] * 5, robot_pos[1] + direction[1] * 5, robot_pos[2]]

hit = p.rayTest(robot_pos, end_pos)[0]

print(f"Ray Start: {robot_pos} | Ray End: {end_pos} | Hit: {hit}")
