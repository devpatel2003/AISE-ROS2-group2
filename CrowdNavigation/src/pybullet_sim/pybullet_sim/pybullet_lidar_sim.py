#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import time
import os
import ament_index_python.packages
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import random



ENV_BOUNDARY = 2.0  # Environment is from -2 to 2 in x and y
ENV_OBSTACLE = 8  # Obstacle size
MIN_DISTANCE = ENV_BOUNDARY * 0.7  # Robot and goal must be at least 50% of the boundary apart


class MovingObstacle:
    def __init__(self, name, position, robot):
        self.body = p.loadURDF("cube.urdf", position, globalScaling=0.2)

        self.robot = robot

        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

        # Disable obstacle-obstacle collisions
        #p.setCollisionFilterGroupMask(self.body, -1, collisionFilterGroup=2, collisionFilterMask=1)

        # Enable collision with the robot only
        p.setCollisionFilterPair(self.body, self.robot, -1, -1, enableCollision=True)


        # Random movement direction (small steps)
        self.direction = np.array([
            random.uniform(-0.02, 0.02), 
            random.uniform(-0.02, 0.02), 
            0
        ])

    def move(self):
        pos, _ = p.getBasePositionAndOrientation(self.body)
        new_pos = np.array(pos) + self.direction

        # Reverse direction if it reaches boundary
        if abs(new_pos[0]) > ENV_BOUNDARY or abs(new_pos[1]) > ENV_BOUNDARY:
            self.direction *= -1  # Flip direction

        p.resetBasePositionAndOrientation(self.body, new_pos, [0, 0, 0, 1])

class LidarSim(Node):
    def __init__(self):
        super().__init__('lidar_sim')
        self.publisher_ = self.create_publisher(LaserScan, '/scan', 10)

        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)


        # Initialize PyBullet
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # kinda bad code but keep trying combinations until distance is good
        while True:
            start_x = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            start_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

            # Tentatively set goal far away
            goal_x = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)
            goal_y = random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5)

            # Calculate distance
            distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)

            # If goal is far enough from the robot, break the loop
            if distance >= MIN_DISTANCE:
                break

        self.goal_position = [goal_x, goal_y]

        # Load robot URDF
        package_path = ament_index_python.packages.get_package_share_directory('yahboomcar_description')
        urdf_path = os.path.join(package_path, 'urdf', 'MicroROS.urdf')
        self.robot = p.loadURDF(urdf_path, 
                                basePosition=[start_x, start_y, 0.05], 
                                useFixedBase=False)
        
        #self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        

        self.goal_visual = p.loadURDF("sphere_small.urdf", 
                              basePosition=[self.goal_position[0], self.goal_position[1], 0.1], 
                              globalScaling=1.0)

        self.obstacles = [
            MovingObstacle(f"obstacle_{i}", [
                random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5),
                random.uniform(-ENV_BOUNDARY + 0.5, ENV_BOUNDARY - 0.5),
                0.1
            ], self.robot)
            for i in range(ENV_OBSTACLE)  # Create 5 obstacles
        ]


        # Simulation timestep
        self.timer = self.create_timer(0.1, self.update_lidar)



    
    
    def get_lidar_scan(self):
        """Simulates LiDAR scanning and returns range data."""
        ranges = []
        num_readings = 360
        angle_min = -np.pi
        angle_max = np.pi

        # Get robot position
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)
        robot_x, robot_y, robot_z = robot_pos
        

        for angle in np.linspace(angle_min, angle_max, num_readings):
            direction = [np.cos(angle), np.sin(angle), 0]
            
            start_pos = [robot_x, robot_y , robot_z + 0.1]  # Adjusted to robot's position
            end_pos = [robot_x  + direction[0] * 5, robot_y + direction[1] * 5, robot_z + 0.1]

            hit = p.rayTest(start_pos, end_pos)[0]



            if hit[0] != -1:
                distance = hit[2]  # Distance to the object
            else:
                distance = 5.0  # Max LiDAR range

            ranges.append(distance)


        return ranges  # Returns LiDAR scan data

    
    def get_robot_state(self):
        """Returns the robot's observation: LiDAR readings + goal distance + goal angle."""
        
        # Get LiDAR scan data
        scan_data = self.get_lidar_scan()

        # Get robot position
        robot_pos, _ = p.getBasePositionAndOrientation(self.robot)

        # Compute distance and angle to goal
        goal_dx = self.goal_position[0] - robot_pos[0]
        goal_dy = self.goal_position[1] - robot_pos[1]
        goal_distance = np.sqrt(goal_dx**2 + goal_dy**2)
        goal_angle = np.arctan2(goal_dy, goal_dx)

        # Return observation as (LiDAR readings + goal distance + goal angle)
        return scan_data + [goal_distance, goal_angle]


    def update_lidar(self):
        """Simulates a LiDAR sensor in PyBullet."""

        # Move each obstacle
        for obs in self.obstacles:
            obs.move()

        # Simulate physics step
        p.stepSimulation()

        contact_points = p.getContactPoints(self.robot)
        if contact_points:
            # Get the unique object ID of the first contact
            hit_object_id = contact_points[0][2]

            # Print only if it's a new collision and ignore ground
            if hit_object_id != 0 and hit_object_id != getattr(self, "last_collision", None) :
                self.last_collision = hit_object_id
                print(f"?? Collision detected! Robot hit object {hit_object_id} at {contact_points[0][6:9]}")

                

        # Get robot observation
        observation = self.get_robot_state()
        goal_distance = observation[-2]  # Last second value in list
        goal_angle = observation[-1]  # Last value in list

       

        # Publish to ROS 2
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = "lidar_frame"
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = (2 * np.pi) / 360
        scan_msg.range_min = 0.1
        scan_msg.range_max = 5.0
        scan_msg.ranges = observation[:-2]  # Everything except goal distance & angle


        self.publisher_.publish(scan_msg)
        # Publish `map -> base_link`
        tf_msg_map = TransformStamped()
        tf_msg_map.header.stamp = self.get_clock().now().to_msg()
        tf_msg_map.header.frame_id = "map"
        tf_msg_map.child_frame_id = "base_link"

        # Get the robot's position from PyBullet
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot)

        tf_msg_map.transform.translation.x = robot_pos[0]
        tf_msg_map.transform.translation.y = robot_pos[1]
        tf_msg_map.transform.translation.z = robot_pos[2]  # Keep actual height

        # Convert PyBullet quaternion (x, y, z, w) to ROS format
        tf_msg_map.transform.rotation.x = robot_orient[0]
        tf_msg_map.transform.rotation.y = robot_orient[1]
        tf_msg_map.transform.rotation.z = robot_orient[2]
        tf_msg_map.transform.rotation.w = robot_orient[3]

        # Send `map -> base_link`
        self.tf_broadcaster.sendTransform(tf_msg_map)

        # Publish `base_link ? lidar_frame`
        tf_msg_lidar = TransformStamped()
        tf_msg_lidar.header.stamp = self.get_clock().now().to_msg()
        tf_msg_lidar.header.frame_id = "base_link"
        tf_msg_lidar.child_frame_id = "lidar_frame"
        tf_msg_lidar.transform.translation.x = 0.0
        tf_msg_lidar.transform.translation.y = 0.0
        tf_msg_lidar.transform.translation.z = 0.3  # Slightly above base_link
        tf_msg_lidar.transform.rotation.w = 1.0

        # Send `base_link -> lidar_frame`
        self.tf_broadcaster.sendTransform(tf_msg_lidar)
            


def main(args=None):
    rclpy.init(args=args)
    lidar_sim = LidarSim()
    rclpy.spin(lidar_sim)
    lidar_sim.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
