### 1. Drive to goal
- Decided on using PyBullet to simulate robot movements as it is more lightweight and less computationally expensive as compared to Gazebo

- Create a simulation that spawns in the Yahboomcar and a goal point

- Used a Soft Actor Critic (SAC) network to learn a policy that guides the robot to towards a goal

    - Action space: velocity, angular velocity (continuous)

    - Observation space: distance to goal, angle to goal, +5 IMU values

    - Reward function: +1 for decreasing distance and angle, +2/1 for turning, +1/2 for moving straight

https://github.com/user-attachments/assets/d0fdad22-a580-4358-a8d1-9b4450599c38

This final model took 1 hour to train, this is the equivalent to 100k steps of the simulation, on average the robot can navigate to the goal in 800 steps. However, while 1 hour seems short, it took maybe 20 hours of experimenting to see what reward function works best.

The next objective is to add static objects, and continue training the model to avoid collisions. This will involve increasing the observation space to include lidar data. My guess is this will take a lot longer to train due to the addition of 360 new inputs. Additionally, this model can be repurposed to be used within a ROS environment since the actions outputted are equivalent to the cmd_vel.linear and cmd_vel.angular msg types found in geometry_msgs.

