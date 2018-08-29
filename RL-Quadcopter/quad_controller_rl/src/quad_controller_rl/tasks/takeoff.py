'''Takeoff task.'''

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Takeoff(BaseTask):
    '''Simple task where the goal is to lift off the ground and reach a target height'''

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0]))
        
        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        
        # Task-specific parameters
        self.max_duration = 5.0  # secs
        self.weight_position = 0.4
        self.weight_velocity = 0.2
        self.target_z = 10.0  # target height (z position) to reach for successful takeoff
        self.target_velocity_z = 5.0
        self.last_timestamp = None
        self.last_position = None
        
    def reset(self):
        self.last_timestamp = None
        self.last_position = None

        return Pose(
                position=Point(0.0, 0.0, np.random.normal(0.5, 0.1)),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Calculate velocity
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03) # prevent divide by zero

        # Create state space and update lag variables
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = abs(self.target_z - pose.position.z)
        error_velocity = (self.target_velocity_z - velocity[2])**2
    
        # reward = zero for matching target z, -ve as you go farther, upto -20
        reward = -min(self.weight_position * error_position + self.weight_velocity * error_velocity, 20.0)

        if pose.position.z >= self.target_z:
            reward += 10.0
            done = True
        elif timestamp > self.max_duration:
            reward -= 10.0
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
