'''Landing task.'''

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    '''Simple task where the goal is land the quadcopter'''

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
        self.max_duration = 10.0  # secs
        self.target_position = np.array([0.0, 0.0, 0.1])
        self.max_error_position = 15.0
        self.position_weight = 0.5
        self.target_velocity = np.array([0.0, 0.0, -2.0]) # less than 2 brings worse results
        self.velocity_weight = 0.2
        self.last_timestamp = None
        self.last_position = None

    def reset(self):
        self.last_timestamp = None
        self.last_position = None

        return Pose(
            position=Point(0.0, 0.0, np.random.normal(0.5, 0.1) + 10),
            orientation=Quaternion(0.0, 0.0, 0.0, 0.0)
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Calculate velocity
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03) # prevent divide by zero

        # we expect that the next reached position is less than the previous one
        reward = 0
        if self.last_position is not None and pose.position.z > self.last_position[2]:
            reward -= 10

        # Create state space and update lag variables
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = abs(np.linalg.norm(self.target_position - position))
        error_velocity = np.linalg.norm(self.target_velocity - velocity)**2

        reward -= (self.position_weight * error_position + \
                   self.velocity_weight * error_velocity)

        if pose.position.z <= self.target_position[2]: # z position
            if abs(velocity[2]) < abs(self.target_velocity[2]):
                reward += 50.0
            done = True
        elif error_position > self.max_error_position:
            reward -= 50.0
            done = True
        elif timestamp > self.max_duration:
            reward -= 50.0

        if timestamp > self.max_duration:
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
