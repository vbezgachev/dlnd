'''Hover task.'''

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Hover(BaseTask):
    '''Simple task where the goal is to hover at desired position'''

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
        self.max_duration = 7.0  # secs
        self.target_position = np.array([0.0, 0.0, 10.0])
        self.max_error_position = 5.0
        self.position_weight = 0.5
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.accel_weight = 0.5
        self.last_timestamp = None
        self.last_position = None
        self.last_velocity = None

    def reset(self):
        self.last_timestamp = None
        self.last_position = None
        self.last_velocity = None

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

        # Calculate velocity and acceleration
        accel = np.array([linear_acceleration.x, linear_acceleration.y, linear_acceleration.z])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03) # prevent divide by zero

        # Create state space and update lag variables
        state = np.concatenate([position, orientation, velocity])
        self.last_timestamp = timestamp
        self.last_position = position
        self.last_velocity = velocity

        # Compute reward / penalty and check if this episode is complete
        done = False
        error_position = np.linalg.norm(self.target_position - position)
        sum_accel = np.linalg.norm(accel)

        reward = -(self.position_weight * error_position + \
                   self.accel_weight * sum_accel)

        if error_position > self.max_error_position:
            reward -= 50.0
            done = True
        elif timestamp > self.max_duration:
            reward += 50.0
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
