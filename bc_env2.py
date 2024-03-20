import json
import numpy as np
import pybullet as p
import pybullet_data
import gym  
import time
from imitation.algorithms import bc
from imitation.data.types import Transitions


def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

import gym
from gym.spaces import Space
import numpy as np

class CustomObservationSpace(Space):
    """
    Custom observation space class to handle observations with specific shapes.
    """

    def __init__(self, low, high):
        assert low.shape == high.shape, "Low and high dimensions must match"
        self.low = low
        self.high = high
        super(CustomObservationSpace, self).__init__(shape=low.shape, dtype=np.float64)

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)


class CustomEnv:
    def __init__(self, json_filenames):
        # Initialize PyBullet physics simulation
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load drone model
        self.drone_id = p.loadURDF("./drone.urdf", [0, 0, 0])

        # Load obstacles at certain positions
        self.obstacle_ids = []
        obstacle_positions = [[0, 1, 0], [1, 2, 0], [1, 3, 0], [1, 6, 0],
                              [0, 6, 0], [-1, 2, 0], [-2, 3, 0]]
        for position in obstacle_positions:
            obstacle_id = p.loadURDF("./obstacle.urdf", position)
            self.obstacle_ids.append(obstacle_id)

        # LIDAR parameters
        self.num_rays = 5
        self.ray_length = 10.0
        self.goal_position = [0, 9, 0]

        # Load data from JSON files
        observations, actions, dones, next_observations = [], [], [], []
        for filename in json_filenames:
            with open("./data/"+filename, "r") as f:
                data = json.load(f)
                dones.extend([False] * (len(data["observations"])-1) + [True])
                # observations.extend(data["observations"])
                actions.extend(data["actions"])

                for i in range(len(data["observations"])):
                    data["observations"][i] = flatten_list(data["observations"][i])

                observations.extend(data["observations"])

                for i in range(len(data["observations"])):
                    if i < len(data['observations']) - 1:
                        next_observation = data["observations"][i+1]
                    else:
                        next_observation = data["observations"][i]
                    next_observations.append(next_observation)

        low_obs = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, -np.inf,-np.inf,-np.inf, np.inf])
        high_obs = np.array([np.inf, np.inf, np.inf, np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf,np.inf,np.inf, np.inf])

        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float64)

        # self.observation_space = CustomObservationSpace(low=low_obs, high=high_obs)

        self.action_space = gym.spaces.Discrete(4)

        # print(np.array(observations).shape)
        # print(np.array(actions).shape)

        # Create a Transitions object
        self.transitions = Transitions(
            obs=np.array(observations,dtype=np.float64),
            acts=np.array(actions),
            infos=np.array(observations,dtype=np.float64),  # You can adjust this if needed
            next_obs=np.array(next_observations),  # You can adjust this if needed
            dones=np.array(dones)  # You can adjust this if needed
        )

        # print("reached here\n");
        # Set up the behavior cloning algorithm
        self.bc_trainer = bc.BC(
            observation_space=self.observation_space,
            action_space=self.action_space,
            demonstrations=self.transitions,
            rng=np.random.default_rng()
        )

    def reset(self):
        # Reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load drone model
        self.drone_id = p.loadURDF("./drone.urdf", [0, 0, 0])

        # Load obstacles at certain positions
        self.obstacle_ids = []
        obstacle_positions = [[0, 1, 0], [1, 2, 0], [1, 3, 0], [1, 6, 0],
                              [0, 6, 0], [-1, 2, 0], [-2, 3, 0]]
        for position in obstacle_positions:
            obstacle_id = p.loadURDF("./obstacle.urdf", position)
            self.obstacle_ids.append(obstacle_id)

        # Set initial drone position and orientation
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 0], [0, 0, 0, 1])

        return self._get_observation()

    def step(self, action):
        # Perform action and simulate the environment
        # Update drone position based on action
        drone_position, _ = p.getBasePositionAndOrientation(self.drone_id)
        new_position = [drone_position[0] + action[0], drone_position[1] + action[1], drone_position[2]]
        p.resetBasePositionAndOrientation(self.drone_id, new_position, [0, 0, 0, 1])

        # Get observation, reward, done, and info
        observation = self._get_observation()
        reward = 0  # Define your reward function
        done = False  # Define your termination condition
        info = {}  # Additional information, if needed

        return observation, reward, done, info

    def _get_observation(self):
        # Get LIDAR readings as observation
        drone_position, orientation = p.getBasePositionAndOrientation(self.drone_id)
        vel = p.getBaseVelocity(self.drone_id)
        return [drone_position,orientation,vel]


    def collide(self,a,b):
        a = np.array(a)
        b = np.array(b)
        distance = np.linalg.norm(a-b)
        return distance <= 0.05

    def render(self, mode='human'):
        # Render the environment
        while True:
            self.render()

            # Get action from the behavioral cloning model
            action = self.bc_trainer.policy(observation)

            observation, reward, done, info = self.step(action)

            if self.collide(observation, self.goal_position):
                print("Goal reached!")
                break

if __name__ == '__main__':
    # Load data from 10 JSON files
    json_filenames = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json",
                      "data6.json", "data7.json", "data8.json", "data9.json", "data10.json"]

    # Create environment
    env = CustomEnv(json_filenames)
    env.reset()

    # Run the environment loop
    env.render()
