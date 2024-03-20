import json
import numpy as np
import pybullet as p
import pybullet_data
import gym  
import time
from imitation.algorithms import bc
from imitation.data.types import Transitions

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
        observations, actions = [], []
        for filename in json_filenames:
            with open(filename, "r") as f:
                data = json.load(f)
                observations.extend(data["observations"])
                actions.extend(data["actions"]) 

        # for filename in json_filenames:
        #     with open(filename, "r") as f:
        #         data = json.load(f)
        #         # Ensure observations are flattened
        #         for obs_sequence in data["observations"]:
        #             for obs_value in obs_sequence:
        #                 observations.append(obs_value)
        #         actions.extend(data["actions"])

        # print(observations[0])

        low_obs = np.array([-np.inf, -np.inf, -np.inf,    # Drone position (x, y, z)
                            -np.inf, -np.inf, -np.inf,   # Drone orientation (roll, pitch, yaw)
                            -np.inf, -np.inf, -np.inf,   # Drone velocity (vx, vy, vz)
                            0.0])                         # Nearest obstacle distance
        high_obs = np.array([np.inf, np.inf, np.inf,
                             np.inf, np.inf, np.inf,
                             np.inf, np.inf, np.inf,
                             np.inf])
        self.observation_space = gym.spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)

        # Create a Transitions object
        self.transitions = Transitions(
            obs=np.array(observations),
            acts=np.array(actions),
            infos=np.array([]),  # You can adjust this if needed
            next_obs=np.array([]),  # You can adjust this if needed
            dones=np.array([])  # You can adjust this if needed
        )

        # Set up the behavior cloning algorithm
        self.bc_trainer = bc.BC(
            observation_space=self.observation_space(),
            action_space=self.action_space(),
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
    json_filenames = ["./data/data1.json", "./data/data2.json", "./data/data3.json", "./data/data4.json", "./data/data5.json",
                      "./data/data6.json", "./data/data7.json", "./data/data8.json", "./data/data9.json", "./data/data10.json"]

    # Create environment
    env = CustomEnv(json_filenames)
    env.reset()

    # Run the environment loop
    env.render()
