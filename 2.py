import gym
import pybullet as p
import pybullet_data
import time
import torch
import json
from imitation.algorithms import bc
import numpy as np
from typing import Dict, Optional
from typing import Any
import numpy as np
import torch
import json
import gymnasium as gym
from gymnasium.spaces import Box


class ObservationMatchingEnv(gym.Env):
    def __init__(self, num_options: int = 2):
        self.state = None
        self.num_options = num_options
        self.observation_space = gym.spaces.Box(low=-10.0, high=20.0, shape=(3,))
        # self.action_space = Box(-10.0, 20.0, shape=(num_options,))
        self.action_space = gym.spaces.Discrete(4)

class CustomEnv(gym.Env):
    def __init__(self,json_filename="simulation.json"):
        super(CustomEnv, self).__init__()

        # Initialize the PyBullet physics simulation
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the drone model
        self.drone_id = p.loadURDF("./drone.urdf", [0, 0, 0])

        self.json_filename = json_filename
        self.simulation_data = {"observations": [], "actions": []}


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
        self.lidar_noise = 0.01  # Noise added to LIDAR readings
        self.goal_position = [0, 9, 0]

    def record_simulation_data(self, observation, action):
        self.simulation_data["observations"].append(observation)
        self.simulation_data["actions"].append(action)

    def save_simulation_data(self):
        with open(self.json_filename, "w") as f:
            json.dump(self.simulation_data, f)

    def reset(self):
        # Reset the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load the drone model
        self.drone_id = p.loadURDF("./drone.urdf", [0, 0, 0])

        # Load obstacles at certain positions
        self.obstacle_ids = []
        obstacle_positions = [[0, 1, 0], [1, 2, 0], [1, 3, 0], [1, 6, 0],
                              [0, 6, 0], [-1, 2, 0], [-2, 3, 0]]
        for position in obstacle_positions:
            obstacle_id = p.loadURDF("./obstacle.urdf", position)
            self.obstacle_ids.append(obstacle_id)

        self.goal_position = [0, 9, 0]
        # Set initial drone position and orientation
        p.resetBasePositionAndOrientation(self.drone_id, [0, 0, 0], [0, 0, 0, 1])
        self.simulation_data = {"observations": [], "actions": []}

        return self._get_observation()

    def step(self, action):
        # Perform action and simulate the environment
        # Update drone position and orientation based on action
        drone_position, drone_orientation = p.getBasePositionAndOrientation(self.drone_id)
        new_position = [drone_position[0] + action[0], drone_position[1] + action[1], drone_position[2]]
        p.resetBasePositionAndOrientation(self.drone_id, new_position, drone_orientation)

        # Get LIDAR readings
        lidar_readings = self._get_lidar_readings(drone_position, goal_position=self.goal_position)

        # Get observation, reward, done, and info
        observation = self._get_observation()
        observation.append(lidar_readings)
        reward = 0  # You can define a reward function based on your requirements
        done = False  # Define your termination condition
        info = {}  # Additional information, if needed

        self.record_simulation_data(observation, action)
        return observation, reward, done, info

    def collide(self,a,b):
        a = np.array(a)
        b = np.array(b)
        distance = np.linalg.norm(a-b)
        return distance <= 0.09

    def render(self, mode='human'):
        # Render the environment
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 1],
                                                          distance=3.0,
                                                          yaw=90,
                                                          pitch=-45,
                                                          roll=0,
                                                          upAxisIndex=2)

        projection_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=1.0,
                                                         nearVal=0.1,
                                                         farVal=100.0)

        img = p.getCameraImage(width=800,
                               height=600,
                               viewMatrix=view_matrix,
                               projectionMatrix=projection_matrix)

        width, height, rgb, depth, seg = img
        rgb_array = rgb.reshape((height, width, 4))[:, :, :3]

        if mode == 'human':
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            time.sleep(0.01)  # You might need a small delay to allow rendering to complete
        elif mode == 'rgb_array':
            return rgb_array

    def _get_observation(self):
        # Get LIDAR readings as observation
        drone_position, orientation = p.getBasePositionAndOrientation(self.drone_id)
        vel = p.getBaseVelocity(self.drone_id)
        return [drone_position,orientation,vel]

    def _get_lidar_readings(self, drone_position, goal_position):
        """
        Perform LIDAR ray casting and return the distance readings.
        """
        # Calculate the direction from drone position to goal position
        goal_direction = np.array(goal_position) - np.array(drone_position)
        goal_direction /= np.linalg.norm(goal_direction)  # Normalize direction vector


        # Calculate the angle between the current direction and the goal direction in the xy-plane
        goal_angle = np.arctan2(goal_direction[1], goal_direction[0])

        # Adjust the range of angles based on the direction towards the goal
        start_angle = goal_angle  # Start angle 45 degrees to the left of the goal direction
        end_angle = goal_angle  # End angle 45 degrees to the right of the goal direction

        lidar_readings = []
        lidar_angles = []

        for angle in np.linspace(start_angle, end_angle, self.num_rays, endpoint=True):
            ray_direction = np.array([np.cos(angle), np.sin(angle), 0])  # 2D LIDAR, ignore z-axis
            ray_end = drone_position + self.ray_length * ray_direction

            # Cast ray and get hit information
            p.addUserDebugLine(drone_position, ray_end, [1, 0, 0], 1)
            results = p.rayTest(drone_position, ray_end)

            # Check if the hit object is an obstacle
            obstacle_hit = False
            for result in results:
                hit_object = result[0]
                if hit_object in [self.drone_id] + self.obstacle_ids:
                    obstacle_hit = True
                    lidar_readings.append(result[2])  # Distance to the obstacle
                    break

            if not obstacle_hit:
                lidar_readings.append(self.ray_length)  # No obstacle detected, set reading to maximum range

            lidar_angles.append(angle)
        # print(lidar_readings[0])
        # print(drone_position,lidar_readings[0])
        return lidar_readings[0]

def flatten_list(lst):
    flattened_list = []
    for item in lst:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

if __name__ == '__main__':
    env = CustomEnv('data11.json')
    env.reset()

    env2 = ObservationMatchingEnv()

    # json_filenames = ["data1.json", "data2.json", "data3.json", "data4.json", "data5.json",
    #                   "data6.json", "data7.json", "data8.json", "data9.json", "data10.json"]
    
    json_filenames = ["data11.json"]
    
    observations, actions, dones, next_observations = [], [], [], []

    for filename in json_filenames:
        with open("./" + filename, "r") as f:
            data = json.load(f)
            dones.extend([False] * (len(data["observations"]) - 1) + [True])

            for i in range(len(data["observations"])):
                data["observations"][i] = flatten_list(data["observations"][i])
                temp = []
                # for j in range(3):
                for j in range(2):
                    temp.append(data["observations"][i][j])
                temp.append(data["observations"][i][-1])
                data["observations"][i] = temp

            observations.extend(data["observations"])
            next_observations.extend(data["observations"][1:] + [data["observations"][-1]])

            for i in range(len(data["actions"])):
                if data["actions"][i] == [0.15, 0.0]:
                    data["actions"][i] = 0
                elif data["actions"][i] == [-0.15, 0.0]:
                    data["actions"][i] = 1
                elif data["actions"][i] == [0.0, 0.15]:
                    data["actions"][i] = 2
                elif data["actions"][i] == [0.0, -0.15]:
                    data["actions"][i] = 3

            actions.extend(data["actions"])

    # observations.insert(0, [0,0,0,1])
    observations.insert(0, [0,0,1])
    actions.insert(0, 2)
    dones.insert(0, False)
    # next_observations.insert(1, [0,0.15,0,0.85])
    next_observations.insert(1, [0,0.15,0.85])

    observations = np.array(observations)
    actions = np.array(actions) 
    dones = np.array(dones)
    next_observations = np.array(next_observations)

    transitions = {
                'obs': observations,
                'acts': actions,
                'infos': observations,  # You can adjust this if needed
                'next_obs': next_observations,  # You can adjust this if needed
                'dones': dones  # You can adjust this if needed
                
            }

    bc_trainer = bc.BC(
        observation_space=env2.observation_space,
        action_space=env2.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng()
    )

    observation = [0,0,1]
    # observation = [0,0,0,1]
    for i in range(200):

        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Pass observation through the feature extractor
        flat_features = bc_trainer.policy.features_extractor(obs_tensor)

        # Pass flattened features through the MLP extractor
        policy_features, _ = bc_trainer.policy.mlp_extractor(flat_features)

        # Pass MLP features through the action network to get action predictions
        action_logits = bc_trainer.policy.action_net(policy_features)

        # Convert logits to probabilities (if needed)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Get the predicted action
        # print(action_probs)
        predicted_action = torch.argmax(action_probs, dim=-1).item()
        # print("Predicted Action:", predicted_action)

        if predicted_action==0:
            action = [0.15, 0.0]  # Move right
        elif predicted_action==1:
            action = [-0.15, 0.0]  # Move left
        elif predicted_action==2:
            action = [0.0, 0.15]  # Move forward
        elif predicted_action==3:
            action = [0.0, -0.15]  # Move backward

        observation, reward, done, info = env.step(action)

        # print(observation)
        # temp = [observation[0][0],observation[0][1],observation[0][2],observation[-1]]
        temp = [observation[0][0],observation[0][1],observation[-1]]
        temp = np.array(temp)
        observation = temp