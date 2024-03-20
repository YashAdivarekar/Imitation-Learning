import gym
import pybullet as p
import pybullet_data
import time
import json
import numpy as np

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
        # obstacle_positions = [[0, 1, 0], [1, 2, 0], [1, 3, 0], [1, 6, 0],
        #                       [0, 6, 0], [-1, 2, 0], [-2, 3, 0]]
        obstacle_positions = [[0, 1, 0], [1, 2, 0], [-2, 3, 0]]
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
        # obstacle_positions = [[0, 1, 0], [1, 2, 0], [1, 3, 0], [1, 6, 0],
        #                       [0, 6, 0], [-1, 2, 0], [-2, 3, 0]]
        obstacle_positions = [[0, 1, 0], [1, 2, 0], [-2, 3, 0]]
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
        print(drone_position,lidar_readings[0])
        return lidar_readings[0]

if __name__ == '__main__':
    env = CustomEnv('data11.json')
    env.reset()

    while True:
        env.render()

        # Get user input
        key = p.getKeyboardEvents()
        if p.B3G_RIGHT_ARROW in key and key[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            action = [0.15, 0.0]  # Move right
        elif p.B3G_LEFT_ARROW in key and key[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            action = [-0.15, 0.0]  # Move left
        elif p.B3G_UP_ARROW in key and key[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            action = [0.0, 0.15]  # Move forward
        elif p.B3G_DOWN_ARROW in key and key[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            action = [0.0, -0.15]  # Move backward
        else:
            continue

        # action = bc_trainer.policy(observation)

        observation, reward, done, info = env.step(action)
    
        if env.collide(observation[0],env.goal_position):
            env.save_simulation_data()
            print("Goal reached!")
            break

    # env.close()
        


