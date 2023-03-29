from dataclasses import dataclass
import random
from time import time
from typing import Callable, List, Optional, Protocol, Tuple, TypedDict

import gym
import numpy as np
import pygame
from episode_manager import EpisodeManager
from episode_manager.episode_manager import Action, WorldState
from gym.spaces import Box, Dict, Discrete
from gym.utils import seeding


class VisionModule(Protocol):
    """
    Defines protocol (interface) for a vision module that is injected to
    the environment to provide vision encoding and selected action postprocessing
    """

    output_shape: Tuple
    high: float
    low: float

    def __call__(self, input: WorldState) -> np.ndarray:
        """
        Returns the vision module encoded vector output based on the current step's world
        state information
        """

        return np.zeros((self.output_shape))

    def postprocess_action(self, action: Action) -> Action:
        """
        Perform any postprocessing on the action based on stored auxilliary information from the vision module
        """
        # return the same action by default

        return action

    def get_auxilliary_render(self) -> pygame.Surface:
        """
        Returns a pygame surface that visualizes auxilliary predections from the vision module
        """
        raise NotImplementedError


class CarlaEnvironmentConfiguration(TypedDict):
    continuous_speed_range: Tuple[float, float]
    continuous_steering_range: Tuple[float, float]
    speed_goal_actions: List[float]
    steering_actions: List[float]
    discrete_actions: bool
    towns: List[str]
    town_change_frequency: int


def default_config() -> CarlaEnvironmentConfiguration:
    return {
        "continuous_speed_range": (-2.0, 10.0),
        "continuous_steering_range": (-0.3, 0.3),
        "speed_goal_actions": [],
        "steering_actions": [],
        "discrete_actions": True,
        "towns": ["Town01", "Town02", "Town03"],
        "town_change_frequency": 10,
    }


class PIDController(Protocol):
    def __call__(
        self, target_vel: float, current_vel: float
    ) -> Tuple[float, float, bool]:
        return (0.0, 0.0, False)


@dataclass
class SpeedController(PIDController):
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.2

    def __post_init__(self):
        self.last_error = 0.0
        self.integral = 0.0

    def __call__(
        self, target_vel: float, current_vel: float
    ) -> Tuple[float, float, bool]:
        error = target_vel - current_vel
        derivative = error - self.last_error
        self.integral += error
        self.last_error = error

        throttle = self.kp * error + self.ki * self.integral + self.kd * derivative
        brake = 0.0

        if throttle > 1.0:
            throttle = 1.0
        elif throttle < -1.0:
            throttle = -1.0

        if throttle < 0.0:
            brake = -throttle
            throttle = 0.0

        reverse = False
        if current_vel < 0.0 and target_vel < current_vel:
            reverse = True

        return throttle, brake, reverse


@dataclass
class SteeringController:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    def __call__(self, goal_angle: float, current_angle: float) -> float:
        return 0


@dataclass
class CarlaEnvironment(gym.Env):
    config: CarlaEnvironmentConfiguration
    carla_manager: EpisodeManager
    vision_module: Optional[VisionModule]
    reward_function: Callable[[WorldState], Tuple[float, bool]]
    speed_controller: PIDController

    def __post_init__(self):
        """
        Sets up action and observation space based on configurations
        """
        self.time = time()
        self._n_episodes = 0
        self._town = random.choice(self.config["towns"])
        self.amount_of_speed_actions = len(self.config["speed_goal_actions"])
        self.amount_of_steering_actions = len(self.config["steering_actions"])

        if self.config["discrete_actions"]:
            self.action_space = Discrete(
                self.amount_of_steering_actions * self.amount_of_speed_actions
            )

        else:
            self.action_space = Box(
                np.array(
                    [
                        self.config["continuous_speed_range"][0],
                        self.config["continuous_steering_range"][0],
                    ]
                ),
                np.array(
                    [
                        self.config["continuous_speed_range"][1],
                        self.config["continuous_steering_range"][1],
                    ]
                ),
            )

        observation_space_dict = (
            self._set_observation_space_without_vision()
            if self.vision_module is None
            else self._set_observation_space_with_vision()
        )

        self.observation_space = Dict(spaces=observation_space_dict)

        return

    def _set_observation_space_without_vision(self) -> dict:
        observation_space_dict = {}
        lidar = self.carla_manager.config.car_config.lidar

        if self.carla_manager.config.car_config.lidar["enabled"]:
            observation_space_dict["lidar"] = Box(
                low=0,
                high=255,
                shape=lidar["shape"],
                dtype=np.uint8,
            )

        camera_configs = self.carla_manager.config.car_config.cameras

        for index, camera in enumerate(camera_configs):
            observation_space_dict[f"image_{index}"] = Box(
                low=0,
                high=255,
                shape=(camera["height"], camera["width"], 3),
                dtype=np.uint8,
            )

        speed_range = (
            self.config["speed_goal_actions"]
            if self.config["discrete_actions"]
            else self.config["continuous_speed_range"]
        )

        observation_space_dict["state"] = Box(
            low=np.array([min(speed_range)]),
            high=np.array([max(speed_range)]),
            dtype=np.float32,
        )

        return observation_space_dict

    def _set_observation_space_with_vision(self) -> dict:
        if self.vision_module is None:
            raise ValueError("Vision module is not set")

        observation_space_dict = {
            "vision_encoding": Box(
                low=self.vision_module.low,
                high=self.vision_module.high,
                shape=self.vision_module.output_shape,
                dtype=np.float32,
            ),
        }

        observation_space_dict["state"] = Box(
            min(
                np.array(
                    [
                        self.config["continuous_speed_range"]
                        if self.config["discrete_actions"]
                        else self.config["speed_goal_actions"]
                    ]
                )
            ),
            max(
                np.array(
                    [
                        self.config["continuous_speed_range"]
                        if self.config["discrete_actions"]
                        else self.config["speed_goal_actions"]
                    ]
                )
            ),
            dtype=np.float32,
        )

        return observation_space_dict

    def reset(self):
        # select random town from configurations
        self._n_episodes += 1
        if self._n_episodes % self.config["town_change_frequency"] == 0:
            self._town = random.choice(self.config["towns"])

        self.carla_manager.stop_episode()
        self.state = self.carla_manager.start_episode(town=self._town)
        print("RESET EPISODE IN TOWN: ", self._town)

        return self._get_obs()

    def _get_obs(self):
        return (
            self._get_obs_without_vision()
            if self.vision_module is None
            else self._get_obs_with_vision()
        )

    def _get_obs_without_vision(self):
        observation = {}

        for index, image in enumerate(self.state.ego_vehicle_state.sensor_data.images):
            observation[f"image_{index}"] = image[:, :, :3]

        if self.state.ego_vehicle_state.sensor_data.lidar_data:
            observation[
                "lidar"
            ] = self.state.ego_vehicle_state.sensor_data.lidar_data.bev

        observation["state"] = np.array([self.state.ego_vehicle_state.speed])

        return observation

    def _get_obs_with_vision(self):
        if self.vision_module is None:
            raise ValueError("Vision module is not set")

        vision_encoding = self.vision_module(self.state)
        # TODO: get direction of target point, and next high level command, and use as observation state for the RL model

        observation = {
            "vision_encoding": vision_encoding,
            "state": np.array([self.state.ego_vehicle_state.speed]),
        }

        return observation

    def step(self, action):
        goal_speed = 0.0
        steering = 0.0

        if self.config["discrete_actions"]:
            goal_speed = self.config["speed_goal_actions"][
                action // self.amount_of_steering_actions
            ]

            steering = self.config["steering_actions"][
                action % self.amount_of_steering_actions
            ]

        else:
            if not isinstance(action, tuple):
                raise ValueError("Action must be a tuple")

            goal_speed, steering = action

        throttle, brake, reverse = self.speed_controller(
            goal_speed, self.state.ego_vehicle_state.speed
        )

        new_action = Action(throttle, brake, reverse, steering)

        if self.vision_module is not None:
            new_action = self.vision_module.postprocess_action(new_action)

        # update state with result of using the new action
        self.state = self.carla_manager.step(new_action)

        reward, done = self.reward_function(self.state)

        self.time = time()
        return (self._get_obs(), reward, done, {})

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
