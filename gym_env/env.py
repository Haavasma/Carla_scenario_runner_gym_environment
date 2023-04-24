from dataclasses import dataclass
import random
from time import time
from typing import Callable, List, Optional, Protocol, Set, Tuple, TypedDict
from episode_manager.renderer import WorldStateRenderer, generate_pygame_surface

from collections import deque
import gymnasium as gym
import numpy as np
import pygame
from episode_manager import EpisodeManager
from episode_manager.episode_manager import Action, WorldState
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import seeding
from srunner.tools.route_parser import RoadOption

from gym_env.route_planner import RoutePlanner


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
        "towns": ["Town01", "Town03", "Town04"],
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


class Controller(object):
    def __init__(self, K_P=5.0, K_I=0.5, K_D=1.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class TestSpeedController(PIDController):
    def __init__(
        self,
        default_speed: float = 4.0,
        brake_speed: float = 0.4,
        brake_ratio: float = 1.1,
        clip_delta=0.25,
        clip_throttle=0.75,
    ):
        self._default_speed = default_speed
        self._brake_speed = brake_speed
        self._brake_ratio = brake_ratio
        self._clip_delta = clip_delta
        self._clip_throttle = clip_throttle
        self._controller = Controller()
        return

    def __call__(self, wanted_speed: float, current_speed: float):
        desired_speed = wanted_speed

        brake = (desired_speed < self._brake_speed) or (
            (current_speed / desired_speed) > self._brake_ratio
        )

        brake = (desired_speed < self._brake_speed) or (
            (current_speed / desired_speed) > self._brake_ratio
        )

        delta = np.clip(desired_speed - current_speed, 0.0, self._clip_delta)
        throttle = self._controller.step(delta)
        throttle = np.clip(throttle, 0.0, self._clip_throttle)
        throttle = throttle if not brake else 0.0

        brake_value = 0.0
        if brake:
            brake_value = 1.0

        return (throttle, brake_value, False)


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
        self._renderer = None
        self._n_episodes = 0
        self._steps = 0
        self._town = random.choice(self.config["towns"])
        self.amount_of_speed_actions = len(self.config["speed_goal_actions"])
        self.amount_of_steering_actions = len(self.config["steering_actions"])
        self._route_planner: Optional[RoutePlanner] = None

        print("INITIALIZING ENVIRONMENT")

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

        observation_space_dict["state"] = self._state_observation_space()

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

        observation_space_dict["state"] = self._state_observation_space()

        return observation_space_dict

    def _state_observation_space(self) -> Box:
        return Box(
            low=np.array([-2.0, -10.0, -10.0, 0, 0, 0, 0, 0, 0]),
            high=np.array([10.0, 10.0, 10.0, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        # select random town from configurations
        self._n_episodes += 1
        if self._n_episodes % self.config["town_change_frequency"] == 0:
            self._town = random.choice(self.config["towns"])

        self.carla_manager.stop_episode()
        self.state = self.carla_manager.start_episode(town=self._town)

        self._route_planner = RoutePlanner()
        self._route_planner.set_route(self.state.scenario_state.global_plan, True)

        return self._get_obs(), {}

    def render(self, mode="human") -> Optional[np.ndarray]:
        if mode == "human":
            if self._renderer is None:
                self._renderer = WorldStateRenderer()

            self._renderer.render(self.state)
            return None
        else:
            pygame_surface = generate_pygame_surface(self.state)
            return np.transpose(np.array(pygame_surface), axes=(1, 0, 2))

    def _get_obs(self):
        return (
            self._get_obs_without_vision()
            if self.vision_module is None
            else self._get_obs_with_vision()
        )

    def _get_obs_without_vision(self):
        observation = self._setup_observation_state()

        for index, image in enumerate(self.state.ego_vehicle_state.sensor_data.images):
            observation[f"image_{index}"] = image[:, :, :3]

        if self.carla_manager.config.car_config.lidar["enabled"]:
            observation[
                "lidar"
            ] = self.state.ego_vehicle_state.sensor_data.lidar_data.bev

        return observation

    def _get_obs_with_vision(self):
        if self.vision_module is None:
            raise ValueError("Vision module is not set")

        observation = self._setup_observation_state()

        vision_encoding = self.vision_module(self.state)

        observation["vision_encoding"] = vision_encoding
        return observation

    def _setup_observation_state(self) -> dict:
        observation = {}
        if self._route_planner is None:
            raise ValueError("Route planner is not set")

        _, pos, target_point, next_cmd = self._route_planner.run_step(
            gps=self.state.ego_vehicle_state.gps
        )

        relative_target_waypoint = find_relative_target_waypoint(
            pos, target_point, self.state.ego_vehicle_state.compass
        )

        relative_target_waypoint = np.clip(relative_target_waypoint, -10.0, 10.0)

        command = np.zeros(6)

        if next_cmd == RoadOption.STRAIGHT:
            command[0] = 1.0
        elif next_cmd == RoadOption.LANEFOLLOW:
            command[1] = 1.0
        elif next_cmd == RoadOption.LEFT:
            command[2] = 1.0
        elif next_cmd == RoadOption.RIGHT:
            command[3] = 1.0
        elif next_cmd == RoadOption.CHANGELANELEFT:
            command[4] = 1.0
        elif next_cmd == RoadOption.CHANGELANERIGHT:
            command[5] = 1.0

        state = np.concatenate(
            (
                np.array([self.state.ego_vehicle_state.speed]),
                relative_target_waypoint,
                command,
            ),
            axis=0,
        )

        observation["state"] = state

        return observation

    def _get_target_point(self) -> np.ndarray:
        """"""
        return np.array([0.0])

    def step(self, action):
        goal_speed = 0.0
        steering = 0.0

        self._steps += 1

        if self.config["discrete_actions"]:
            goal_speed = self.config["speed_goal_actions"][
                action // self.amount_of_steering_actions
            ]

            steering = self.config["steering_actions"][
                action % self.amount_of_steering_actions
            ]

        else:
            if not isinstance(action, np.ndarray):
                raise ValueError("Action must be a numpy array")

            goal_speed, steering = action[0], action[1]

        goal_speed = float(goal_speed)
        steering = float(steering)

        throttle, brake, reverse = self.speed_controller(
            goal_speed, self.state.ego_vehicle_state.speed
        )

        new_action = Action(throttle, brake, reverse, steering)

        print("NEW ACTION: ", new_action)

        if self.vision_module is not None:
            new_action = self.vision_module.postprocess_action(new_action)

        # update state with result of using the new action
        self.state = self.carla_manager.step(new_action)

        reward, done = self.reward_function(self.state)

        result = (self._get_obs(), reward, done, False, {})
        self.time = time()

        return result

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
