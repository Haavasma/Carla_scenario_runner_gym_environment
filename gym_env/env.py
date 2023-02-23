from dataclasses import dataclass, field
from typing import Any, Callable, List, Protocol, Tuple
from episode_manager.episode_manager import Action, WorldState
import gym
from gym.spaces import Box, Dict, Discrete
from gym.utils import seeding
from episode_manager import EpisodeManager
import numpy as np


class VisionModule(Protocol):
    """
    Defines protocol (interface) for a vision module that is injected to
    the environment to provide vision encoding and selected action postprocessing
    """

    output_shape: Tuple[Any]
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


@dataclass
class CarlaEnvironmentConfiguration:
    continuous_speed_range: Tuple[float, float] = (-2.0, 10.0)  # m/s
    continuous_steering_range: Tuple[float, float] = (-0.3, 0.3)  # rad i think

    speed_goal_actions: List[float] = field(default_factory=lambda: [])
    steering_actions: List[float] = field(default_factory=lambda: [])
    discrete_actions: bool = True


@dataclass
class PIDController:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    def __call__(
        self, goal_speed: float, current_speed: float
    ) -> Tuple[float, float, bool]:
        """

        Returns Throttle, Barke, and whether the agent should be reversing
        """

        return (0, 0, False)


@dataclass
class CarlaEnvVisionModule(gym.Env):
    config: CarlaEnvironmentConfiguration
    carla_manager: EpisodeManager
    vision_module: VisionModule
    reward_function: Callable[[WorldState], float]
    is_done: Callable[[WorldState], bool]
    pid_controller: PIDController

    def __init__(self):
        """
        Sets up action and observation space based on configurations
        """
        self.amount_of_speed_actions = len(self.config.speed_goal_actions)
        self.amount_of_steering_actions = len(self.config.steering_actions)

        if self.config.discrete_actions:
            self.action_space = Discrete(
                self.amount_of_steering_actions * self.amount_of_speed_actions
            )

        else:
            self.action_space = Box(
                np.array(
                    [
                        self.config.continuous_speed_range[0],
                        self.config.continuous_steering_range[0],
                    ]
                ),
                np.array(
                    [
                        self.config.continuous_speed_range[1],
                        self.config.continuous_steering_range[1],
                    ]
                ),
            )

        observation_space_dict = {
            "vision_encoding": Box(
                low=self.vision_module.low,
                high=self.vision_module.high,
                shape=self.vision_module.output_shape,
                dtype=np.float32,
            ),
            "state": Box(
                low=min(
                    self.config.continuous_speed_range
                    if self.config.discrete_actions
                    else self.config.speed_goal_actions
                ),
                high=max(
                    self.config.continuous_speed_range
                    if self.config.discrete_actions
                    else self.config.speed_goal_actions
                ),
                dtype=np.float32,
            ),
        }
        self.observation_space = Dict(spaces=observation_space_dict)

        return

    def reset(self):
        self.carla_manager.stop_episode()
        self.state = self.carla_manager.start_episode()

        return self._get_obs()

    def _get_obs(self):
        vision_encoding = self.vision_module(self.state)
        # TODO: get direction of target point, and next high level command, and use as observation state for the RL model

        observation = {
            "vision_encoding": vision_encoding,
            "state": self.state.ego_vehicle_state.speed,
        }

        return observation

    def step(self, action):
        goal_speed = 0.0
        steering = 0.0

        if self.config.discrete_actions:
            if not isinstance(action, int):
                raise ValueError("Action must be an integer")

            goal_speed = self.config.speed_goal_actions[
                action // self.amount_of_speed_actions
            ]

            steering = self.config.steering_actions[
                action % self.amount_of_steering_actions
            ]

        else:
            if not isinstance(action, tuple):
                raise ValueError("Action must be a tuple")

            goal_speed, steering = action

        throttle, brake, reverse = self.pid_controller(
            goal_speed, self.state.ego_vehicle_state.speed
        )

        action = Action(throttle, brake, reverse, steering)
        action = self.vision_module.postprocess_action(action)

        # update state with result of using the new action
        self.state = self.carla_manager.step(Action(throttle, brake, reverse, steering))

        reward = self.reward_function(self.state)
        done = self.is_done(self.state)

        return (self._get_obs(), reward, done)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
