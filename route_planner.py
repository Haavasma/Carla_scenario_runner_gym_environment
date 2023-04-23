# Taken from LBC
from collections import deque
from typing import Tuple
import numpy as np
from srunner.tools.route_parser import RoadOption
from copy import deepcopy


class RoutePlanner(object):
    def __init__(self, min_distance=7.5, max_distance=50.0):
        self.saved_route = deque()
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.is_last = False
        self._gps_buffer = deque(maxlen=100)

        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos["lat"], pos["lon"]])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(
        self, gps, gps_coord=True
    ) -> Tuple[deque, np.ndarray, np.ndarray, RoadOption]:
        pos = (gps - self.mean) * self.scale if gps_coord else gps - self.mean
        self._gps_buffer.append(pos)

        def get_target_wp_and_cmd(route: deque):
            return route[1] if len(route) > 1 else route[0]

        if len(self.route) <= 2:
            self.is_last = True
            target_point, new_cmd = get_target_wp_and_cmd(self.route)
            return (self.route, pos, target_point, new_cmd)

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(
                self.route[i][0] - self.route[i - 1][0]
            )
            distance = np.linalg.norm(self.route[i][0] - pos)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        next_wp, next_cmd = self.route[1] if len(self.route) > 1 else self.route[0]

        return (self.route, pos, next_wp, next_cmd)

    def save(self):

        self.saved_route = deepcopy(self.route)

    def load(self):
        self.route = self.saved_route
        self.is_last = False


def find_relative_target_waypoint(
    ego_pos: np.ndarray, target_pos: np.ndarray, compass: float
) -> np.ndarray:
    theta = np.radians(compass) + np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    local_command_point = np.array(
        [target_pos[0] - ego_pos[0], target_pos[1] - ego_pos[1]]
    )
    local_command_point = R.T.dot(local_command_point)

    return local_command_point
