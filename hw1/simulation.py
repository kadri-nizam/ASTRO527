import math
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hw1.root_finder import bisection

# Constants
# G = 1
G = 4 * math.pi**2


@dataclass
class Body:
    mass: float
    position: np.ndarray


@dataclass
class System:
    distance: float
    large_body: Body
    small_body: Body

    def __post_init__(self):
        # Shift the system so that the center of mass is at the origin
        self.large_body.position -= self.center_of_mass
        self.small_body.position -= self.center_of_mass

    @cached_property
    def reduced_mass(self) -> float:
        return (
            self.large_body.mass
            * self.small_body.mass
            / (self.large_body.mass + self.small_body.mass)
        )

    @cached_property
    def center_of_mass(self) -> NDArray[np.floating[Any]]:
        return (
            self.large_body.mass * self.large_body.position
            + self.small_body.mass * self.small_body.position
        ) / (self.large_body.mass + self.small_body.mass)

    @cached_property
    def angular_frequency(self) -> float:
        return math.sqrt(
            G * (self.large_body.mass + self.small_body.mass) / self.distance**3
        )

    def grav_potential_at(self, position: ArrayLike) -> NDArray[np.floating[Any]]:
        position = np.array(position)

        return (
            System.grav_potential(self.large_body, position)
            + System.grav_potential(self.small_body, position)
            - 0.5
            * self.angular_frequency**2
            * np.sum(position * position, axis=-1, keepdims=True)
        )

    def grav_acceleration_at(self, position: ArrayLike) -> NDArray[np.floating[Any]]:
        position = np.array(position)

        return (
            System.grav_acceleration(self.large_body, position)
            + System.grav_acceleration(self.small_body, position)
            + self.angular_frequency**2 * position
        )

    @staticmethod
    def grav_potential(body: Body, test_position: ArrayLike) -> float:
        r_position = np.array(test_position) - body.position
        return (
            -G
            * body.mass
            / np.sum(r_position * r_position, axis=-1, keepdims=True) ** 0.5
        )

    @staticmethod
    def grav_acceleration(
        body: Body, test_position: ArrayLike
    ) -> NDArray[np.floating[Any]]:
        r_position = np.array(test_position) - body.position
        return (
            -G
            * body.mass
            * r_position
            / np.sum(r_position * r_position, axis=-1, keepdims=True) ** 1.5
        )


def initialize_system(D: float, M_1: float, M_2: float) -> System:
    # By default, we'll put the large body at the origin and the small body at
    # the origin + D

    return System(
        D,
        Body(M_1, np.array([0, 0], dtype=float)),
        Body(M_2, np.array([D, 0], dtype=float)),
    )


def find_lagrange_points(
    system: System,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    xlim: list[float] = [-1.5, 1.5],
    ylim: list[float] = [-1.5, 1.5],
) -> NDArray[np.floating[Any]]:
    x_mid, _ = (system.large_body.position + system.small_body.position) / 2

    def lagrange_on_axis(x) -> float:
        return system.grav_acceleration_at(np.array([x, 0])).flatten()[0]

    def lagrange_off_axis(y) -> float:
        return system.grav_acceleration_at(np.array([x_mid, y])).flatten()[1]

    SINGLE_PRECISION_EPSILON = np.finfo(np.float32).eps

    # Find the on x-axis roots
    l3 = bisection(
        lagrange_on_axis,
        xlim[0],
        system.large_body.position[0] - SINGLE_PRECISION_EPSILON,
        max_iteration=max_iter,
        tolerance=tol,
    )
    l1 = bisection(
        lagrange_on_axis,
        system.large_body.position[0] + SINGLE_PRECISION_EPSILON,
        system.small_body.position[0] - SINGLE_PRECISION_EPSILON,
        max_iteration=max_iter,
        tolerance=tol,
    )
    l2 = bisection(
        lagrange_on_axis,
        system.small_body.position[0] + SINGLE_PRECISION_EPSILON,
        xlim[1],
        max_iteration=max_iter,
        tolerance=tol,
    )

    # Find the off x-axis roots
    l4 = bisection(
        lagrange_off_axis,
        0,
        ylim[1],
        max_iteration=max_iter,
        tolerance=tol,
    )
    l5 = bisection(
        lagrange_off_axis,
        ylim[0],
        0,
        max_iteration=max_iter,
        tolerance=tol,
    )

    lagrange_points = np.array(
        [
            [l1, 0],
            [l2, 0],
            [l3, 0],
            [x_mid, l4],
            [x_mid, l5],
        ]
    )

    assert not np.isnan(lagrange_points).any(), "Failed to find all Lagrange points"
    return lagrange_points
