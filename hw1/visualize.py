from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from .simulation import System

plt.style.use("./figures.mplstyle")


def plot_potential(
    system: System,
    *,
    grid_size: int = 200,
    xlim: list[float] = [-1.5, 1.5],
    ylim: list[float] = [-1.5, 1.5],
    save_fig: bool = True,
    filename: str = "potential.pdf",
    contourf_level=500,
    contour_level=80,
) -> None:
    N = grid_size
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    X, Y = np.meshgrid(x, y)
    potential = np.squeeze(system.grav_potential_at(np.stack([X, Y]).T).T)
    phi = np.sign(potential) * np.log10(1 + np.abs(potential))

    plt.figure(figsize=(10, 8))
    plt.contourf(
        x,
        y,
        phi,
        levels=contourf_level,
        cmap="magma",
    )
    plt.colorbar()

    plt.contour(
        x,
        y,
        phi,
        levels=contour_level,
        linewidths=0.3,
        colors="k",
        alpha=0.5,
    )

    plt.title(r"Gravitational Potential")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename, dpi=300)


def plot_gravity(
    system: System,
    lagrange_points: NDArray[np.floating[Any]],
    *,
    grid_size: int = 200,
    contour_level=80,
    xlim: list[float] = [-1.5, 1.5],
    ylim: list[float] = [-1.5, 1.5],
    save_fig: bool = True,
    filename: str = "vectorplot.pdf",
    **kwargs,
) -> None:
    N = grid_size
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    X, Y = np.meshgrid(x, y)
    potential = np.squeeze(system.grav_potential_at(np.stack([X, Y]).T).T)
    phi = np.sign(potential) * np.log10(1 + np.abs(potential))

    x_subset = x[::4]
    y_subset = y[::4]
    X_subset, Y_subset = np.meshgrid(x_subset, y_subset)

    g = np.squeeze(system.grav_acceleration_at(np.stack([X_subset, Y_subset]).T).T)
    normed_g = np.linalg.norm(g, axis=0)
    g_scaled = g / normed_g

    plt.figure(figsize=(10, 8))
    plt.contour(
        x,
        y,
        phi,
        levels=contour_level,
        linewidths=0.5,
        colors="k",
        alpha=0.8,
    )
    plt.quiver(
        X_subset,
        Y_subset,
        g_scaled[0],
        g_scaled[1],
        np.log10(1 + normed_g),
        width=0.0017,
        scale=1 / 0.017,
        # cmap="YlGnBu",
        cmap="inferno",
    )
    plt.colorbar()

    for idx, point in enumerate(lagrange_points):
        plt.scatter(*point, marker="o", s=160, label=f"L{idx + 1}", zorder=10)

    plt.title("Gravitational Acceleration and Lagrange Points")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi=300)


def plot_orbits(
    system: System,
    radius: list[float],
    labels: list[str],
    *,
    grid_size: int = 200,
    contour_level=80,
    xlim: list[float] = [-1.5, 1.5],
    ylim: list[float] = [-1.5, 1.5],
    save_fig: bool = True,
    filename: str = "orbits.pdf",
    **kwargs,
):
    plt.figure(figsize=(10, 8))

    plot_potential(
        system,
        grid_size=grid_size,
        xlim=xlim,
        ylim=ylim,
        contour_level=contour_level,
        save_fig=False,
    )

    theta = np.linspace(0, 2 * np.pi, 1000)
    for r, l in zip(radius, labels):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        plt.plot(x, y, label=l, **kwargs)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.legend()

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()

    if save_fig:
        plt.savefig(filename, dpi=300)
