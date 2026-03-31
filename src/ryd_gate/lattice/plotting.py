"""Visualization for N-atom 3-level lattice simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_spatial_rydberg(
    coords: np.ndarray,
    rydberg_occ: np.ndarray,
    sublattice: np.ndarray | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Rydberg population as colored circles at atom positions.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Atom positions.
    rydberg_occ : ndarray, shape (N,)
        Per-atom Rydberg population.
    sublattice : ndarray or None
        If given, use squares for +1 sublattice, circles for -1.
    title : str
        Plot title.
    ax : Axes or None
        Existing axes. Creates new figure if None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()

    if sublattice is not None:
        for sub_val, marker in [(1, 's'), (-1, 'o')]:
            mask = sublattice == sub_val
            sc = ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=rydberg_occ[mask], cmap='coolwarm',
                vmin=0, vmax=1, s=300, marker=marker,
                edgecolors='black', linewidths=1.0,
            )
    else:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=rydberg_occ, cmap='coolwarm',
            vmin=0, vmax=1, s=300,
            edgecolors='black', linewidths=1.0,
        )

    fig.colorbar(sc, ax=ax, label=r'$P_r$', shrink=0.8)

    for i, (x, y) in enumerate(coords):
        ax.annotate(f'{rydberg_occ[i]:.2f}', (x, y),
                    ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    if title:
        ax.set_title(title)
    return fig


def plot_population_evolution(
    times: np.ndarray,
    rydberg_occ: np.ndarray,
    sublattice: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Per-atom P_r(t) curves colored by sublattice.

    Parameters
    ----------
    times : ndarray, shape (n_times,)
    rydberg_occ : ndarray, shape (n_times, N)
    sublattice : ndarray, shape (N,)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.get_figure()

    N = rydberg_occ.shape[1]
    t_us = times * 1e6

    for i in range(N):
        color = '#d62728' if sublattice[i] > 0 else '#1f77b4'
        ax.plot(t_us, rydberg_occ[:, i], color=color, alpha=0.6, lw=1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#d62728', label='Sublattice +1'),
        Line2D([0], [0], color='#1f77b4', label='Sublattice -1'),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel(r'Time ($\mu$s)')
    ax.set_ylabel(r'$P_r$')
    ax.set_ylim(-0.05, 1.05)
    return fig
