"""
Publication-grade plotting functions for simulation analysis.
Provides headless plotting with consistent styling and export to SVG/PNG formats.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Union, Optional

# Set publication-grade defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.transparent'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'


def plot_energy(time: Union[List[float], np.ndarray],
                energy: Union[List[float], np.ndarray],
                filename_prefix: str = 'energy_plot',
                title: Optional[str] = None) -> None:
    """
    Create publication-grade energy vs time plot.

    Parameters:
    -----------
    time : array-like
        Time points (reduced units)
    energy : array-like
        Energy values (reduced units)
    filename_prefix : str
        Prefix for output files (default: 'energy_plot')
    title : str, optional
        Plot title (default: 'Energy vs Time')
    """
    plt.figure(figsize=(8, 6))

    plt.plot(time, energy, 'b-', linewidth=2, alpha=0.8)

    plt.xlabel('Time (reduced units)')
    plt.ylabel('Total Energy (reduced units)')

    if title is None:
        title = 'Energy vs Time'
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save in both formats
    plt.savefig(f'{filename_prefix}.svg', format='svg')
    plt.savefig(f'{filename_prefix}.png', format='png')

    plt.close()


def plot_products(time: Union[List[float], np.ndarray],
                  counts: Union[Dict[str, Union[List[float], np.ndarray]],
                               List[Union[List[float], np.ndarray]]],
                  filename_prefix: str = 'products_plot',
                  title: Optional[str] = None) -> None:
    """
    Create publication-grade product counts vs time plot.

    Parameters:
    -----------
    time : array-like
        Time points (reduced units)
    counts : dict or list
        If dict: {product_name: count_array}
        If list: [count_array1, count_array2, ...] with default names
    filename_prefix : str
        Prefix for output files (default: 'products_plot')
    title : str, optional
        Plot title (default: 'Product Counts vs Time')
    """
    plt.figure(figsize=(8, 6))

    if isinstance(counts, dict):
        for name, count in counts.items():
            plt.plot(time, count, label=name, linewidth=2, alpha=0.8)
    elif isinstance(counts, list):
        for i, count in enumerate(counts):
            plt.plot(time, count, label=f'Product {i+1}', linewidth=2, alpha=0.8)
    else:
        raise ValueError("counts must be dict or list of arrays")

    plt.xlabel('Time (reduced units)')
    plt.ylabel('Product Count')

    if title is None:
        title = 'Product Counts vs Time'
    plt.title(title)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save in both formats
    plt.savefig(f'{filename_prefix}.svg', format='svg')
    plt.savefig(f'{filename_prefix}.png', format='png')

    plt.close()


def plot_energy_drift(time: Union[List[float], np.ndarray],
                      drift: Union[List[float], np.ndarray],
                      filename_prefix: str = 'energy_drift_plot',
                      title: Optional[str] = None) -> None:
    """
    Create publication-grade energy drift vs time plot.

    Parameters:
    -----------
    time : array-like
        Time points (reduced units)
    drift : array-like
        Relative energy drift values
    filename_prefix : str
        Prefix for output files (default: 'energy_drift_plot')
    title : str, optional
        Plot title (default: 'Energy Drift vs Time')
    """
    plt.figure(figsize=(8, 6))

    plt.semilogy(time, np.abs(drift), 'r-', linewidth=2, alpha=0.8)

    plt.xlabel('Time (reduced units)')
    plt.ylabel('Relative Energy Drift')

    if title is None:
        title = 'Energy Drift vs Time'
    plt.title(title)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save in both formats
    plt.savefig(f'{filename_prefix}.svg', format='svg')
    plt.savefig(f'{filename_prefix}.png', format='png')

    plt.close()


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    time = np.linspace(0, 100, 1000)
    energy = 10 + 0.1 * np.sin(0.1 * time) + 0.01 * np.random.randn(1000)

    # Test energy plot
    plot_energy(time, energy)

    # Sample product counts
    products = {
        'H2O': np.maximum(0, 10 - 0.05 * time + np.random.randn(1000)),
        'H2': np.maximum(0, 0.02 * time + np.random.randn(1000)),
        'O2': np.maximum(0, 0.01 * time + 0.5 * np.random.randn(1000))
    }

    # Test products plot
    plot_products(time, products)

    # Sample drift
    drift = 0.001 * np.exp(-0.01 * time) + 1e-6 * np.random.randn(1000)

    # Test drift plot
    plot_energy_drift(time, drift)

    print("Sample plots saved as SVG and PNG files.")