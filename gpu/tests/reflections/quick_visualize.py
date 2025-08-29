#!/usr/bin/env python3
"""
Quick visualization script for ViennaRay reflection test data.

This is a simplified version that focuses on the specific output from the
GPU reflection test (testReflections.cpp).

Usage:
    python quick_visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def load_reflection_data():
    """Load reflection data from the current directory."""
    data = {}

    # Expected files from the testReflections program
    files = {
        "diffuse_reflection.txt": "Diffuse Reflection",
        "coned_specular_reflection.txt": "Coned Specular (GPU)",
        "coned_specular_reflection_cpu.txt": "Coned Specular (CPU)",
    }

    for filename, label in files.items():
        if os.path.exists(filename):
            try:
                vectors = np.loadtxt(filename)
                data[filename] = {"vectors": vectors, "label": label}
                print(f"✓ Loaded {filename}: {len(vectors)} vectors")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"⚠ File not found: {filename}")

    return data


def plot_3d_arrows(data, max_vectors=150):
    """Plot 3D arrows showing reflection directions."""
    n_plots = len(data)
    if n_plots == 0:
        print("No data to plot!")
        return

    fig = plt.figure(figsize=(6 * n_plots, 6))

    for i, (filename, info) in enumerate(data.items()):
        ax = fig.add_subplot(1, n_plots, i + 1, projection="3d")
        vectors = info["vectors"]

        # Sample vectors if too many
        if len(vectors) > max_vectors:
            indices = np.random.choice(len(vectors), max_vectors, replace=False)
            plot_vectors = vectors[indices]
        else:
            plot_vectors = vectors

        # Plot reflection vectors
        for vec in plot_vectors:
            ax.quiver(
                0,
                0,
                0,
                vec[0],
                vec[1],
                vec[2],
                color="blue",
                alpha=0.5,
                arrow_length_ratio=0.1,
            )

        # Add surface normal (pointing up)
        ax.quiver(
            0,
            0,
            0,
            0,
            0,
            1,
            color="red",
            alpha=1.0,
            linewidth=4,
            arrow_length_ratio=0.15,
            label="Surface Normal",
        )

        # Add incident ray (for coned specular)
        if "coned" in filename:
            # From testReflections.cpp: incident angle is 90 degrees
            # inDir = {0.0, -sin(90°), -cos(90°)}
            inAngle = np.deg2rad(80)
            incident = np.array([0.0, -np.sin(inAngle), -np.cos(inAngle)])
            incident = -incident / np.linalg.norm(incident)  # Normalize
            ax.quiver(
                0,
                0,
                0,
                incident[0],
                incident[1],
                incident[2],
                color="green",
                alpha=1.0,
                linewidth=3,
                arrow_length_ratio=0.15,
                label="Incident Ray",
            )

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f'{info["label"]}\n({len(plot_vectors)} vectors shown)')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_angular_distributions(data):
    """Plot angular distributions (theta and phi)."""
    if not data:
        print("No data to plot!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["blue", "red", "green", "orange"]

    for i, (filename, info) in enumerate(data.items()):
        vectors = info["vectors"]
        label = info["label"]
        color = colors[i % len(colors)]

        # Calculate spherical coordinates
        # Ensure z-component is in valid range for arccos
        z_clipped = np.clip(vectors[:, 2], -1, 1)
        theta = np.arccos(z_clipped)  # Polar angle (0 to π)
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])  # Azimuthal angle (-π to π)

        # Plot distributions
        ax1.hist(
            theta,
            bins=40,
            histtype="step",
            density=True,
            color=color,
            label=label,
            linewidth=2,
            alpha=0.8,
        )
        ax2.hist(
            phi,
            bins=40,
            histtype="step",
            density=True,
            color=color,
            label=label,
            linewidth=2,
            alpha=0.8,
        )

    # Add theoretical expectation for diffuse reflection
    if any("diffuse" in filename for filename in data.keys()):
        theta_theory = np.linspace(0, np.pi / 2, 100)
        # Lambert's cosine law: p(θ) ∝ sin(2θ)
        ax1.plot(
            theta_theory,
            np.sin(2 * theta_theory),
            "k--",
            linewidth=2,
            alpha=0.7,
            label="Lambert Law (sin(2θ))",
        )

    ax1.set_xlabel("Polar Angle θ (radians)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Polar Angle Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Azimuthal Angle φ (radians)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Azimuthal Angle Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_statistics(data):
    """Print basic statistics for each dataset."""
    print("\n" + "=" * 60)
    print("REFLECTION STATISTICS")
    print("=" * 60)

    for filename, info in data.items():
        vectors = info["vectors"]
        label = info["label"]

        print(f"\n{label}:")
        print("-" * len(label))
        print(f"  Number of vectors: {len(vectors)}")
        print(
            f"  Mean direction: [{vectors.mean(axis=0)[0]:.4f}, {vectors.mean(axis=0)[1]:.4f}, {vectors.mean(axis=0)[2]:.4f}]"
        )

        # Check if vectors are normalized
        magnitudes = np.linalg.norm(vectors, axis=1)
        print(
            f"  Vector magnitudes - mean: {np.mean(magnitudes):.6f}, std: {np.std(magnitudes):.6f}"
        )

        # Angular statistics
        theta = np.arccos(np.clip(vectors[:, 2], -1, 1))
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])

        print(
            f"  Mean polar angle: {np.degrees(np.mean(theta)):.2f}° (std: {np.degrees(np.std(theta)):.2f}°)"
        )
        print(
            f"  Mean azimuthal angle: {np.degrees(np.mean(phi)):.2f}° (std: {np.degrees(np.std(phi)):.2f}°)"
        )
        print(
            f"  Z-component range: [{np.min(vectors[:, 2]):.4f}, {np.max(vectors[:, 2]):.4f}]"
        )


def compare_implementations(data):
    """Compare GPU and CPU implementations if both are available."""
    gpu_key = next(
        (k for k in data.keys() if "coned_specular_reflection.txt" in k), None
    )
    cpu_key = next(
        (k for k in data.keys() if "coned_specular_reflection_cpu.txt" in k), None
    )

    if not (gpu_key and cpu_key):
        print("\nCannot compare implementations - need both GPU and CPU data")
        return

    gpu_vectors = data[gpu_key]["vectors"]
    cpu_vectors = data[cpu_key]["vectors"]

    print(f"\n" + "=" * 60)
    print("GPU vs CPU IMPLEMENTATION COMPARISON")
    print("=" * 60)

    # Basic comparison
    print(f"GPU vectors: {len(gpu_vectors)}")
    print(f"CPU vectors: {len(cpu_vectors)}")

    # Angular comparison
    theta_gpu = np.arccos(np.clip(gpu_vectors[:, 2], -1, 1))
    theta_cpu = np.arccos(np.clip(cpu_vectors[:, 2], -1, 1))
    phi_gpu = np.arctan2(gpu_vectors[:, 1], gpu_vectors[:, 0])
    phi_cpu = np.arctan2(cpu_vectors[:, 1], cpu_vectors[:, 0])

    print(f"\nAngular Statistics:")
    print(
        f"  Theta (GPU): mean={np.degrees(np.mean(theta_gpu)):.3f}°, std={np.degrees(np.std(theta_gpu)):.3f}°"
    )
    print(
        f"  Theta (CPU): mean={np.degrees(np.mean(theta_cpu)):.3f}°, std={np.degrees(np.std(theta_cpu)):.3f}°"
    )
    print(
        f"  Theta difference: {np.degrees(np.mean(theta_gpu) - np.mean(theta_cpu)):.6f}°"
    )

    print(
        f"  Phi (GPU): mean={np.degrees(np.mean(phi_gpu)):.3f}°, std={np.degrees(np.std(phi_gpu)):.3f}°"
    )
    print(
        f"  Phi (CPU): mean={np.degrees(np.mean(phi_cpu)):.3f}°, std={np.degrees(np.std(phi_cpu)):.3f}°"
    )
    print(f"  Phi difference: {np.degrees(np.mean(phi_gpu) - np.mean(phi_cpu)):.6f}°")

    # Vector component comparison
    gpu_mean = gpu_vectors.mean(axis=0)
    cpu_mean = cpu_vectors.mean(axis=0)
    print(f"\nVector Component Comparison:")
    print(f"  GPU mean: [{gpu_mean[0]:.6f}, {gpu_mean[1]:.6f}, {gpu_mean[2]:.6f}]")
    print(f"  CPU mean: [{cpu_mean[0]:.6f}, {cpu_mean[1]:.6f}, {cpu_mean[2]:.6f}]")
    print(
        f"  Difference: [{gpu_mean[0]-cpu_mean[0]:.6f}, {gpu_mean[1]-cpu_mean[1]:.6f}, {gpu_mean[2]-cpu_mean[2]:.6f}]"
    )


def main():
    """Main function."""
    print("ViennaRay Reflection Visualization")
    print("==================================")

    # Load data
    data = load_reflection_data()

    if not data:
        print("No reflection data files found!")
        print(
            "Expected files: diffuse_reflection.txt, coned_specular_reflection.txt, coned_specular_reflection_cpu.txt"
        )
        return

    # Print statistics
    print_statistics(data)

    # Compare implementations
    compare_implementations(data)

    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_3d_arrows(data)
    plot_angular_distributions(data)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
