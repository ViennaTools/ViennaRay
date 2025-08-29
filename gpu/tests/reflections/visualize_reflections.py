#!/usr/bin/env python3
"""
Comprehensive visualization script for ViennaRay reflection test data.

This script reads reflection direction data from text files and creates various
visualizations including 3D vector plots, angular distributions, and statistical
analyses.

Usage:
    python visualize_reflections.py [--data-dir <path>] [--output-dir <path>]

Data files expected:
    - diffuse_reflection.txt
    - coned_specular_reflection.txt
    - coned_specular_reflection_cpu.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import sys
from pathlib import Path

# Set style for better plots
plt.style.use("seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default")


class ReflectionVisualizer:
    """Class for visualizing reflection data from ViennaRay simulations."""

    def __init__(self, data_dir=".", output_dir="plots"):
        """
        Initialize the visualizer.

        Args:
            data_dir: Directory containing the reflection data files
            output_dir: Directory to save the generated plots
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data containers
        self.data = {}
        self.labels = {
            "diffuse_reflection.txt": "Diffuse Reflection",
            "coned_specular_reflection.txt": "Coned Specular (GPU)",
            "coned_specular_reflection_cpu.txt": "Coned Specular (CPU)",
        }

    def load_data(self):
        """Load reflection data from text files."""
        print("Loading reflection data...")

        for filename, label in self.labels.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    data = np.loadtxt(filepath)
                    if data.shape[1] == 3:  # Ensure we have 3D vectors
                        self.data[filename] = data
                        print(f"✓ Loaded {filename}: {data.shape[0]} vectors")
                    else:
                        print(f"✗ Invalid data shape in {filename}: {data.shape}")
                except Exception as e:
                    print(f"✗ Error loading {filename}: {e}")
            else:
                print(f"⚠ File not found: {filename}")

        if not self.data:
            print("No valid data files found!")
            sys.exit(1)

    def plot_3d_vectors(self, max_vectors=200, save=True):
        """
        Create 3D vector plots showing reflection directions.

        Args:
            max_vectors: Maximum number of vectors to plot per type
            save: Whether to save the plots
        """
        print("Creating 3D vector plots...")

        n_plots = len(self.data)
        fig = plt.figure(figsize=(5 * n_plots, 5))

        for i, (filename, vectors) in enumerate(self.data.items()):
            ax = fig.add_subplot(1, n_plots, i + 1, projection="3d")

            # Sample vectors if too many
            if len(vectors) > max_vectors:
                indices = np.random.choice(len(vectors), max_vectors, replace=False)
                plot_vectors = vectors[indices]
            else:
                plot_vectors = vectors

            # Plot vectors as arrows from origin
            colors = plt.cm.viridis(np.linspace(0, 1, len(plot_vectors)))

            for vec, color in zip(plot_vectors, colors):
                ax.quiver(
                    0,
                    0,
                    0,
                    vec[0],
                    vec[1],
                    vec[2],
                    color=color,
                    alpha=0.6,
                    arrow_length_ratio=0.1,
                )

            # Customize plot
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([0, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"{self.labels[filename]}\n({len(plot_vectors)} vectors)")

            # Add surface normal
            ax.quiver(
                0,
                0,
                0,
                0,
                0,
                1,
                color="red",
                alpha=1.0,
                linewidth=3,
                arrow_length_ratio=0.15,
                label="Surface Normal",
            )

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "3d_reflection_vectors.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"✓ Saved 3D vector plot: {self.output_dir / '3d_reflection_vectors.png'}"
            )

        plt.show()

    def plot_angular_distributions(self, save=True):
        """
        Plot angular distributions (theta and phi) of reflection directions.

        Args:
            save: Whether to save the plots
        """
        print("Creating angular distribution plots...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Colors for different datasets
        colors = ["blue", "red", "green", "orange", "purple"]

        # Plot theta distributions
        ax_theta = axes[0]
        ax_phi = axes[1]
        ax_combined = axes[2]
        ax_polar = axes[3]

        for i, (filename, vectors) in enumerate(self.data.items()):
            label = self.labels[filename]
            color = colors[i % len(colors)]

            # Calculate spherical coordinates
            theta = np.arccos(np.clip(vectors[:, 2], -1, 1))  # Polar angle (0 to π)
            phi = np.arctan2(vectors[:, 1], vectors[:, 0])  # Azimuthal angle (-π to π)

            # Theta distribution
            ax_theta.hist(
                theta,
                bins=50,
                histtype="step",
                density=True,
                color=color,
                label=label,
                linewidth=2,
            )

            # Phi distribution
            ax_phi.hist(
                phi,
                bins=50,
                histtype="step",
                density=True,
                color=color,
                label=label,
                linewidth=2,
            )

            # Combined plot
            ax_combined.hist(
                theta,
                bins=50,
                histtype="step",
                density=True,
                color=color,
                alpha=0.7,
                label=f"{label} (θ)",
                linestyle="-",
            )
            ax_combined.hist(
                phi,
                bins=50,
                histtype="step",
                density=True,
                color=color,
                alpha=0.7,
                label=f"{label} (φ)",
                linestyle="--",
            )

        # Theoretical expectation for diffuse reflection (Lambert's cosine law)
        if "diffuse_reflection.txt" in self.data:
            theta_theory = np.linspace(0, np.pi / 2, 100)
            # For diffuse reflection: p(θ) ∝ sin(2θ)
            ax_theta.plot(
                theta_theory,
                np.sin(2 * theta_theory),
                "k--",
                linewidth=2,
                label="Lambert Law (sin(2θ))",
            )

        # Customize theta plot
        ax_theta.set_xlabel("Polar Angle θ (radians)")
        ax_theta.set_ylabel("Probability Density")
        ax_theta.set_title("Polar Angle Distribution")
        ax_theta.legend()
        ax_theta.grid(True, alpha=0.3)

        # Customize phi plot
        ax_phi.set_xlabel("Azimuthal Angle φ (radians)")
        ax_phi.set_ylabel("Probability Density")
        ax_phi.set_title("Azimuthal Angle Distribution")
        ax_phi.legend()
        ax_phi.grid(True, alpha=0.3)

        # Customize combined plot
        ax_combined.set_xlabel("Angle (radians)")
        ax_combined.set_ylabel("Probability Density")
        ax_combined.set_title("Combined Angular Distributions")
        ax_combined.legend()
        ax_combined.grid(True, alpha=0.3)

        # Polar histogram
        if self.data:
            # Use first dataset for polar plot
            first_data = list(self.data.values())[0]
            theta = np.arccos(np.clip(first_data[:, 2], -1, 1))
            phi = np.arctan2(first_data[:, 1], first_data[:, 0])

            ax_polar.remove()
            ax_polar = fig.add_subplot(2, 2, 4, projection="polar")
            ax_polar.hist(phi, bins=36, alpha=0.7, density=True)
            ax_polar.set_title(f"Polar Distribution\n{list(self.labels.values())[0]}")

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "angular_distributions.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"✓ Saved angular distributions: {self.output_dir / 'angular_distributions.png'}"
            )

        plt.show()

    def plot_hemisphere_projection(self, save=True):
        """
        Create hemisphere projections of reflection directions.

        Args:
            save: Whether to save the plots
        """
        print("Creating hemisphere projections...")

        n_plots = len(self.data)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for i, (filename, vectors) in enumerate(self.data.items()):
            ax = axes[i]

            # Convert to spherical coordinates
            theta = np.arccos(np.clip(vectors[:, 2], 0, 1))  # Only upper hemisphere
            phi = np.arctan2(vectors[:, 1], vectors[:, 0])

            # Stereographic projection
            r = np.tan(theta / 2)
            x_proj = r * np.cos(phi)
            y_proj = r * np.sin(phi)

            # Create density plot
            h = ax.hist2d(x_proj, y_proj, bins=50, cmap="viridis", density=True)

            # Draw unit circle
            circle = plt.Circle((0, 0), 1, fill=False, color="white", linewidth=2)
            ax.add_patch(circle)

            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_aspect("equal")
            ax.set_title(f"{self.labels[filename]}\nStereographic Projection")
            ax.set_xlabel("X projection")
            ax.set_ylabel("Y projection")

            # Add colorbar
            plt.colorbar(h[3], ax=ax, label="Density")

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "hemisphere_projections.png",
                dpi=300,
                bbox_inches="tight",
            )
            print(
                f"✓ Saved hemisphere projections: {self.output_dir / 'hemisphere_projections.png'}"
            )

        plt.show()

    def analyze_statistics(self, save=True):
        """
        Compute and display statistical analysis of the reflection data.

        Args:
            save: Whether to save the statistics
        """
        print("Computing statistics...")

        stats_text = []
        stats_text.append("REFLECTION DATA STATISTICAL ANALYSIS")
        stats_text.append("=" * 50)

        for filename, vectors in self.data.items():
            label = self.labels[filename]
            stats_text.append(f"\n{label}:")
            stats_text.append("-" * len(label))

            # Basic statistics
            stats_text.append(f"Number of vectors: {len(vectors)}")
            stats_text.append(
                f"Mean direction: [{vectors.mean(axis=0)[0]:.4f}, {vectors.mean(axis=0)[1]:.4f}, {vectors.mean(axis=0)[2]:.4f}]"
            )

            # Angular statistics
            theta = np.arccos(np.clip(vectors[:, 2], -1, 1))
            phi = np.arctan2(vectors[:, 1], vectors[:, 0])

            stats_text.append(
                f"Mean polar angle (θ): {np.mean(theta):.4f} rad ({np.degrees(np.mean(theta)):.2f}°)"
            )
            stats_text.append(
                f"Std polar angle (θ): {np.std(theta):.4f} rad ({np.degrees(np.std(theta)):.2f}°)"
            )
            stats_text.append(
                f"Mean azimuthal angle (φ): {np.mean(phi):.4f} rad ({np.degrees(np.mean(phi)):.2f}°)"
            )
            stats_text.append(
                f"Std azimuthal angle (φ): {np.std(phi):.4f} rad ({np.degrees(np.std(phi)):.2f}°)"
            )

            # Vector magnitude analysis
            magnitudes = np.linalg.norm(vectors, axis=1)
            stats_text.append(f"Mean magnitude: {np.mean(magnitudes):.6f}")
            stats_text.append(f"Std magnitude: {np.std(magnitudes):.6f}")
            stats_text.append(f"Min magnitude: {np.min(magnitudes):.6f}")
            stats_text.append(f"Max magnitude: {np.max(magnitudes):.6f}")

            # Z-component analysis (important for reflection)
            stats_text.append(f"Mean Z-component: {np.mean(vectors[:, 2]):.4f}")
            stats_text.append(f"Min Z-component: {np.min(vectors[:, 2]):.4f}")

        # Print to console
        for line in stats_text:
            print(line)

        # Save to file
        if save:
            with open(self.output_dir / "reflection_statistics.txt", "w") as f:
                f.write("\n".join(stats_text))
            print(
                f"\n✓ Saved statistics: {self.output_dir / 'reflection_statistics.txt'}"
            )

    def compare_gpu_cpu(self, save=True):
        """
        Compare GPU and CPU implementations if both are available.

        Args:
            save: Whether to save the comparison plots
        """
        gpu_file = "coned_specular_reflection.txt"
        cpu_file = "coned_specular_reflection_cpu.txt"

        if gpu_file not in self.data or cpu_file not in self.data:
            print("⚠ Both GPU and CPU data needed for comparison")
            return

        print("Creating GPU vs CPU comparison...")

        gpu_data = self.data[gpu_file]
        cpu_data = self.data[cpu_file]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Calculate angular data
        theta_gpu = np.arccos(np.clip(gpu_data[:, 2], -1, 1))
        phi_gpu = np.arctan2(gpu_data[:, 1], gpu_data[:, 0])
        theta_cpu = np.arccos(np.clip(cpu_data[:, 2], -1, 1))
        phi_cpu = np.arctan2(cpu_data[:, 1], cpu_data[:, 0])

        # Theta comparison
        axes[0, 0].hist(
            theta_gpu, bins=50, alpha=0.7, label="GPU", density=True, color="blue"
        )
        axes[0, 0].hist(
            theta_cpu, bins=50, alpha=0.7, label="CPU", density=True, color="red"
        )
        axes[0, 0].set_xlabel("Polar Angle θ (radians)")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Polar Angle Distribution Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Phi comparison
        axes[0, 1].hist(
            phi_gpu, bins=50, alpha=0.7, label="GPU", density=True, color="blue"
        )
        axes[0, 1].hist(
            phi_cpu, bins=50, alpha=0.7, label="CPU", density=True, color="red"
        )
        axes[0, 1].set_xlabel("Azimuthal Angle φ (radians)")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Azimuthal Angle Distribution Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Difference in distributions
        bins = 50
        hist_gpu_theta, bin_edges = np.histogram(theta_gpu, bins=bins, density=True)
        hist_cpu_theta, _ = np.histogram(theta_cpu, bins=bin_edges, density=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        axes[1, 0].plot(bin_centers, hist_gpu_theta - hist_cpu_theta, "g-", linewidth=2)
        axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Polar Angle θ (radians)")
        axes[1, 0].set_ylabel("Density Difference (GPU - CPU)")
        axes[1, 0].set_title("Theta Distribution Difference")
        axes[1, 0].grid(True, alpha=0.3)

        # Statistical comparison
        stats_text = []
        stats_text.append("GPU vs CPU Statistical Comparison:")
        stats_text.append(
            f"Theta - GPU mean: {np.mean(theta_gpu):.4f}, CPU mean: {np.mean(theta_cpu):.4f}"
        )
        stats_text.append(
            f"Theta - Difference: {np.mean(theta_gpu) - np.mean(theta_cpu):.6f}"
        )
        stats_text.append(
            f"Phi - GPU mean: {np.mean(phi_gpu):.4f}, CPU mean: {np.mean(phi_cpu):.4f}"
        )
        stats_text.append(
            f"Phi - Difference: {np.mean(phi_gpu) - np.mean(phi_cpu):.6f}"
        )

        # Kolmogorov-Smirnov test
        try:
            from scipy import stats

            ks_theta = stats.ks_2samp(theta_gpu, theta_cpu)
            ks_phi = stats.ks_2samp(phi_gpu, phi_cpu)
            stats_text.append(
                f"KS test theta - statistic: {ks_theta.statistic:.6f}, p-value: {ks_theta.pvalue:.6f}"
            )
            stats_text.append(
                f"KS test phi - statistic: {ks_phi.statistic:.6f}, p-value: {ks_phi.pvalue:.6f}"
            )
        except ImportError:
            stats_text.append("Scipy not available for KS test")

        # Display stats in plot
        axes[1, 1].text(
            0.05,
            0.95,
            "\n".join(stats_text),
            transform=axes[1, 1].transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
        )
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Statistical Comparison")

        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "gpu_cpu_comparison.png", dpi=300, bbox_inches="tight"
            )
            print(
                f"✓ Saved GPU vs CPU comparison: {self.output_dir / 'gpu_cpu_comparison.png'}"
            )

        plt.show()

    def create_summary_plot(self, save=True):
        """
        Create a comprehensive summary plot with all visualizations.

        Args:
            save: Whether to save the summary plot
        """
        print("Creating summary plot...")

        if not self.data:
            print("No data available for summary plot")
            return

        # Create a large figure with subplots
        fig = plt.figure(figsize=(20, 12))

        # 3D vector plot
        ax1 = fig.add_subplot(2, 4, 1, projection="3d")
        first_data = list(self.data.values())[0]
        plot_vectors = first_data[:100] if len(first_data) > 100 else first_data

        for vec in plot_vectors:
            ax1.quiver(0, 0, 0, vec[0], vec[1], vec[2], alpha=0.6, color="blue")
        ax1.quiver(0, 0, 0, 0, 0, 1, color="red", linewidth=3, label="Normal")
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([0, 1])
        ax1.set_title("3D Reflection Vectors")

        # Angular distributions
        ax2 = fig.add_subplot(2, 4, 2)
        ax3 = fig.add_subplot(2, 4, 3)

        colors = ["blue", "red", "green"]
        for i, (filename, vectors) in enumerate(self.data.items()):
            label = self.labels[filename]
            color = colors[i % len(colors)]

            theta = np.arccos(np.clip(vectors[:, 2], -1, 1))
            phi = np.arctan2(vectors[:, 1], vectors[:, 0])

            ax2.hist(
                theta,
                bins=30,
                histtype="step",
                density=True,
                color=color,
                label=label,
                linewidth=2,
            )
            ax3.hist(
                phi,
                bins=30,
                histtype="step",
                density=True,
                color=color,
                label=label,
                linewidth=2,
            )

        ax2.set_title("Polar Angle Distribution")
        ax2.legend()
        ax3.set_title("Azimuthal Angle Distribution")
        ax3.legend()

        # Hemisphere projection
        ax4 = fig.add_subplot(2, 4, 4)
        vectors = first_data
        theta = np.arccos(np.clip(vectors[:, 2], 0, 1))
        phi = np.arctan2(vectors[:, 1], vectors[:, 0])
        r = np.tan(theta / 2)
        x_proj = r * np.cos(phi)
        y_proj = r * np.sin(phi)

        h = ax4.hist2d(x_proj, y_proj, bins=30, cmap="viridis")
        circle = plt.Circle((0, 0), 1, fill=False, color="white", linewidth=2)
        ax4.add_patch(circle)
        ax4.set_aspect("equal")
        ax4.set_title("Hemisphere Projection")

        # Statistics summary
        ax5 = fig.add_subplot(2, 4, (5, 8))
        stats_text = ["REFLECTION STATISTICS SUMMARY", "=" * 35, ""]

        for filename, vectors in self.data.items():
            label = self.labels[filename]
            theta = np.arccos(np.clip(vectors[:, 2], -1, 1))
            stats_text.extend(
                [
                    f"{label}:",
                    f"  Vectors: {len(vectors)}",
                    f"  Mean θ: {np.degrees(np.mean(theta)):.1f}°",
                    f"  Mean Z: {np.mean(vectors[:, 2]):.3f}",
                    "",
                ]
            )

        ax5.text(
            0.05,
            0.95,
            "\n".join(stats_text),
            transform=ax5.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=10,
        )
        ax5.axis("off")

        plt.suptitle("ViennaRay Reflection Analysis Summary", fontsize=16)
        plt.tight_layout()

        if save:
            plt.savefig(
                self.output_dir / "reflection_summary.png", dpi=300, bbox_inches="tight"
            )
            print(f"✓ Saved summary plot: {self.output_dir / 'reflection_summary.png'}")

        plt.show()

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting comprehensive reflection analysis...")
        print("=" * 50)

        self.load_data()
        print()

        self.analyze_statistics()
        print()

        self.plot_3d_vectors()
        self.plot_angular_distributions()
        self.plot_hemisphere_projection()
        self.compare_gpu_cpu()
        self.create_summary_plot()

        print("=" * 50)
        print("Analysis complete! Check the plots directory for saved visualizations.")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize ViennaRay reflection data")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=".",
        help="Directory containing reflection data files (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="reflection_plots",
        help="Directory to save plots (default: reflection_plots)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save plots to files"
    )

    args = parser.parse_args()

    # Create visualizer and run analysis
    visualizer = ReflectionVisualizer(args.data_dir, args.output_dir)

    # Monkey patch save parameter if no-save is specified
    if args.no_save:

        def make_no_save_wrapper(original_method):
            def wrapper(*args, **kwargs):
                kwargs["save"] = False
                return original_method(*args, **kwargs)

            return wrapper

        for method_name in [
            "plot_3d_vectors",
            "plot_angular_distributions",
            "plot_hemisphere_projection",
            "analyze_statistics",
            "compare_gpu_cpu",
            "create_summary_plot",
        ]:
            original_method = getattr(visualizer, method_name)
            setattr(visualizer, method_name, make_no_save_wrapper(original_method))

    visualizer.run_full_analysis()


if __name__ == "__main__":
    main()
