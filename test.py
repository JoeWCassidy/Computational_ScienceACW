import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import chisquare, kstest, chi2
from concurrent.futures import ThreadPoolExecutor

# ---- Part I: Cellular Automata Movement Simulation ----

def task_1_2(total_steps=10000, checkpoints=[1000, 10000], run_id=1):
    """Simulate movement in 8 directions for 1000 and 10000 steps with checkpoints for uniformity analysis."""
    grid_size = 100
    directions_map = {
        0: "Right", 1: "Left", 2: "Down", 3: "Up", 
        4: "Down-Right", 5: "Down-Left", 6: "Up-Right", 7: "Up-Left"
    }
    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1)
    ]
    
    x, y = grid_size // 2, grid_size // 2  # Start at the center
    positions = [(x, y)]
    directions = []
    step_data = {checkpoint: [] for checkpoint in checkpoints}
    
    # Simulate cell movement
    for step in range(1, total_steps + 1):
        direction = np.random.choice(range(8))  # Randomly select one of the 8 directions
        dx, dy = moves[direction]
        x = min(max(0, x + dx), grid_size - 1)
        y = min(max(0, y + dy), grid_size - 1)
        directions.append(directions_map[direction])
        positions.append((x, y))
        
        if step in checkpoints:
            step_data[step] = Counter(directions).copy()

    # Return the simulation data and directions_map for further processing
    return run_id, positions, step_data, directions_map

def run_multiple_simulations():
    """Run three simulations of task_1_2 concurrently to reduce bias."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task_1_2, run_id=i) for i in range(1, 4)]
        results = [future.result() for future in futures]  # Collect results for all runs
    return results  # Return all simulation results

def plot_simulation_results(results):
    """Plot the results for all runs after they are finished."""
    for run_id, positions, step_data, directions_map in results:
        for step, counts in step_data.items():
            # Plot cell movement and direction distribution
            x_positions, y_positions = zip(*positions[:step])

            # Create a figure with a 2x2 grid of subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

            # Plot movement (first subplot)
            axs[0, 0].plot(x_positions, y_positions, marker='o', linestyle='-', markersize=2)
            axs[0, 0].set_title(f"Cell Movement (Run {run_id} - First {step} Steps)")
            axs[0, 0].set_xlabel("X Position")
            axs[0, 0].set_ylabel("Y Position")
            axs[0, 0].grid(True)

            # Plot direction distribution and Chi-Squared Test (Observed vs Expected) in the same graph
            observed = [counts.get(direction, 0) for direction in directions_map.values()]
            expected = [step / 8] * 8  # Uniform distribution expectation for 8 directions
            axs[0, 1].bar(counts.keys(), observed, color='orange', label="Observed")
            axs[0, 1].axhline(y=expected[0], color='r', linestyle='--', label="Expected (Uniform)")
            axs[0, 1].set_title(f"Direction Distribution and Chi-Squared Test (Checkpoint {step})")
            axs[0, 1].set_xlabel("Direction")
            axs[0, 1].set_ylabel("Frequency")
            axs[0, 1].legend()
            axs[0, 1].grid(True)

            # Perform chi-squared test and print the results
            chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
            print(f"\nChi-Squared Test Results (Run {run_id}, Checkpoint {step}):")
            print(f"Observed Frequencies: {observed}")
            print(f"Expected Frequencies: {expected}")
            print(f"Chi-Squared Statistic: {chi2_stat:.4f}, p-value: {p_value:.4f}")
            if p_value > 0.05:
                print("Result: The data is consistent with a uniform distribution (p > 0.05).")
            else:
                print("Result: The data significantly deviates from a uniform distribution (p ≤ 0.05).")

            # Chi-Squared Distribution plot with test statistic (third subplot)
            df = len(observed) - 1  # Degrees of freedom
            x_vals = np.linspace(0, 15, 1000)
            y_vals = chi2.pdf(x_vals, df)
            axs[1, 0].plot(x_vals, y_vals, label=f'Chi-Squared Distribution (df={df})')
            axs[1, 0].fill_between(x_vals, y_vals, where=(x_vals >= chi2_stat), color='r', alpha=0.5, label='Critical Region')
            axs[1, 0].axvline(chi2_stat, color='k', linestyle='--', label=f'Test Statistic = {chi2_stat:.4f}')
            axs[1, 0].legend()
            axs[1, 0].set_title(f"Chi-Squared Distribution with Test Statistic (Checkpoint {step})")
            axs[1, 0].set_xlabel('Chi-Squared Value')
            axs[1, 0].set_ylabel('Density')
            axs[1, 0].grid(True)

            # Perform Kolmogorov-Smirnov test and print the results
            observed_probabilities = [counts.get(direction, 0) / step for direction in directions_map.values()]
            ks_statistic, ks_p_value = kstest(observed_probabilities, 'uniform')
            print(f"\nKolmogorov-Smirnov Test Results (Run {run_id}, Checkpoint {step}):")
            print(f"KS Statistic: {ks_statistic:.4f}, p-value: {ks_p_value:.4f}")
            if ks_p_value > 0.05:
                print("Result: The data is consistent with a uniform distribution (p > 0.05) according to the KS test.")
            else:
                print("Result: The data significantly deviates from a uniform distribution (p ≤ 0.05) according to the KS test.")

            # KS Test (CDF plot) and KS Statistic (fourth subplot)
            uniform_cdf = np.cumsum([1/8] * 8)  # Uniform distribution CDF for 8 directions
            observed_cdf = np.cumsum(observed_probabilities)

            axs[1, 1].plot(list(directions_map.values()), observed_cdf, label="Observed CDF", color='b', marker='o')
            axs[1, 1].plot(list(directions_map.values()), uniform_cdf, label="Expected CDF (Uniform)", color='r', linestyle='--')
            axs[1, 1].set_title(f"KS Test: CDF of Observed vs Expected (Checkpoint {step})")
            axs[1, 1].set_xlabel("Direction")
            axs[1, 1].set_ylabel("CDF")
            axs[1, 1].legend()
            axs[1, 1].grid(True)

            # Automatically adjust layout to avoid overlapping elements
            plt.tight_layout()
            plt.show()

# Example of running multiple simulations
def main():
    results = run_multiple_simulations()  # Run the simulation with multiple runs
    plot_simulation_results(results)

main()
