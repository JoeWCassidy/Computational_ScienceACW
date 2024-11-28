import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import Counter
from scipy import stats
from scipy.stats import chisquare, kstest, bernoulli,chi2
from concurrent.futures import ThreadPoolExecutor


# ---- Part I: Cellular Automata Movement Simulation ----

def task_1_1(total_steps=100, checkpoints=[10, 25, 50, 100], run_id=1):
    """Simulate movement of a cell with checkpoints for uniformity analysis."""
    grid_size = 100
    x, y = grid_size // 2, grid_size // 2  # Start at the center
    positions = [(x, y)]
    directions = []
    step_data = {checkpoint: [] for checkpoint in checkpoints}
    
    # Simulate cell movement
    for step in range(1, total_steps + 1):
        rand1, rand2 = np.random.choice([0, 1], size=2)  # Generate random directions
        if rand1 == 1 and rand2 == 1:
            y = max(0, y - 1)  # Move up
            directions.append("Up")
        elif rand1 == 1 and rand2 == 0:
            y = min(grid_size - 1, y + 1)  # Move down
            directions.append("Down")
        elif rand1 == 0 and rand2 == 1:
            x = max(0, x - 1)  # Move left
            directions.append("Left")
        elif rand1 == 0 and rand2 == 0:
            x = min(grid_size - 1, x + 1)  # Move right
            directions.append("Right")
        positions.append((x, y))

        if step in checkpoints:
            step_data[step] = Counter(directions).copy()

    # Return the simulation data to be used for plotting later
    return run_id, positions, step_data

def run_multiple_simulations():
    """Run three simulations of task_1_1 concurrently to reduce bias."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(task_1_1, run_id=i) for i in range(1, 4)]
        results = [future.result() for future in futures]  # Collect results for all runs
    return results  # Return all simulation results

def plot_simulation_results(results):
    """Plot the results for all runs after they are finished."""
    for run_id, positions, step_data in results:
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
            observed = [counts.get(direction, 0) for direction in ["Up", "Down", "Left", "Right"]]
            expected = [step / 4] * 4  # Uniform distribution expectation
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
            x_vals = np.linspace(0, 10, 1000)
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
            observed_probabilities = [counts.get(direction, 0) / step for direction in ["Up", "Down", "Left", "Right"]]
            ks_statistic, ks_p_value = kstest(observed_probabilities, 'uniform')
            print(f"\nKolmogorov-Smirnov Test Results (Run {run_id}, Checkpoint {step}):")
            print(f"KS Statistic: {ks_statistic:.4f}, p-value: {ks_p_value:.4f}")
            if ks_p_value > 0.05:
                print("Result: The data is consistent with a uniform distribution (p > 0.05) according to the KS test.")
            else:
                print("Result: The data significantly deviates from a uniform distribution (p ≤ 0.05) according to the KS test.")

            # KS Test (CDF plot) and KS Statistic (fourth subplot)
            uniform_cdf = np.cumsum([1/4] * 4)  # Uniform distribution CDF
            observed_cdf = np.cumsum(observed_probabilities)

            axs[1, 1].plot(["Up", "Down", "Left", "Right"], observed_cdf, label="Observed CDF", color='b', marker='o')
            axs[1, 1].plot(["Up", "Down", "Left", "Right"], uniform_cdf, label="Expected CDF (Uniform)", color='r', linestyle='--')
            axs[1, 1].set_title(f"KS Test: CDF of Observed vs Expected (Checkpoint {step})")
            axs[1, 1].set_xlabel("Direction")
            axs[1, 1].set_ylabel("CDF")
            axs[1, 1].legend()
            axs[1, 1].grid(True)

            # Automatically adjust layout to avoid overlapping elements
            plt.tight_layout()
            plt.show()

def task_1_2():
    """Simulate movement in 8 directions for 1000 and 10000 steps."""
    grid_size = 100
    steps = [1000, 10000]

    directions_map = {
        0: "Right", 1: "Left", 2: "Down", 3: "Up", 
        4: "Down-Right", 5: "Down-Left", 6: "Up-Right", 7: "Up-Left"
    }

    moves = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1)
    ]

    for step_count in steps:
        x, y = grid_size // 2, grid_size // 2  # Start at the center of the grid
        positions = [(x, y)]
        directions = []

        for _ in range(step_count):
            # Generate a random integer between 0 and 7
            direction = bernoulli.rvs(0.5, size=3).dot([1, 2, 4])  # Binary to integer for 8 directions
            dx, dy = moves[direction]
            x = min(max(0, x + dx), grid_size - 1)
            y = min(max(0, y + dy), grid_size - 1)
            directions.append(directions_map[direction])
            positions.append((x, y))

        x_positions, y_positions = zip(*positions)

        # Plot movement and direction distribution
        plt.figure(figsize=(10, 5))

        # Plot path (first subplot)
        plt.subplot(1, 2, 1)
        plt.plot(x_positions, y_positions, marker='o', linestyle='-', markersize=2, label=f'{step_count} Steps')
        plt.title(f"Cell Movement on 100x100 Grid ({step_count} Steps)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.legend(title="Movement Path")

        # Direction distribution (second subplot)
        direction_counts = Counter(directions)
        plt.subplot(1, 2, 2)
        plt.bar(direction_counts.keys(), direction_counts.values(), color='orange')
        plt.title(f"Direction Distribution ({step_count} Steps)")
        plt.xlabel("Direction")
        plt.ylabel("Frequency")
        
        # Adjusting x-axis labels for better readability
        plt.xticks(ticks=range(8), labels=directions_map.values(), rotation=45, ha="right", fontsize=12)
        plt.xlim(-0.5, 7.5)  # Slightly expand the x-axis for better space for labels
        
        plt.grid(True)

        # Making sure the layout is tight and all elements fit nicely
        plt.tight_layout()
        plt.show()

def task_2_1():
    """Simulate tumor growth using the Gompertz model."""
    k = 0.006
    M = 10**13
    N0 = 10**9
    t = np.linspace(0, 1200, 1000)

    def gompertz(N, t, k, M):
        return k * N * np.log(M / N)

    N = odeint(gompertz, N0, t, args=(k, M))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, N, label="Tumor Size")
    plt.title("Tumor Growth Using the Gompertz Model")
    plt.xlabel("Time")
    plt.ylabel("Tumor Size (N)")
    plt.grid(True)
    
    # Logarithmic plot for better visualization of growth
    plt.subplot(1, 2, 2)
    plt.plot(t, np.log(N), label="Log(Tumor Size)", color='green')
    plt.title("Log-Tumor Growth")
    plt.xlabel("Time")
    plt.ylabel("Log(Tumor Size)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def task_2_2():
    """Simulate tumor movement and growth across a 100x100 grid."""
    grid_size = 100
    steps = [1000, 10000]

    for step_count in steps:
        x, y = grid_size // 2, grid_size // 2
        positions = [(x, y)]

        for _ in range(step_count):
            x += 1 if random.randint(0, 1) == 0 else -1
            y += 1 if random.randint(0, 1) == 0 else -1
            x = max(0, min(grid_size - 1, x))
            y = max(0, min(grid_size - 1, y))
            positions.append((x, y))

        x_positions, y_positions = zip(*positions)

        plt.figure(figsize=(6, 6))
        plt.plot(x_positions, y_positions, marker='o', linestyle='-', markersize=1)
        plt.title(f"Tumor Movement on 100x100 Grid ({step_count} Steps)")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()

# ---- Menu System ----
def menu():
    while True:
        print("\nMenu:")
        print("1. Task 1.1: Cellular Automata Movement (100 Steps) - Simulates movement on a 100x100 grid with 4 directions.")
        print("2. Task 1.2: Cellular Automata Movement (1000 and 10000 Steps) - Simulates movement with 8 directions and more steps.")
        print("3. Task 2.1: Tumor Growth (Gompertz Model) - Models tumor growth using the Gompertz equation over time.")
        print("4. Task 2.2: Tumor Movement on Grid (1000 and 10000 Steps) - Simulates tumor movement across a 100x100 grid with random steps.")
        print("5. Exit")
        
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                results = run_multiple_simulations()  # Run the simulation with multiple runs
                plot_simulation_results(results)
            elif choice == 2:
                task_1_2()

            elif choice == 3:
                task_2_1()
            elif choice == 4:
                task_2_2()
            elif choice == 5:
                print("Exiting program.")
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# ---- Main Execution ----
if __name__ == "__main__":
    menu()
