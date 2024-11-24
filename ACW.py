import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import Counter

# ---- Part I: Cellular Automata Movement Simulation ----

def task_1_1():
    """Simulate movement of a cell on a 100x100 grid (4 directions, 100 steps)."""
    grid_size = 100
    x, y = grid_size // 2, grid_size // 2  # Start at the center of the grid
    positions = [(x, y)]
    directions = []

    for _ in range(100):
        rand1, rand2 = random.randint(0, 1), random.randint(0, 1)
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

    x_positions, y_positions = zip(*positions)

    # Plot movement and direction distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_positions, y_positions, marker='o', linestyle='-', markersize=4)
    plt.title("Cell Movement on 100x100 Grid (100 Steps)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)

    direction_counts = Counter(directions)
    plt.subplot(1, 2, 2)
    plt.bar(direction_counts.keys(), direction_counts.values(), color='orange')
    plt.title("Direction Distribution (100 Steps)")
    plt.xlabel("Direction")
    plt.ylabel("Frequency")
    plt.grid(True)

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
            direction = random.randint(0, 7)
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
                task_1_1()
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
