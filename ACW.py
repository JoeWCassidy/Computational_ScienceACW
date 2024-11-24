import numpy as np
import matplotlib.pyplot as plt
import time  # Import time for measuring execution time

# Show Bar Graph for Simulating the tumour show the bias can still be shown and demonstrated 

# Dictionary for caching growth data
growth_cache = {}

# Function to simulate Gompertz tumor growth with caching
def gompertz_growth(N0, k, M, t_max, dt=0.01):  
    cache_key = (N0, k, M, t_max, dt)
    if cache_key in growth_cache:
        return growth_cache[cache_key]
    
    time_range = np.arange(0, t_max, dt)
    N_t = np.zeros_like(time_range)
    N_t[0] = N0
    for i in range(1, len(time_range)):
        N_t[i] = N_t[i-1] * np.exp(k * (1 - np.log(N_t[i-1] / M)) * dt)
        if np.abs(N_t[i] - M) < 0.001:  # Consider steady state if it's very close to M not going to be M (See lecture notes) 
            result = (time_range[:i+1], N_t[:i+1], i)
            growth_cache[cache_key] = result
            return result
    
    result = (time_range, N_t, None)  # None means no steady state
    growth_cache[cache_key] = result
    return result

# Function to simulate tumor growth in a single cell
def simulate_single_cell_growth(N0, k, M, t_max):
    time_range, N_t, _ = gompertz_growth(N0, k, M, t_max)
    
    # Plot the tumor growth data
    plt.figure(figsize=(10, 6))
    plt.plot(time_range, N_t, label='Tumor Growth', color='blue')
    plt.axhline(y=M, color='r', linestyle='--', label=f'Steady State M = {M:.0e}')
    plt.title('Tumor Growth in a Single Cell')
    plt.xlabel('Time')
    plt.ylabel('Number of Cells')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to simulate 4-direction movement, count directions, and show 0's and 1's in X and Y directions
def simulate_4_direction_movement(grid_size=100):
    # Initialize grid
    grid = np.zeros((grid_size, grid_size))
    x, y = grid_size // 2, grid_size // 2  # Start at the center
    grid[x, y] = 1  # Mark the starting cell as visited

    # Randomly move in 4 directions
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    move_counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0}  # Direction counts
    path = [(x, y)]

    for _ in range(100):  # Limit number of moves
        dx, dy = moves[np.random.choice(len(moves))]  # Choose a random direction
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            x, y = new_x, new_y
            path.append((x, y))
            grid[x, y] = 1  # Mark the new cell as visited

            # Update direction counts
            if dx == -1:
                move_counts['left'] += 1
            elif dx == 1:
                move_counts['right'] += 1
            elif dy == -1:
                move_counts['down'] += 1
            elif dy == 1:
                move_counts['up'] += 1

    # Count visited cells in X and Y directions (rows and columns)
    visited_rows = np.sum(grid, axis=1)  # Sum of 1s in each row
    visited_columns = np.sum(grid, axis=0)  # Sum of 1s in each column

    num_ones_rows = np.count_nonzero(visited_rows)  # Count of rows with at least one visited cell
    num_zeros_rows = grid_size - num_ones_rows  # Rows with no visited cells
    num_ones_columns = np.count_nonzero(visited_columns)  # Count of columns with at least one visited cell
    num_zeros_columns = grid_size - num_ones_columns  # Columns with no visited cells

    # Plot direction counts and 0's vs 1's
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot histogram of direction counts
    axes[0].bar(move_counts.keys(), move_counts.values(), color=['blue', 'green', 'red', 'purple'])
    axes[0].set_title('Direction Counts (4 Directions)')
    axes[0].set_xlabel('Direction')
    axes[0].set_ylabel('Count')

    # Plot histogram of visited rows and columns
    axes[1].bar(['0s in X', '1s in X', '0s in Y', '1s in Y'], 
                [num_zeros_rows, num_ones_rows, num_zeros_columns, num_ones_columns],
                color=['blue', 'green', 'orange', 'red'])
    axes[1].set_title('Visited Rows and Columns (0s and 1s)')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

# Function to simulate 8-direction movement, count directions, and show 0's and 1's in X and Y directions
def simulate_8_direction_movement(grid_size=100):
    # Initialize grid
    grid = np.zeros((grid_size, grid_size))
    x, y = grid_size // 2, grid_size // 2  # Start at the center
    grid[x, y] = 1  # Mark the starting cell as visited

    # Randomly move in 8 directions
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # up, down, left, right, diagonals
    move_counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'upleft': 0, 'upright': 0, 'downleft': 0, 'downright': 0}
    path = [(x, y)]

    for _ in range(100):  # Limit number of moves
        dx, dy = moves[np.random.choice(len(moves))]  # Choose a random direction
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            x, y = new_x, new_y
            path.append((x, y))
            grid[x, y] = 1  # Mark the new cell as visited

            # Update direction counts
            if dx == -1 and dy == 0:
                move_counts['left'] += 1
            elif dx == 1 and dy == 0:
                move_counts['right'] += 1
            elif dx == 0 and dy == -1:
                move_counts['down'] += 1
            elif dx == 0 and dy == 1:
                move_counts['up'] += 1
            elif dx == -1 and dy == -1:
                move_counts['upleft'] += 1
            elif dx == -1 and dy == 1:
                move_counts['upright'] += 1
            elif dx == 1 and dy == -1:
                move_counts['downleft'] += 1
            elif dx == 1 and dy == 1:
                move_counts['downright'] += 1

    # Count visited cells in X and Y directions (rows and columns)
    visited_rows = np.sum(grid, axis=1)  # Sum of 1s in each row
    visited_columns = np.sum(grid, axis=0)  # Sum of 1s in each column

    num_ones_rows = np.count_nonzero(visited_rows)  # Count of rows with at least one visited cell
    num_zeros_rows = grid_size - num_ones_rows  # Rows with no visited cells
    num_ones_columns = np.count_nonzero(visited_columns)  # Count of columns with at least one visited cell
    num_zeros_columns = grid_size - num_ones_columns  # Columns with no visited cells

    # Plot direction counts and 0's vs 1's
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot histogram of direction counts
    axes[0].bar(move_counts.keys(), move_counts.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'pink', 'brown'])
    axes[0].set_title('Direction Counts (8 Directions)')
    axes[0].set_xlabel('Direction')
    axes[0].set_ylabel('Count')

    # Plot histogram of visited rows and columns
    axes[1].bar(['0s in X', '1s in X', '0s in Y', '1s in Y'], 
                [num_zeros_rows, num_ones_rows, num_zeros_columns, num_ones_columns],
                color=['blue', 'green', 'orange', 'red'])
    axes[1].set_title('Visited Rows and Columns (0s and 1s)')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


# Updated grid simulation for tumor growth
def simulate_grid_tumor_growth(grid_size=100, M=10**13, k=0.006, N0=10**9, t_max=1200):
    # Initialize grid
    grid = np.zeros((grid_size, grid_size))
    x, y = grid_size // 2, grid_size // 2  # Start at the center
    grid[x, y] = 1  # Mark the starting cell as visited

    # Start timing after grid setup
    start_time = time.time()

    steady_cells = 0
    visited_cells = {(x, y)}  # Use a set to keep track of visited cells
    tumor_path = [(x, y)]
    growth_data = {}

    # Moves in the grid: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while steady_cells < 2:  # Continue until two cells have reached steady state
        print(f"Simulating tumor growth in cell ({x}, {y})")
        time_range, N_t, steady_index = gompertz_growth(N0, k, M, t_max)

        if steady_index is not None:
            print(f"Steady state reached in cell ({x}, {y}) at time {time_range[steady_index]:.2f}")
            steady_cells += 1
            growth_data[(x, y)] = (time_range[:steady_index + 1], N_t[:steady_index + 1])  # Capture growth data up to steady state

        # Randomly move to an unvisited neighboring cell
        np.random.shuffle(moves)
        moved = False  # Flag to track if a move was made
        for dx, dy in moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < grid_size and 0 <= new_y < grid_size and (new_x, new_y) not in visited_cells:
                x, y = new_x, new_y
                tumor_path.append((x, y))
                grid[x, y] = 1  # Mark the new cell as visited
                visited_cells.add((x, y))  # Add new cell to visited cells
                moved = True  # Move made
                break
        
        if not moved:
            print("No unvisited neighbors left to move to, stopping simulation.")
            break

    # Plot the tumor path on the grid
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='Greys', origin='lower', extent=(0, grid_size, 0, grid_size))
    plt.scatter([p[1] for p in tumor_path], [p[0] for p in tumor_path], color='red', label='Tumor Path')
    plt.title('Tumor Movement on Grid')
    plt.xlabel('X-axis (grid)')
    plt.ylabel('Y-axis (grid)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the tumor growth data for all cells that reached steady state
    plt.figure(figsize=(10, 6))
    for cell, (time_range, N_t) in growth_data.items():
        plt.plot(time_range, N_t, label=f'Cell {cell} Growth')
    plt.axhline(y=M, color='r', linestyle='--', label=f'Steady State M = {M:.1e}')
    plt.title('Tumor Growth in Cells')
    plt.xlabel('Time')
    plt.ylabel('Number of Cells')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate and print the total execution time
    end_time = time.time()  # End timing the function
    print(f"Grid tumor growth simulation executed in {end_time - start_time:.2f} seconds.")

# CLI menu for user interaction
def cli_menu():
    print("Choose a simulation:")
    print("1. Simulate 4-direction movement")
    print("2. Simulate 8-direction movement")
    print("3. Simulate single cell tumor growth")
    print("4. Simulate grid tumor growth")
    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        simulate_4_direction_movement()
    elif choice == '2':
        simulate_8_direction_movement()
    elif choice == '3':
        N0 = 10**9
        k = 0.006
        M = 10**13
        t_max = 1200
        simulate_single_cell_growth(N0, k, M, t_max)
    elif choice == '4':
        simulate_grid_tumor_growth()
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")

# Run the menu
cli_menu()
