import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to calculate Euclidean distance between two points
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to find the nearest unvisited city
def nearest_neighbor(current_city, unvisited_cities):
    min_dist = float('inf')
    nearest_city = None
    for city in unvisited_cities:
        dist = distance(current_city, city)
        if dist < min_dist:
            min_dist = dist
            nearest_city = city
    return nearest_city

# Function to solve TSP using nearest neighbor algorithm
def solve_tsp(points):
    visited_cities = []
    unvisited_cities = points.tolist()  # Convert numpy array to list of tuples
    start_city = unvisited_cities.pop(0)  # Start from the first city
    visited_cities.append(start_city)
    
    current_city = start_city
    while unvisited_cities:
        nearest_city = nearest_neighbor(current_city, unvisited_cities)
        visited_cities.append(nearest_city)
        unvisited_cities.remove(nearest_city)
        current_city = nearest_city
    
    # Return to the starting city
    visited_cities.append(start_city)
    
    return np.array(visited_cities)  # Convert back to numpy array for easier plotting

# Generate random points (cities)
np.random.seed(2)
num_points = 15
points = np.random.rand(num_points, 2)  # Random 2D points

# Solve TSP
path = solve_tsp(points)

# Animation function
def animate(i):
    plt.cla()
    plt.plot(points[:, 0], points[:, 1], 'bo', label='Cities')  # Plot cities
    plt.plot(path[:i+1, 0], path[:i+1, 1], 'r-', marker='o')  # Plot tour path
    plt.text(0.65, 0.015, 'PrasannaMummigatti',bbox=dict(facecolor='gray', alpha=0.5))  # Plot cities
    plt.title('Nearest Neighbour TSP')
    plt.legend()

# Plot and animate
fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, frames=len(path), interval=1000, repeat=True)
plt.show()
