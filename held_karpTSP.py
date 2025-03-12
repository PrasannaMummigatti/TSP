import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import combinations

def held_karp(dist_matrix):
    n = len(dist_matrix)
    dp = {}
    
    for i in range(1, n):
        dp[(1 << i, i)] = (dist_matrix[0][i], 0)
    
    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = sum(1 << i for i in subset)
            
            for j in subset:
                prev_bits = bits & ~(1 << j)
                best_cost = float("inf")
                best_prev = -1
                
                for k in subset:
                    if k == j:
                        continue
                    cost = dp[(prev_bits, k)][0] + dist_matrix[k][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = k
                
                dp[(bits, j)] = (best_cost, best_prev)
    
    bits = (1 << n) - 2
    best_cost = float("inf")
    best_prev = -1
    
    for j in range(1, n):
        cost = dp[(bits, j)][0] + dist_matrix[j][0]
        if cost < best_cost:
            best_cost = cost
            best_prev = j
    
    path = [0]
    last = best_prev
    bits = (1 << n) - 2
    
    for _ in range(n - 1):
        path.append(last)
        bits, last = bits & ~(1 << last), dp[(bits, last)][1]
    
    path.append(0)
    
    return best_cost, path

def animate_tsp_path(coords, path):
    fig, ax = plt.subplots()
    ax.scatter(*zip(*coords), color='red', s=100, zorder=2)
    
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), fontsize=12, ha='right', color='blue')
    
    plt.ion()
    ax.set_title("Held–Karp algorithm TSP")
    plt.text(0.01, 0.01, 'PrasannaMummigatti',bbox=dict(facecolor='lightgray', alpha=0.5))  # Plot cities
    plt.show()

    plt.pause(10)
    for i in range(len(path) - 1):
        x_start, y_start = coords[path[i]]
        x_end, y_end = coords[path[i + 1]]
        
        num_steps = 5
        
        for t in np.linspace(0, 1, num_steps):
            x_interp = x_start + t * (x_end - x_start)
            y_interp = y_start + t * (y_end - y_start)
            
            ax.plot([x_start, x_interp], [y_start, y_interp], 'b-', zorder=1)
            
            plt.draw()
            plt.pause(0.005)
        
        #ax.plot([x_start, x_end], [y_start, y_end], 'b-', zorder=1)
    
    plt.ioff()
    plt.show()

np.random.seed(42)
n = 15
coords = np.random.rand(n, 2) * 100
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])

cost, path = held_karp(distance_matrix)
print(f"Optimal TSP cost: {cost}")
print(f"Optimal TSP path: {path}")
animate_tsp_path(coords, path)