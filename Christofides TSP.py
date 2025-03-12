import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from itertools import combinations
from matplotlib.animation import FuncAnimation

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Generate a random weighted graph for TSP
def generate_random_graph(num_nodes):
    G = nx.Graph()
    nodes = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = euclidean_distance(nodes[i], nodes[j])
            G.add_edge(i, j, weight=dist)
    return G, nodes

# Christofides Algorithm implementation
def christofides_tsp(G):
    # Step 1: Compute Minimum Spanning Tree (MST)
    MST = nx.minimum_spanning_tree(G)

    # Step 2: Find odd-degree vertices
    odd_vertices = [v for v in MST.nodes if MST.degree(v) % 2 == 1]
    
    print(f"Odd-degree vertices (before matching): {odd_vertices}")

    # Step 3: Compute Minimum Weight Perfect Matching on odd-degree vertices
    if len(odd_vertices) % 2 != 0:
        raise ValueError("Odd-degree vertices count is not even, which should not happen!")

    if odd_vertices:
        odd_graph = nx.Graph()
        for u, v in combinations(odd_vertices, 2):
            odd_graph.add_edge(u, v, weight=G[u][v]['weight'])  # Use positive weights

        # Ensure perfect matching of all odd-degree vertices
        min_weight_matching = nx.algorithms.matching.min_weight_matching(odd_graph)

        if len(min_weight_matching) * 2 != len(odd_vertices):  # Ensure all odd vertices are matched
            raise ValueError("Matching step failed to perfectly match all odd-degree nodes.")

        # Add matching edges to MST
        for u, v in min_weight_matching:
            MST.add_edge(u, v, weight=G[u][v]['weight'])

    # Debugging: Check if all degrees are even now
    degrees = {v: MST.degree(v) for v in MST.nodes}
    print(f"Node degrees after matching: {degrees}")

    # Step 4: Convert to Eulerian graph and find Eulerian Circuit
    if not nx.is_eulerian(MST):
        raise ValueError("Graph is not Eulerian after matching. Check the matching step.")

    eulerian_circuit = list(nx.eulerian_circuit(MST))

    # Step 5: Convert Eulerian circuit into TSP tour (removing duplicates)
    visited = set()
    tour = []
    for edge in eulerian_circuit:
        if edge[0] not in visited:
            tour.append(edge[0])
            visited.add(edge[0])
    tour.append(tour[0])  # Complete the cycle

    return MST, min_weight_matching, eulerian_circuit, tour

# Plot the graph
def plot_graph(ax, G, nodes, edges=None, title="Graph", highlight_edges=None, highlight_color="r"):
    ax.clear()
    pos = {i: nodes[i] for i in range(len(nodes))}
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    
    # Highlight selected edges
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color=highlight_color, width=2, ax=ax)
    
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# Animate the steps of Christofides Algorithm
def animate_christofides(num_nodes=10):
    G, nodes = generate_random_graph(num_nodes)
    MST, matching, eulerian_circuit, tour = christofides_tsp(G)

    fig, ax = plt.subplots(figsize=(8, 6))

    steps = [
        ("Original Graph", G.edges, "gray"),
        ("Minimum Spanning Tree (MST)", MST.edges, "blue"),
        ("Minimum Weight Matching", matching, "green"),
        ("Eulerian Circuit", eulerian_circuit, "purple"),
        ("Final TSP Tour", list(zip(tour, tour[1:])), "red"),
    ]

    def update(frame):
        title, edges, color = steps[frame]
        plot_graph(ax, G, nodes, edges=edges, title=title, highlight_edges=edges, highlight_color=color)
    plt.text(0.0, 0.0, 'PrasannaMummigatti',bbox=dict(facecolor='lightgray', alpha=0.5))
    ani = FuncAnimation(fig, update, frames=len(steps), interval=1500, repeat=True)
    plt.show()

# Run the animation
animate_christofides(num_nodes=8)
