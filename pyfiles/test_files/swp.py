from collections import deque
import pandas as pd

def loadGraph(edgeFilename): # required function
    # Read file
    edges = pd.read_csv(edgeFilename, sep=' ', header=None)
    
    # empty dictionary
    adjacent_list = {}
    
    # Iterate over each row in the dataframe
    for index, row in edges.iterrows():
        # Add the edge to the adjacency list
        adjacent_list.setdefault(row[0], []).append(row[1])
        adjacent_list.setdefault(row[1], []).append(row[0])
    
    return adjacent_list

# Test
adjacent_list = loadGraph('edges_2_2.txt')

# Print the first items in adjacent list
print("Adjacent List/Dictionary:")
print(list(adjacent_list.items())[:5])
print("")

# Define MyQueue class; required
class MyQueue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.popleft()

    def empty(self):
        return len(self.items) == 0

# Define the BFS function
def BFS(G, s):
    max_node = max(max(G.keys()), max([item for sublist in G.values() for item in sublist]))  # Find the maximum node number
    queue = MyQueue()  # Initialize a queue
    visited = [False] * (max_node + 1)  # a list to keep track of visited nodes
    distance = [-1] * (max_node + 1)  # a list to store distances

    queue.enqueue(s)  # Add the source node to the queue
    visited[s] = True  # Mark the source node as visited
    distance[s] = 0  # The distance from the source node to itself is 0

    # While loop: the queue is not empty
    while not queue.empty():
        vertex = queue.dequeue()  # Dequeue a vertex
        for neighbour in G.get(vertex, []):  # For each neighbour of the vertex
            if visited[neighbour] == False:  # If the neighbour has not been visited
                queue.enqueue(neighbour)  # Enqueue the neighbour
                visited[neighbour] = True  # Mark the neighbour as visited
                distance[neighbour] = distance[vertex] + 1  # Update the distance to the neighbour
        print(f"Processed node {vertex}. Queue: {list(queue.items)}")


    return distance  # Return the list of distances

# Define distanceDistribution function
def distanceDistribution(G):
    distance_distribution = {}  # Initialize a dictionary to store the distance distribution
    total_distances = 0  # Initialize a counter for the total number of distances

    # For each vertex in the graph
    for vertex in G.keys():
        distances = BFS(G, vertex)  # Run BFS to get distances from the vertex to all other vertices
        for distance in distances:  # For each distance
            if distance > 0:  # Check if the distance is positive
                # Increment the count for this distance in the dictionary
                distance_distribution[distance] = distance_distribution.get(distance, 0) + 1
                total_distances += 1  # Increment total number of distances

    # Convert counts to percentages
    for key in distance_distribution.keys():
        distance_distribution[key] = (distance_distribution[key] / total_distances) * 100

    return distance_distribution  # Return the distance distribution dictionary

# Open data file and read edges
with open('edgesshort_2_2_2.txt', 'r') as file:
    edges = file.readlines()

# Convert edges to list of pairs
edges = [list(map(int, edge.strip().split())) for edge in edges]

# empty dictionary container
G = {}

# For each edge in the list of edges
for edge in edges:
    # If the first node of the edge is not in the graph, add it and set its neighbours to be a list that has the second node
    if edge[0] not in G:
        G[edge[0]] = [edge[1]]
    else:  # If the first node is already in graph, then append the second node to list of neighbours
        G[edge[0]].append(edge[1])

    # Again for the second node of the edge
    if edge[1] not in G:
        G[edge[1]] = [edge[0]]
    else:
        G[edge[1]].append(edge[0])

final_distribution = distanceDistribution(G)
print("Final Distance Distribution Dictionary:")
print(final_distribution)


#To what extent does this network satisfy the small world phenomenon? ;

