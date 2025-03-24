import heapq
import numpy as np
import math

def heuristic(a, b):
    """Euclidean distance heuristic for grid"""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def astar(grid, start, goal):
    """
    Run A* on a binary grid with 8-connected movement (diagonals allowed).
    grid: 2D numpy array (0 = free, 1 = obstacle)
    start, goal: (y, x) integer tuples
    Returns: list of (y, x) tuples for path or None
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    # 8-connected movement: (dy, dx)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            ny, nx = neighbor
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 0:

                # Prevent corner cutting for diagonal directions
                if dy != 0 and dx != 0:
                    # Check bounds for both adjacent sides before accessing grid
                    if not (0 <= current[0] + dy < rows and 0 <= current[1] < cols and
                            0 <= current[0] < rows and 0 <= current[1] + dx < cols):
                        continue
                    if grid[current[0] + dy, current[1]] == 1 or grid[current[0], current[1] + dx] == 1:
                        continue
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] == 0:
                # Cost = 1 for cardinal, sqrt(2) for diagonal
                step_cost = math.sqrt(2) if dy != 0 and dx != 0 else 1
                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    return None  # No path found