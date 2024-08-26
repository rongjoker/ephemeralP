from collections import deque


def robot_navigation(grid: list[list[int]], k: int) -> bool:
    rows, cols = len(grid), len(grid[0])
    # 记录当前的行和列以及到达这个位置时已经demolish的墙的数量
    queue = deque([(0, 0, 0)])
    # 记录某个点最小demolish 的墙的数量
    visited = {(0, 0): 0}
    # 包含四个方向（右、下、左、上），用于遍历当前点的邻居
    forwards = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        r, c, walls_demolished = queue.popleft()
        # 抵达右下角
        if r == rows - 1 and c == cols - 1:
            return True

        for dr, dc in forwards:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                # 更新当前需要demolish的墙的数量
                new_walls_demolished = walls_demolished + grid[nr][nc]

                if new_walls_demolished <= k:
                    # 如果当前点没有访问过，或者当前需要demolish的墙的数量更少，则更新当前需要demolish的墙的数量
                    if (nr, nc) not in visited or visited[(nr, nc)] > new_walls_demolished:
                        visited[(nr, nc)] = new_walls_demolished
                        queue.append((nr, nc, new_walls_demolished))

    return False


# Example usage:
grid = [[0, 1, 0, 0, 0],
 [0, 1, 0 ,1, 0],
 [0, 1, 0, 1, 0],
 [0, 1, 0, 1, 0],
 [0, 0, 0, 1, 0]]
k = 0

print(robot_navigation(grid, k))  # Output: True