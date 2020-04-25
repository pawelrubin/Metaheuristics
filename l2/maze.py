import random
import sys
from enum import Enum
from typing import List, Tuple

import metaheuristics as meta


class Move(Enum):
    U = -1, 0
    R = 0, 1
    D = 1, 0
    L = 0, -1


class Field(Enum):
    EMPTY = "0"
    WALL = "1"
    AGENT = "5"
    EXIT = "8"


Path = List[Tuple[int, int]]


def tuple_sum(t1: tuple, t2: tuple):
    return tuple(sum(t) for t in zip(t1, t2))


class Maze:
    def __init__(self, maze_map: List[str]):
        self.map = maze_map
        self.start = next(
            (i, j) for i, row in enumerate(maze_map) for j, field in enumerate(row) if field == Field.AGENT.value
        )
        self.size = len(maze_map) * len(maze_map[0])
        self.n, self.m = len(maze_map), len(maze_map[0])

    def path_cost(self, path: Path) -> int:
        pos = self.start
        cost = 0
        for i, move in enumerate(path):
            cost = i + 1
            if cost >= self.size:
                return self.size

            x, y = tuple_sum(pos, move)

            if x not in range(self.n) or y not in range(self.m):
                return self.size

            if self.map[x][y] == Field.EXIT.value:
                break

            if self.map[x][y] != Field.WALL.value:
                pos = x, y

        return cost

    @staticmethod
    def tweak_path(path) -> Path:
        if random.random() < 0.6:
            i = random.randrange(len(path))
            j = random.randrange(len(path))
            path[i], path[j] = path[j], path[i]
            return path
        i = random.randrange(len(path) - 1)
        j = random.choice([c for c in range(len(path)) if c > i])
        return path[:i] + list(reversed(path[i:j])) + path[j:]

    def init_path(self) -> Path:
        """
        creates a path of length no greater than a third of the maze
        Random walk resests after a third of the maze
        """
        moves = [m.value for m in Move]

        while True:
            path: Path = []
            x, y = self.start
            move = random.choice(moves)
            while len(path) < self.n * self.m / 3:
                nx, ny = tuple_sum((x, y), move)

                if nx in range(self.n) and ny in range(self.m):
                    if self.map[nx][ny] == Field.EXIT.value:
                        path.append(move)
                        return path
                    if self.map[nx][ny] == Field.WALL.value:
                        move = random.choice(moves)
                    else:
                        x, y = nx, ny
                        path.append(move)
                        if random.random() < 0.2:
                            move = random.choice(moves)
                else:
                    move = random.choice(moves)


def main():
    t, n, _ = map(int, input().split())
    maze = Maze([input() for _ in range(n)])

    path, cost = meta.simulated_annealing(
        function=maze.path_cost,
        initial_solution=maze.init_path(),
        initial_temperature=10 ** 5,
        cooling_schedule=lambda t, _: t * 0.99,
        tweak=maze.tweak_path,
        timeout=t,
        deep_input=True,
    )

    print(cost)
    print("".join(Move(move).name for move in path[:cost]), file=sys.stderr)


if __name__ == "__main__":
    main()
