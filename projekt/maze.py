import sys
import random
from enum import Enum
from functools import lru_cache
from math import sqrt
from random import choices
from typing import List, Tuple

import metaheuristics as meta


class Move(Tuple[int, int], Enum):
    U = -1, 0
    R = 0, 1
    D = 1, 0
    L = 0, -1


class Field(str, Enum):
    EMPTY = "0"
    WALL = "1"
    VERTICAL_TUNNEL = "2"
    HORIZONTAL_TUNNEL = "3"
    AGENT = "5"
    EXIT = "8"

    # The agent, inside tunnels, can move only in one direction
    @lru_cache(maxsize=None)
    def tunnel_validate(self, move: Tuple[int, int]) -> bool:
        if self == Field.HORIZONTAL_TUNNEL and move in (Move.U, Move.D):
            return False
        if self == Field.VERTICAL_TUNNEL and move in (Move.L, Move.R):
            return False
        return True


Path = List[Tuple[int, int]]


@lru_cache(maxsize=None)
def tuple_sum(t1: tuple, t2: tuple):
    return tuple(sum(t) for t in zip(t1, t2))


class Maze:
    def __init__(self, maze_map: List[str]):
        self.map = maze_map
        self.start = next(
            (i, j)
            for i, row in enumerate(maze_map)
            for j, field in enumerate(row)
            if field == Field.AGENT
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

            # in tunnel, move only in tunnel direction
            cur_x, cur_y = pos
            cur_field = self.map[cur_x][cur_y]
            if not Field(cur_field).tunnel_validate(move):
                continue

            # new position
            x, y = tuple_sum(pos, move)

            new_field = self.map[x][y]
            if new_field == Field.EXIT:
                return cost

            # treat sides of tunnels as walls
            if (new_field != Field.WALL) and Field(new_field).tunnel_validate(move):
                pos = x, y

        return self.size

    @staticmethod
    def tweak_path(path) -> Path:
        if random.random() < 0.75:  # swap moves
            i = random.randrange(len(path))
            j = random.randrange(len(path))
            path[i], path[j] = path[j], path[i]
            return path

        i = random.randrange(len(path) - 1)
        j = random.choice([c for c in range(len(path)) if c >= i])

        if random.random() > 0.2:
            return path[:i] + choices([m.value for m in Move], k=j - i) + path[j:]

        return path[:i] + list(reversed(path[i:j])) + path[j:]


def main():
    t, n, m = map(int, input().split())
    maze = Maze([input() for _ in range(n)])
    initial_path = [Move[m].value for m in input().strip()]

    path, cost = meta.simulated_annealing(
        function=maze.path_cost,
        initial_solution=initial_path,
        initial_temperature=1.618,
        cooling_schedule=lambda t: t * 0.85,
        tweak=maze.tweak_path,
        timeout=t,
        deep_input=True,
        local_timeout=sqrt(t),
    )
    print(cost)
    print("".join(Move(move).name for move in path[:cost]), file=sys.stderr)


if __name__ == "__main__":
    main()
