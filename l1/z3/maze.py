import random
import sys
import time
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Deque, List, Tuple


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


def neighbours(pos):
    return [tuple_sum(pos, move.value) for move in Move]


class Maze:
    def __init__(self, maze_map: List[str]):
        self.map = maze_map
        self.start = next(
            (i, j)
            for i, row in enumerate(maze_map)
            for j, field in enumerate(row)
            if field == Field.AGENT.value
        )
        self.size = len(maze_map) * len(maze_map[0])
        self.n, self.m = len(maze_map), len(maze_map[0])

    def path_cost(self, path: List[Move]) -> int:
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

    def neighbours(self, pos):
        n = [tuple_sum(pos, move.value) for move in Move]
        return [((x, y), self.map[x][y]) for x, y in n]

    @staticmethod
    def next_move(move):
        moves = [m.value for m in Move]
        return moves[(moves.index(move) + 1) % 4]

    @staticmethod
    def tweak_path(path):
        if len(path) < 2:
            return path
        if random.random() < 0.6:
            i = random.randrange(len(path))
            j = random.choice([c for c in range(1, len(path) - 1) if c < i - 1 or c > i + 1])
            path[i], path[j] = path[j], path[i]
            return path
        i = random.randrange(len(path) - 1)
        j = random.choice([c for c in range(len(path)) if c > i])
        return path[:i] + list(reversed(path[i:j])) + path[j:]

    def init_path(self):
        path = []
        x, y = self.start
        move = Move.R.value
        while self.map[x][y] != Field.EXIT.value:
            for m in [m.value for m in Move]:
                nx, ny = tuple_sum((x, y), m)
                if (
                    nx in range(self.n)
                    and ny in range(self.m)
                    and self.map[nx][ny] == Field.EXIT.value
                ):
                    path.append(m)
                    return path

            nx, ny = tuple_sum((x, y), move)

            if (
                nx not in range(self.n)
                or ny not in range(self.m)
                or self.map[nx][ny] == Field.WALL.value
            ):
                move = self.next_move(move)
            else:
                x, y = nx, ny
                path.append(move)
        return path


class TabuList:
    def __init__(self, max_size: int):
        self._list: Deque[Any] = deque([], max_size)

    def push(self, element: Any):
        self._list.append(element)

    def __str__(self):
        return str(list(self._list))

    def contains(self, element: Any):
        return element in self._list


def now():
    return time.time()


def tuple_sum(t1, t2):
    return tuple(sum(t) for t in zip(t1, t2))


def tabu_search(
    initial: List[Move],
    tweak: Callable[[List[Move]], List[Move]],
    quality: Callable[[List[Move]], int],
    tabu_size: int,
    num_of_tweaks: int,
    timeout: float,
) -> Tuple[List[Move], int]:
    s = initial
    best = s
    tabu = TabuList(tabu_size)
    tabu.push(s)

    start = now()
    while now() - start <= timeout:
        r = tweak(deepcopy(s))
        for _ in range(num_of_tweaks):
            if now() - start > timeout:
                break
            w = tweak(deepcopy(s))
            if not tabu.contains(w) and (tabu.contains(r) or quality(w) < quality(r)):
                r = w
        if not tabu.contains(r):
            s = r
            tabu.push(r)
        if quality(s) < quality(best):
            print(f"new best! {quality(s)} - after {now() - start} s.", file=sys.stderr)
            best = s
    return best, quality(best)


def main():
    t, n, m = map(int, input().split())
    maze = Maze([input() for _ in range(n)])

    path, cost = tabu_search(
        initial=maze.init_path(),
        tweak=maze.tweak_path,
        quality=maze.path_cost,
        tabu_size=n * 10,
        num_of_tweaks=n * m,
        timeout=float(t),
    )

    print(cost)
    print("".join(Move(move).name for move in path[:cost]), file=sys.stderr)


if __name__ == "__main__":
    main()
