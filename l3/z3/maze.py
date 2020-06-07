from random import choice, random, randrange, shuffle
import sys
from enum import Enum
from typing import Any, List, Tuple

try:
    from .ga import ga, Vector, VectorFunc, VectorTweak, VectorSelection
except ImportError:
    from ga import ga, Vector, VectorFunc, VectorTweak, VectorSelection


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
            (i, j)
            for i, row in enumerate(maze_map)
            for j, field in enumerate(row)
            if field == Field.AGENT.value
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
    def tweak_path(path: Path) -> Path:
        if random() < 0.6:
            i = randrange(len(path))
            j = randrange(len(path))
            path[i], path[j] = path[j], path[i]
            return path
        i = randrange(len(path) - 1)
        j = choice([c for c in range(len(path)) if c > i])
        return path[:i] + list(reversed(path[i:j])) + path[j:]


def shuffle_tuple(t: Tuple[Any, ...]) -> Tuple[Any, ...]:
    lst = list(t)
    shuffle(lst)
    return tuple(lst)


def crossover(parents):
    shuffled = []
    p = 1 / min(map(len, parents))
    for col in zip(*parents):
        if random() < p:
            shuffled.append(shuffle_tuple(col))
        else:
            shuffled.append(col)
    return list(list(path) for path in zip(*shuffled))


def tournament_selection(
    population: List[Vector], fitness: VectorFunc, t: int
) -> Vector:
    best = choice(population)
    for _ in range(1, t):
        new = choice(population)
        if fitness(new) < fitness(best):
            best = new
    return best


def selection_factory(fitness: VectorFunc, t: int) -> VectorSelection:
    def selection(population: List[Vector]) -> Vector:
        return tournament_selection(population, fitness, t)

    return selection


def main():
    t, n, _, s, p = map(int, input().split())
    maze = Maze([input() for _ in range(n)])
    solutions = [[Move[char].value for char in input().strip()] for _ in range(s)]

    path, cost = ga(
        popsize=p,
        fitness=maze.path_cost,
        crossover=crossover,
        mutate=maze.tweak_path,
        selection=selection_factory(maze.path_cost, 4),
        first_generation=solutions,
        timeout=t,
    )

    print(cost)
    print("".join(Move(move).name for move in path[:cost]), file=sys.stderr)


if __name__ == "__main__":
    main()
