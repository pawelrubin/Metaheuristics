import math
import random
import sys
import time
from collections import deque
from copy import deepcopy
from typing import Any, Callable, Deque, List, Tuple


def now():
    return time.time()


class TabuList:
    def __init__(self, max_size: int):
        self._list: Deque[Any] = deque([], max_size)

    def push(self, element: Any):
        self._list.append(element)

    def __str__(self):
        return str(list(self._list))

    def contains(self, element: Any):
        return element in self._list


def random_path(n: int) -> List[int]:
    result: List[int] = []
    for _ in range(n - 2):
        result.append(random.choice(list(set(range(1, n - 1)) - set(result))))
    return [0] + result + [0]


def greedy_path(costs: List[List[int]]) -> List[int]:
    path = [0]
    curr_city = 0
    while len(path) < len(costs):
        min_cost = math.inf
        min_index = 0

        for i, _ in enumerate(costs[curr_city]):
            if costs[curr_city][i] < min_cost and i not in path:
                min_cost = costs[curr_city][i]
                min_index = i

        path.append(min_index)
        curr_city = min_index
    return path + [0]


def tweak_path(path: List[int]) -> List[int]:
    if random.random() < 0.7:
        city1 = random.randrange(1, len(path) - 1)
        city2 = random.choice([c for c in range(1, len(path) - 1) if c != city1])
        path[city1], path[city2] = path[city2], path[city1]
        return path
    city1 = random.randrange(1, len(path) - 2)
    city2 = random.choice([c for c in range(1, len(path) - 1) if c > city1])
    return path[:city1] + list(reversed(path[city1:city2])) + path[city2:]


def path_cost(costs: List[List[int]]) -> Callable[[List[int]], int]:
    return lambda path: sum(costs[path[i]][path[i + 1]] for i in range(len(path) - 1))


def tabu_search(
    initial: List[int],
    tweak: Callable[[List[int]], List[int]],
    quality: Callable[[List[int]], int],
    tabu_size: int,
    num_of_tweaks: int,
    timeout: float,
) -> Tuple[List[int], int]:
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
    t, n = map(int, input().split())
    costs = [[*map(int, input().split())] for i in range(n)]

    path, cost = tabu_search(
        initial=greedy_path(costs),
        tweak=tweak_path,
        quality=path_cost(costs),
        tabu_size=n * 10,
        num_of_tweaks=int(n ** 2 / 3),
        timeout=float(t),
    )

    print(*list(map(lambda city: city + 1, path)), file=sys.stderr)
    print(cost)


if __name__ == "__main__":
    main()
