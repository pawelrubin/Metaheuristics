import math
import random
import time
from collections import deque
from typing import Any, Callable, Deque, List


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


def random_path(n: int):
    result: List[int] = list()
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
    return path + [0]


def tweak_path(path):
    city1 = random.choice(list(set(range(len(path))) - {0, len(path) - 1}))
    city2 = random.choice(list(set(range(len(path))) - {0, len(path) - 1, city1}))
    path[city1], path[city2] = path[city1], path[city2]
    return path


def path_cost(costs):
    return lambda path: sum(costs[path[i]][path[i + 1]] for i in range(len(path) - 1))


def tabu_search(initial, tweak, quality, l, n, timeout):
    s = initial
    best = s
    tabu = TabuList(l)
    tabu.push(s)

    start = now()
    while now() - start <= timeout:
        r = tweak(s)
        for _ in range(n):
            w = tweak(s)
            if not tabu.contains(w) and (tabu.contains(r) or quality(w) < quality(r)):
                r = w
        if not tabu.contains(r):
            s = r
            tabu.push(r)
        if quality(s) < quality(best):
            best = s

    return best, quality(best)


def main():
    t, n = map(int, input().split())
    costs = [[*map(int, input().split())] for i in range(n)]

    path, cost = tabu_search(
        initial=greedy_path(costs),
        tweak=tweak_path,
        quality=path_cost(costs),
        l=1000,
        n=10000,
        timeout=float(t),
    )

    print(*list(map(lambda city: city + 1, path)))
    print(cost)


if __name__ == "__main__":
    main()
