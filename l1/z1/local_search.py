import math
import random
import time
from typing import Callable, Tuple

Vector = Tuple[float, ...]


def norm(xv: Vector) -> float:
    return math.sqrt(sum(x ** 2 for x in xv))


def happy_cat(xv: Vector) -> float:
    return ((norm(xv) ** 2 - 4) ** 2) ** (0.125) + 0.25 * (0.5 * norm(xv) ** 2 + sum(xv)) + 0.5


def griewank(xv: Vector) -> float:
    return 1.0 + sum(((x ** 2) / 4000.0) for x in xv) - math.prod((math.cos(x / (i + 1)) for i, x in enumerate(xv)))


def random_vector(n: int) -> Vector:
    return tuple(random.gauss(0, 0.001) for i in range(n))


def now() -> float:
    return time.time()


def tweak(xv: Vector) -> Vector:
    return tuple(x + random.gauss(0, 0.0001) for x in xv)


def large_tweak(xv: Vector) -> Vector:
    return tuple(x + random.gauss(0, 0.000001) for x in xv)


def iterated_local_search(n: int, f: Callable[[Vector], float], timeout: float) -> Vector:
    s = random_vector(n)
    best = s
    h = s
    start = time.time()
    while now() - start < timeout:
        t = 0.01

        st = time.time()
        while now() - st < t and now() - start < timeout:
            r = tweak(s)
            if f(r) < f(s):
                s = r

        if f(s) <= f(best):
            best = s

        if f(s) <= f(h):
            h = s

        s = large_tweak(h)

    return (*best, f(best))
