import math
import random
import time

from itertools import product
from typing import Callable, Tuple

Vector4D = Tuple[float, float, float, float]
VectorND = Tuple[float, ...]


def norm(xv: VectorND) -> float:
    return math.sqrt(sum(x ** 2 for x in xv))


def happy_cat(xv: Vector4D) -> float:
    return ((norm(xv) ** 2 - 4) ** 2) ** (0.125) + 0.25 * (0.5 * norm(xv) ** 2 + sum(xv)) + 0.5


def griewank(xv: Vector4D) -> float:
    return 1.0 + sum(((x ** 2) / 4000.0) for x in xv) - math.prod((math.cos(x / (i + 1)) for i, x in enumerate(xv)))


def random_vector(n: int) -> VectorND:
    return tuple(random.gauss(0, 1) for i in range(n))


def now() -> float:
    return time.time()


def tweak(xv: VectorND) -> VectorND:
    return tuple(x + random.gauss(0, 1) for x in xv)


def hill_climbing(n: int, f: Callable, timeout: float) -> VectorND:
    s = random_vector(n)
    start = time.time()
    while time.time() - start < timeout:
        r = tweak(s)
        if f(r) < f(s):
            s = r
    return (*s, f(s))


def hill_climbing_with_random_restarts(n, f, timeout):
    s = random_vector(n)
    best = s

    start = time.time()
    while now() - start < timeout:
        t = abs(random.random())

        st = time.time()
        while now() - st < t and now() - start < timeout:
            r = tweak(s)
            if f(r) < f(s):
                s = r

        if f(s) <= f(best):
            best = s

        s = random_vector(n)

    return (*best, f(best))
