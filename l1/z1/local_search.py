import math
import random
import time
from functools import reduce
from typing import Callable, Tuple

Vector = Tuple[float, ...]


def prod(iterable):
    return reduce(lambda a, b: a * b, iterable)


def norm(xv: Vector) -> float:
    return math.sqrt(sum(x ** 2 for x in xv))


def happy_cat(xv: Vector) -> float:
    return (
        ((norm(xv) ** 2 - 4) ** 2) ** (0.125)
        + 0.25 * (0.5 * norm(xv) ** 2 + sum(xv))
        + 0.5
    )


def griewank(xv: Vector) -> float:
    return (
        1.0
        + sum(((x ** 2) / 4000.0) for x in xv)
        - prod((math.cos(x / math.sqrt(i + 1)) for i, x in enumerate(xv)))
    )


def random_vector(n: int) -> Vector:
    return tuple(random.gauss(0, 0.001) for i in range(n))


def now() -> float:
    return time.time()


def tweak_factory(sigma: float) -> Callable[[Vector], Vector]:
    return lambda xv: tuple(x + random.gauss(0, sigma) for x in xv)


def iterated_local_search(
    tweak: Callable[[Vector], Vector],
    large_tweak: Callable[[Vector], Vector],
    initial: Vector,
    quality: Callable[[Vector], float],
    timeout: float,
) -> Vector:
    s = initial
    best = s
    homebase = s
    start = now()
    while now() - start < timeout:
        t = 0.01

        st = now()
        while now() - st < t and now() - start < timeout:
            r = tweak(s)
            if quality(r) < quality(s):
                s = r

        if quality(s) <= quality(best):
            best = s

        if quality(s) <= quality(homebase):
            homebase = s

        s = large_tweak(homebase)

    return (*best, quality(best))


def main():
    t, b = input().split()
    if b == "0":
        result = iterated_local_search(
            tweak=tweak_factory(0.0001),
            large_tweak=tweak_factory(0.00001),
            initial=random_vector(4),
            quality=happy_cat,
            timeout=float(t),
        )
    else:
        result = iterated_local_search(
            tweak=tweak_factory(0.000001),
            large_tweak=tweak_factory(0.00000001),
            initial=random_vector(4),
            quality=griewank,
            timeout=float(t),
        )

    print(" ".join(map(str, result)))


if __name__ == "__main__":
    main()
