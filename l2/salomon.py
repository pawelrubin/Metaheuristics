from math import cos, pi, sqrt
from random import gauss
from typing import Tuple

import metaheuristics as meta

Vector = Tuple[float, ...]


def salomon(xs: Vector) -> float:
    sqrt_sum = sqrt(sum(x ** 2 for x in xs))
    return 1 - cos(2 * pi * sqrt_sum) + 0.1 * sqrt_sum


def tweak(xs: Vector) -> Vector:
    return tuple(x * gauss(1, 0.1) for x in xs)


def main():
    t, x1, x2, x3, x4 = map(float, input().split())

    best_solution, best_result = meta.simulated_annealing(
        function=salomon,
        initial_solution=(x1, x2, x3, x4),
        initial_temperature=10 ** 6,
        cooling_schedule=lambda t, _: t * 0.9999,
        timeout=t,
        tweak=tweak,
    )

    print(*best_solution, best_result)


if __name__ == "__main__":
    main()
