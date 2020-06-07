import time
from math import log
from random import choice
from typing import Any, Callable, List, Tuple, Union

Number = Union[float, int]
Vector = Any
VectorFunc = Callable[[Vector], float]
VectorCrossover = Callable[[List[Vector]], List[Vector]]
VectorTweak = Callable[[Vector], Vector]
VectorSelection = Callable[[List[Vector]], Vector]


def now() -> float:
    return time.time()


def ga(
    popsize: int,
    fitness: VectorFunc,
    first_generation: List[Vector],
    crossover: VectorCrossover,
    mutate: VectorTweak,
    selection: VectorSelection,
    timeout: Number,
) -> Tuple[Vector, float]:
    population = first_generation
    best = choice(population)

    start = now()
    last_best = now()
    while now() - start < timeout and now() - last_best < log(timeout):
        for individual in population:
            if fitness(individual) < fitness(best):
                best = individual
                last_best = now()

        next_generation = []
        for _ in range(popsize // 2):
            parent_a = selection(population)
            parent_b = selection(population)
            child_a, child_b = crossover([parent_a, parent_b])
            next_generation.extend([mutate(child_a), mutate(child_b)])
        population = next_generation

    return best, fitness(best)
