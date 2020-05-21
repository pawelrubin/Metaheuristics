import time
from math import log
from random import choice, gauss, random, shuffle
from typing import Any, Callable, List, Tuple, Union


Number = Union[float, int]
Vector = Tuple[float, ...]
VectorFunc = Callable[[Vector], float]
VectorCrossover = Callable[[List[Vector]], List[Vector]]
VectorTweak = Callable[[Vector], Vector]
VectorSelection = Callable[[List[Vector]], Vector]


def now() -> float:
    return time.time()


def yang(xs: Vector, eps: Vector) -> float:
    """
    Xin-She Yang Function.
    eps vector components should be drawn uniformly from [0, 1].
    """
    return sum(e * abs(x) ** (i + 1) for i, (e, x) in enumerate(zip(eps, xs)))


def make_yang(eps: Vector) -> VectorFunc:
    def _yang(xs: Vector) -> float:
        return yang(xs, eps)

    return _yang


def shuffle_tuple(t: Tuple[Any, ...]) -> Tuple[Any, ...]:
    lst = list(t)
    shuffle(lst)
    return tuple(lst)


def uniform_crossover(vectors: List[Vector]) -> List[Vector]:
    shuffled = []
    p = 1 / len(vectors)
    for col in zip(*vectors):  # for column in vector matrix
        if random() < p:
            shuffled.append(shuffle_tuple(col))
        else:
            shuffled.append(col)
    return list(zip(*shuffled))


def tournament_selection(
    population: List[Vector], fitness: VectorFunc, t: int
) -> Vector:
    best = choice(population)
    for _ in range(1, t):
        new = choice(population)
        if fitness(new) < fitness(best):
            best = new
    return best


def tweak_factory(sigma: float) -> VectorTweak:
    return lambda xs: tuple(x * gauss(1, sigma) for x in xs)


def selection_factory(fitness: VectorFunc, t: int) -> VectorSelection:
    def selection(population: List[Vector]):
        return tournament_selection(population, fitness, t)

    return selection



def ga(
    popsize: int,
    fitness: VectorFunc,
    initial: Vector,
    crossover: VectorCrossover,
    mutate: VectorTweak,
    selection: VectorSelection,
    timeout: Number,
) -> Tuple[Vector, float]:
    population = [mutate(initial) for _ in range(popsize)]
    best = choice(population)

    start = now()
    last_best = start
    while now() - start < timeout and now() - last_best < log(timeout):
        for individual in population:
            if fitness(individual) < fitness(best):
                best = individual
                last_best = now()

        next_generation: List[Vector] = []
        for _ in range(popsize // 2):
            parent_a = selection(population)
            parent_b = selection(population)
            while parent_b is parent_a:
                parent_b = selection(population)
            child_a, child_b = crossover([parent_a, parent_b])
            next_generation.extend([mutate(child_a), mutate(child_b)])
        population = next_generation

    return best, fitness(best)


def main():
    _input = input().split()
    t = int(_input[0])
    xs = tuple(map(int, _input[1:6]))
    eps = tuple(map(float, _input[6:]))

    fitness = make_yang(eps)

    best, value = ga(
        popsize=10,
        fitness=fitness,
        initial=xs,
        crossover=uniform_crossover,
        mutate=tweak_factory(sigma=0.1),
        selection=selection_factory(fitness, t=4),
        timeout=t,
    )

    print(*best, value)


if __name__ == "__main__":
    main()
