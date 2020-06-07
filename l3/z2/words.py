import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy
from random import choice, randint, random, randrange, sample, shuffle
from typing import (
    Any,
    Callable,
    Counter as CounterType,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Tuple,
)

try:
    from .ga import ga, Vector, VectorFunc, VectorTweak, VectorSelection
except ImportError:
    from ga import ga, Vector, VectorFunc, VectorTweak, VectorSelection


def make_fitness(
    dictionary: Iterable[str], letters_scores: Dict[str, int], letters: List[str]
) -> Callable[[str], int]:
    def _fitness(word: str) -> int:
        temp_letters = deepcopy(letters)
        score = 0
        if word in dictionary:
            for c in word:
                if c in temp_letters:
                    score += letters_scores[c]
                    temp_letters.remove(c)
                else:
                    return 0
        return score

    return _fitness


def make_mutate(letters_count: Dict[str, int]):
    def mutate(word: str) -> str:
        if random() > 0.2:
            return word

        if random() > 0.1:
            lst_word = list(word) + list(sample(list(letters_count), randint(0, 2)))
            shuffle(lst_word)
            return "".join(lst_word)

        letter = choice(list(letters_count))
        index = randint(0, len(word))
        if random() > 0.7:
            word = word[:index] + letter + word[index + 1 :]
        else:
            word = word[:index] + letter + word[index:]
        return word

    return mutate


def shuffle_tuple(t: Tuple[Any, ...]) -> Tuple[Any, ...]:
    lst = list(t)
    shuffle(lst)
    return tuple(lst)


def crossover(parents):  # two point crossover
    shuffled = []
    p = 1 / min(map(len, parents))
    for word in zip(*parents):
        if random() < p:
            shuffled.append(shuffle_tuple(word))
        else:
            shuffled.append(word)

    return ["".join(word) for word in zip(*shuffled)]


def tournament_selection(
    population: List[Vector], fitness: VectorFunc, t: int
) -> Vector:
    best = choice(population)
    for _ in range(1, t):
        new = choice(population)
        if fitness(new) > fitness(best):
            best = new
    return best


def selection_factory(fitness: VectorFunc, t: int) -> VectorSelection:
    def selection(population: List[Vector]) -> Vector:
        return tournament_selection(population, fitness, t)

    return selection


def main() -> None:
    t, n, s = map(int, input().split())
    letters_scores = {}
    letters_count: DefaultDict[str, int] = defaultdict(int)
    for _ in range(n):
        c, p = input().split()
        letters_scores[c] = int(p)
        letters_count[c] += 1

    words = [input().strip() for _ in range(s)]

    with open("dict.txt", "r") as f:
        dictionary = {word.strip() for word in f}

    fitness = make_fitness(
        dictionary, letters_scores, list(Counter(letters_count).elements())
    )

    best, score = ga(
        popsize=10,
        elitesize=4,
        fitness=fitness,
        first_generation=words,
        crossover=crossover,
        mutate=make_mutate(letters_count),
        selection=selection_factory(fitness, 4),
        timeout=t,
    )

    print(score)
    print(best, file=sys.stderr)


if __name__ == "__main__":
    main()
