import time
from math import exp
from random import random
from typing import Callable, Tuple, TypeVar


def now() -> float:
    return time.time()


T = TypeVar("T")


def simulated_annealing(
    function: Callable[[T], float],
    initial_solution: T,
    initial_temperature: float,
    cooling_schedule: Callable[[float], float],
    tweak: Callable[[T], T],
    timeout: float,
) -> Tuple[T, float]:
    """
    Returns a minimum of a given function via simulated annealing.

    Parameters:
        function - function to minimize
        initiail_solution - initial solution for a given function
        initial_temperature - initial temperature of simulated annealing. Should be high.
        cooling_schedule - function for calculating the next temperature
        tweak - solution tweak function
        timeout - algorithm timeout in seconds

    Returns:
        Best solution found in a given time, function value in that solution
    """
    temperature = initial_temperature
    current_solution = initial_solution
    best_solution = current_solution

    current_result = function(current_solution)
    best_result = current_result

    start = now()
    while now() - start <= timeout:
        tweaked_solution = tweak(current_solution)
        tweaked_result = function(tweaked_solution)

        if tweaked_result < current_result or (
            random() < exp((current_result - tweaked_result) / temperature)
        ):
            current_solution = tweaked_solution
            current_result = tweaked_result

            if current_result < best_result:
                best_solution = current_solution
                best_result = current_result

        temperature = cooling_schedule(temperature)

    return best_solution, best_result
