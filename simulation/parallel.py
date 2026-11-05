"""Running independent computations (simulations) on several processes.

Each simulation is deterministic and shares nothing with the others: running them in
separate processes gives exactly the same results as running them one after the other (checked in
tests/test_regression_check.py).

Each process restarts Python (the "spawn" method, the one used on Windows, also enforced on Linux
for identical behavior): the calling script must protect its main code with
`if __name__ == "__main__":`, and the function passed must be defined at module level
(no lambda).
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os


def execute(function, arguments, processes: int = None) -> list:
    """[function(a) for a in arguments], in the same order, spread over `processes` processes.

    processes = None: one per core; processes = 1: everything in the current process.
    """
    arguments = list(arguments)
    processes = min(processes or os.cpu_count() or 1, len(arguments))
    if processes <= 1:
        return [function(a) for a in arguments]
    with ProcessPoolExecutor(max_workers=processes, mp_context=multiprocessing.get_context("spawn")) as executor:
        return list(executor.map(function, arguments))
