"""Exécution de calculs indépendants (des simulations) sur plusieurs processus.

Chaque simulation est déterministe et ne partage rien avec les autres : les lancer dans des
processus séparés donne exactement les mêmes résultats qu'à la suite (vérifié dans
tests/test_regression_check.py).

Chaque processus redémarre Python (méthode « spawn », celle de Windows, imposée aussi sous Linux
pour un comportement identique) : le script appelant doit protéger son code principal par
`if __name__ == "__main__":`, et la fonction passée doit être définie au niveau d'un module
(pas de lambda).
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os


def executer(fonction, arguments, processus: int = None) -> list:
    """[fonction(a) for a in arguments], dans le même ordre, réparti sur `processus` processus.

    processus = None : un par cœur ; processus = 1 : tout dans le processus courant.
    """
    arguments = list(arguments)
    processus = min(processus or os.cpu_count() or 1, len(arguments))
    if processus <= 1:
        return [fonction(a) for a in arguments]
    with ProcessPoolExecutor(max_workers=processus, mp_context=multiprocessing.get_context("spawn")) as executeur:
        return list(executeur.map(fonction, arguments))
