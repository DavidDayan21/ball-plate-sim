"""Non-régression au bit près : enregistre les résultats de toutes les simulations, puis vérifie
qu'une modification du code ne les change pas (règle 5 du projet).

    python non_regression.py --enregistrer     AVANT la modification : écrit la référence
    python non_regression.py --comparer        APRÈS : liste les écarts (code de sortie 1 s'il y en a)

Cas couverts (graine 0, réglages par défaut), liste figée ici car campagne.py va évoluer :
    - toutes les simulations de la campagne de mesures du 03/11/2026 (commit 7f2fd90) ;
    - ses balayages, réduits à deux points autour de la limite (lancers à ±10° et ±20°, poussées) ;
    - la validation de la physique (roulement, glissement, rebond) ;
    - un enchaînement du bac à sable : lâcher, lancer rattrapé, lancer trop rapide.

L'identité au bit près n'a de sens que sur une même machine, avec les mêmes versions : dès que la
balle rebondit, Windows et Linux divergent. La référence enregistre donc la plateforme et les
versions, et la comparaison prévient si elles ont changé.

Référence : results/reference/reference.npz (ignorée par git, propre à chaque machine).
"""
import argparse
from dataclasses import fields, replace
import datetime
from functools import partial
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from simulation.boucle import Resultats, simuler
from simulation.config import Config, ParametresCamera, carre, centre
from simulation.parallele import executer
from simulation.physique_mujoco import acceleration_roulement, restitution_mesuree
from simulation.session import Session

FICHIER = Path(__file__).parent / "results" / "reference" / "reference.npz"
INFOS = "__infos__"          # clé réservée : date et environnement de la référence

# --- Cas de référence --------------------------------------------------------------------------
CAMERA = Config(capteur="camera", camera=ParametresCamera(largeur=160, hauteur_image=120))
PM = CAMERA.mujoco_params


def _lancer(inclinaison_deg: float, vitesse: float) -> Config:
    """Balle lancée depuis x = -10 cm vers +x, consigne au centre."""
    return replace(CAMERA, theta_max=math.radians(inclinaison_deg), trajectoire=centre, duree=5.0,
                   depart=(-0.10, 0.0), vitesse_depart=(vitesse, 0.0))


def _poussee(dv: float) -> Config:
    """Balle au repos au centre, poussée à t = 1 s."""
    return replace(CAMERA, trajectoire=centre, duree=5.0, depart=(0.0, 0.0),
                   poussees=((1.0, 0.8 * dv, 0.6 * dv),))


SIMULATIONS = {
    "maison_plateau": Config(physique="simple", mecanisme=False),
    "maison_robot": Config(physique="simple"),
    "mujoco_plateau": Config(mecanisme=False),
    "mujoco_robot": Config(),
    "camera": CAMERA,
    "lancer_10deg_0.5": _lancer(10, 0.5),
    "lancer_10deg_0.6": _lancer(10, 0.6),
    "lancer_20deg_0.6": _lancer(20, 0.6),
    "lancer_20deg_0.7": _lancer(20, 0.7),
    "poussee_0.4": _poussee(0.4),
    "poussee_0.5": _poussee(0.5),
    "bille_pleine": replace(CAMERA, mujoco_params=replace(PM, k_reel=2 / 5)),
    "balle_lourde": replace(CAMERA, mujoco_params=replace(PM, masse_balle=0.027)),
    "frottement_0.2": replace(CAMERA, mujoco_params=replace(PM, frottement=0.2)),
    "frottement_0.02": replace(CAMERA, mujoco_params=replace(PM, frottement=0.02)),
    "plateau_lourd": replace(CAMERA, mujoco_params=replace(PM, masse_plateau=0.5)),
    "latence_20ms_compensee": replace(CAMERA, compensation_retard=True),
    "latence_60ms": replace(CAMERA, retard=3),
    "latence_60ms_compensee": replace(CAMERA, retard=3, compensation_retard=True),
    "latence_100ms": replace(CAMERA, retard=5),
    "latence_100ms_compensee": replace(CAMERA, retard=5, compensation_retard=True),
    "carre": replace(CAMERA, trajectoire=carre, duree=16.0, depart=(0.05, 0.05)),
    "carre_compense": replace(CAMERA, trajectoire=carre, duree=16.0, depart=(0.05, 0.05),
                              compensation_retard=True),
}


def extraire(r: Resultats) -> dict:
    """Tous les champs de r en tableaux numpy (les historiques en float64)."""
    tableaux = {}
    for f in fields(Resultats):
        valeur = getattr(r, f.name)
        tableaux[f.name] = np.asarray(valeur, dtype=float) if isinstance(valeur, list) else np.asarray(valeur)
    return tableaux


def resultats_simulation(cfg: Config) -> dict:
    return extraire(simuler(cfg))


def _physique() -> dict:
    """Validation de la physique : section 1 de la campagne."""
    k = CAMERA.plateau.k_inertie
    theta = math.radians(10)
    mu_min = k * math.tan(theta) / (1 + k)
    return {"roulement": np.asarray(acceleration_roulement(k, theta, frottement=1.3 * mu_min)),
            "glissement": np.asarray(acceleration_roulement(k, theta, frottement=0.7 * mu_min)),
            "restitution": np.asarray(restitution_mesuree())}


def _session() -> dict:
    """Bac à sable : lâcher de 30,5 cm au centre, lancer rattrapé, lancer trop rapide (chute)."""
    s = Session(enregistrer=False)
    instantanes = []

    def avancer(duree: float) -> None:
        for _ in range(round(duree / s.dt)):
            s.pas()
        instantanes.append(np.asarray(list(s.courbes), dtype=float))   # 10 dernières secondes

    avancer(1.0)
    s.lancer(0.0, 0.0, hauteur=0.305)
    avancer(6.0)
    s.lancer(0.10, 180, vitesse=0.4)
    avancer(4.0)
    s.lancer(0.10, 180, vitesse=1.0)
    avancer(3.0)
    resume = s.fermer()
    # repr d'un nombre à virgule flottante est exact ; l'heure de la tentative est exclue
    return {"courbes": np.concatenate(instantanes),
            "resume": np.asarray([repr({k: v for k, v in ligne.items() if k != "heure"}) for ligne in resume])}


CAS = {**{nom: partial(resultats_simulation, cfg) for nom, cfg in SIMULATIONS.items()},
       "physique": _physique, "session": _session}


def calculer(nom: str) -> dict:
    """Résultats du cas `nom` : {champ: tableau}."""
    return CAS[nom]()


def calculer_tout(processus: int = None) -> dict:
    """Résultats de tous les cas : {"cas/champ": tableau}."""
    noms = list(CAS)
    tableaux = {}
    for nom, resultat in zip(noms, executer(calculer, noms, processus)):
        tableaux.update({f"{nom}/{champ}": valeur for champ, valeur in resultat.items()})
    return tableaux


# --- Enregistrement et comparaison ---------------------------------------------------------------
def environnement() -> dict:
    """Ce dont dépend l'identité au bit près : plateforme et versions."""
    import cv2
    import mujoco
    return {"plateforme": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "numpy": np.__version__,
            "mujoco": mujoco.__version__, "opencv": cv2.__version__}


def enregistrer(tableaux: dict, fichier: Path = FICHIER) -> None:
    fichier.parent.mkdir(parents=True, exist_ok=True)
    infos = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "environnement": environnement()}
    np.savez_compressed(fichier, **tableaux, **{INFOS: np.asarray(json.dumps(infos, ensure_ascii=False))})


def charger(fichier: Path = FICHIER):
    """Renvoie (tableaux, infos) d'une référence enregistrée."""
    with np.load(fichier) as donnees:
        tableaux = {cle: donnees[cle] for cle in donnees.files if cle != INFOS}
        infos = json.loads(str(donnees[INFOS]))
    return tableaux, infos


def _ecart(a: np.ndarray, b: np.ndarray) -> str:
    """Description de la différence entre deux tableaux ; chaîne vide s'ils sont identiques au bit près."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return f"type ou forme différents ({a.dtype} {a.shape} -> {b.dtype} {b.shape})"
    if a.tobytes() == b.tobytes():
        return ""
    # Comparaison élément par élément, sur les octets (0,0 et -0,0 diffèrent, NaN est comparable)
    octets_a = np.ascontiguousarray(a).reshape(-1).view(np.uint8).reshape(a.size, a.dtype.itemsize)
    octets_b = np.ascontiguousarray(b).reshape(-1).view(np.uint8).reshape(b.size, b.dtype.itemsize)
    differents = (octets_a != octets_b).any(axis=1)
    premier = np.unravel_index(int(np.argmax(differents)), a.shape) if a.ndim else ()
    texte = (f"{int(differents.sum())} valeur(s) différente(s) sur {a.size}, "
             f"la première à l'indice {tuple(int(i) for i in premier)}")
    if np.issubdtype(a.dtype, np.number) or a.dtype == bool:
        d = np.abs(a.astype(float).reshape(-1) - b.astype(float).reshape(-1))[differents]
        texte += f", écart max {np.max(d):.3g}"
    return texte


def comparer(reference: dict, actuel: dict) -> list:
    """Écarts au bit près entre deux jeux {clé: tableau}. Liste vide : identiques."""
    ecarts = []
    for cle in sorted(reference.keys() | actuel.keys()):
        if cle not in actuel:
            ecarts.append(f"{cle} : absent des résultats actuels")
        elif cle not in reference:
            ecarts.append(f"{cle} : nouveau, absent de la référence")
        elif texte := _ecart(reference[cle], actuel[cle]):
            ecarts.append(f"{cle} : {texte}")
    return ecarts


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-régression au bit près de toutes les simulations")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--enregistrer", action="store_true", help="écrit la référence (avant une modification)")
    action.add_argument("--comparer", action="store_true", help="compare à la référence (après la modification)")
    parser.add_argument("--remplacer", action="store_true", help="avec --enregistrer : écrase la référence existante")
    parser.add_argument("--processus", type=int, default=None, help="nombre de processus (défaut : un par cœur)")
    args = parser.parse_args()

    if args.enregistrer and FICHIER.exists() and not args.remplacer:
        sys.exit(f"Une référence existe déjà ({FICHIER}). Ajoute --remplacer pour l'écraser.")
    if args.comparer and not FICHIER.exists():
        sys.exit(f"Aucune référence ({FICHIER}) : lance d'abord --enregistrer.")

    debut = time.perf_counter()
    print(f"Calcul de {len(CAS)} cas...")
    actuel = calculer_tout(args.processus)
    print(f"Terminé en {time.perf_counter() - debut:.0f} s.")

    if args.enregistrer:
        enregistrer(actuel)
        print(f"Référence écrite : {FICHIER}")
        return

    reference, infos = charger()
    print(f"Référence du {infos['date']}.")
    env = environnement()
    for cle, valeur in env.items():
        if infos["environnement"].get(cle) != valeur:
            print(f"Attention, {cle} différent : {infos['environnement'].get(cle)} -> {valeur} "
                  "(l'identité au bit près n'est plus garantie).")
    ecarts = comparer(reference, actuel)
    if not ecarts:
        print(f"Identique au bit près : {len(CAS)} cas sur {len(CAS)}.")
        return
    cas = sorted({e.split("/")[0] for e in ecarts})
    print(f"ÉCARTS dans {len(cas)} cas sur {len(CAS)} ({', '.join(cas)}) :")
    for e in ecarts:
        print(f"  {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
