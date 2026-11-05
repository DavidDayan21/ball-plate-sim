"""Session du bac à sable, sans affichage : actions, fin des tentatives, fichiers écrits."""
import csv
import math

import pytest

from simulation.session import Session


def avancer(s: Session, duree: float) -> None:
    for _ in range(round(duree / s.dt)):
        s.pas()


@pytest.fixture
def session(tmp_path):
    s = Session(dossier=tmp_path)
    avancer(s, 1.0)
    return s


def test_balle_lachee_rebondit_puis_est_stabilisee(session):
    session.lancer(0.0, 0.0, hauteur=0.305)                   # lâcher de 30,5 cm au centre
    avancer(session, 15.0)
    l = session.resume[-1]
    assert l["action"] == "lâcher" and l["resultat"] == "stabilisée"
    assert l["atterrissages"] >= 5
    # Premier rebond : un peu sous e² · h = 0,88² × 305 mm (le plateau, monté sur moteurs, amortit)
    assert 150 < l["rebond_max_mm"] < 0.91 ** 2 * 305


def test_lancer_rattrape_puis_trop_rapide_tombe(session):
    session.lancer(0.10, 180, vitesse=0.4)
    avancer(session, 15.0)
    assert session.resume[-1]["resultat"] == "stabilisée"
    session.lancer(0.10, 180, vitesse=1.0)
    avancer(session, 3.0)
    assert session.resume[-1]["resultat"] == "tombée" and session.etat == "tombée"
    avancer(session, 2.0)                                     # la simulation continue sans balle
    session.centrer()                                         # et repart
    avancer(session, 5.0)
    assert session.resume[-1]["action"] == "centre" and session.resume[-1]["resultat"] == "stabilisée"


def test_perturbation_detectee_automatiquement(session):
    session.centrer()
    avancer(session, 5.0)
    assert session.etat == "attente"
    session.phys.pousser(0.3, 0.0)                            # comme une poussée à la souris
    avancer(session, 1.0)
    assert session.etat == "tentative" and session.tentative["action"] == "perturbation"
    avancer(session, 10.0)
    assert session.resume[-1]["action"] == "perturbation"
    assert session.resume[-1]["resultat"] == "stabilisée"


def test_fichiers_ecrits(session):
    session.lancer(0.10, 180, vitesse=0.4)
    avancer(session, 2.0)
    session.changer_consigne("cercle")
    session.tentative_libre()                                 # interrompt le lancer
    avancer(session, 2.0)
    resume = session.fermer()
    assert [l["resultat"] for l in resume] == ["interrompue", "interrompue"]

    with open(session.dossier / "summary.csv", encoding="utf-8-sig") as f:
        lignes = list(csv.DictReader(f, delimiter=";"))
    assert len(lignes) == 2 and lignes[0]["action"] == "lancer" and lignes[1]["consigne"] == "cercle"
    assert lignes[0]["vitesse_m_s"] == "0,400"                # virgule décimale (Excel en français)

    with open(session.dossier / "attempt_002.csv", encoding="utf-8-sig") as f:
        mesures = list(csv.DictReader(f, delimiter=";"))
    nombre = lambda texte: float(texte.replace(",", "."))
    assert len(mesures) == 100                                # 2 s à 50 Hz
    # La consigne en cercle commence à son point de départ : 6 cm à 45°
    x, y = nombre(mesures[0]["x_consigne_mm"]), nombre(mesures[0]["y_consigne_mm"])
    assert math.isclose(x, 60 * math.cos(math.pi / 4), abs_tol=0.01) and math.isclose(y, x, abs_tol=0.01)
