"""Session du bac à sable : une suite de tentatives sur un même robot MuJoCo, qui ne s'arrête jamais.

Une tentative commence par une action (lancer, lâcher, centre, tentative libre) ou, en attente,
dès que la balle s'écarte de plus de 2 cm de la consigne (poussée à la souris par exemple).
Quand la balle est déplacée (lancer, lâcher, centre), le capteur et le contrôleur repartent de zéro
(intégrale, prédicteur, latence) : les tentatives sont indépendantes les unes des autres.
Elle se termine quand :
    - la balle tombe ;
    - la balle reste à moins de 1 cm de la consigne, posée sur le plateau, pendant 2 s (stabilisée) ;
    - elle dure plus de duree_max (non stabilisée) ;
    - une autre tentative commence (interrompue).

Données écrites dans results/tentatives/<date_heure>/ :
    tentative_001.csv ...  toutes les mesures de la tentative, à 50 Hz
    resume.csv             une ligne par tentative (paramètres et résultat)
Format adapté à Excel en français : séparateur « ; », virgule décimale.
"""
from collections import deque
from dataclasses import fields, replace
import csv
import datetime
import math
from pathlib import Path

import numpy as np

from simulation.boucle import Resultats, Simulation
from simulation.config import Config, carre, centre, cercle

CONSIGNES = {"centre": centre, "cercle": cercle, "carré": carre}
SEUIL_STABLE = 0.01          # (m) distance à la consigne
DUREE_STABLE = 2.0           # (s)
SEUIL_PERTURBATION = 0.02    # (m) en attente : au-delà, une tentative démarre toute seule
AVANT_PERTURBATION = 0.5     # (s) de mesures conservées avant le déclenchement
VOL_MIN = 0.010              # (s) un vol plus court n'est pas compté comme un atterrissage
POSEE = 0.001                # (m) balle considérée posée sous cette hauteur


def config_bac_a_sable() -> Config:
    """Robot complet, balle au repos au centre, consigne au centre."""
    return Config(trajectoire=centre, depart=(0.0, 0.0), duree=math.inf)


def _decaler(fonction, t0: float):
    """Consigne qui démarre à t0 (un cercle commence à son point de départ)."""
    return lambda t: fonction(t - t0)


def _garder_fin(r: Resultats, n: int) -> Resultats:
    """Copie des n derniers échantillons de r."""
    copie = Resultats()
    for f in fields(Resultats):
        valeur = getattr(r, f.name)
        if isinstance(valeur, list):
            setattr(copie, f.name, valeur[-n:] if n > 0 else [])
    return copie


def _nombre(v: float, decimales: int = 3) -> str:
    """Nombre au format français (virgule décimale)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{decimales}f}".replace(".", ",")


class Session:
    def __init__(self, cfg: Config = None, dossier="results/tentatives", duree_max: float = 30.0,
                 sol: bool = True, enregistrer: bool = True):
        self.cfg = cfg or config_bac_a_sable()
        if self.cfg.physique != "mujoco":
            raise ValueError("Le bac à sable utilise la physique MuJoCo")
        self.sim = Simulation(self.cfg)
        self.phys = self.sim.phys
        self.dt = self.cfg.dt_physique
        self.ppc = self.sim.pas_par_controle
        self.duree_max = duree_max
        self.consigne = "centre"
        self.t_consigne = 0.0
        self._appliquer_config()
        if sol:
            self._activer_sol()
        # Enregistrement
        self.enregistrer = enregistrer
        self.dossier = Path(dossier) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.resume = []                   # une ligne (dict) par tentative terminée
        self.numero = 0
        self.tentative = None              # dict de la tentative en cours, None en attente
        self.etat = "attente"              # "tentative", "attente" ou "tombée"
        self.message = "Prêt."
        # Affichage en direct, 10 s d'historique : (t, distance à la consigne, θx, θy, hauteur, x, y,
        # x_consigne, y_consigne)
        self.courbes = deque(maxlen=round(10.0 / self.cfg.periode_controle))
        # Suivi pas à pas (atterrissages)
        self._pas_en_vol = 0

    # --- Réglages ----------------------------------------------------------------------------
    def _activer_sol(self) -> None:
        """Le sol arrête la balle tombée (sans lui, elle tomberait indéfiniment).
        Bit de collision 2 : le sol ne touche que la balle, jamais le plateau."""
        m = self.phys.model
        sol, balle = m.geom("sol").id, m.geom("balle").id
        m.geom_contype[sol] = 2
        m.geom_conaffinity[sol] = 2
        m.geom_conaffinity[balle] |= 2

    def _appliquer_config(self, **changements) -> None:
        self.cfg = replace(self.cfg, **changements)
        self.sim.cfg = replace(self.cfg, trajectoire=_decaler(CONSIGNES[self.consigne], self.t_consigne))

    def changer_consigne(self, nom: str) -> None:
        self.consigne, self.t_consigne = nom, self.sim.t
        self._appliquer_config()

    def changer_latence(self, periodes: int) -> None:
        """Latence de mesure en périodes de contrôle (1 = 20 ms). Remet le contrôleur à zéro."""
        self._appliquer_config(retard=periodes)
        self.sim.reinitialiser_controle()

    def changer_compensation(self, active: bool) -> None:
        self._appliquer_config(compensation_retard=active)
        self.sim.reinitialiser_controle()

    def activer_camera(self) -> None:
        """Mesure par la caméra virtuelle, à appeler depuis le fil qui fait avancer la simulation."""
        if self.sim.camera is None:
            self.sim.creer_camera()
            self.sim.reinitialiser_controle()

    def fermer_camera(self) -> None:
        """Libère la caméra (depuis le fil qui l'a créée) ; retour au capteur fictif."""
        if self.sim.camera is not None:
            self.sim.camera.fermer()
            self.sim.camera = None
            self.sim.reinitialiser_controle()

    @property
    def capteur(self) -> str:
        return "caméra" if self.sim.camera is not None else "fictif"

    @property
    def t(self) -> float:
        return self.sim.t

    # --- Actions -----------------------------------------------------------------------------
    def lancer(self, distance: float, angle_deg: float, vitesse: float = 0.0, hauteur: float = 0.0,
               deviation_deg: float = 0.0, action: str = None) -> None:
        """Place la balle au point de départ (distance, angle) du centre, à `hauteur` au-dessus du
        plateau, avec une vitesse horizontale dirigée vers le centre, tournée de `deviation_deg`.
        vitesse = 0 : simple lâcher. hauteur = 0 : balle lancée en roulant."""
        a = math.radians(angle_deg)
        x, y = distance * math.cos(a), distance * math.sin(a)
        visee = a + math.pi + math.radians(deviation_deg)
        vx, vy = vitesse * math.cos(visee), vitesse * math.sin(visee)
        action = action or ("lâcher" if vitesse == 0 else "lancer")
        self._nouvelle_tentative(action, dict(distance_cm=100 * distance, angle_deg=angle_deg,
                                              vitesse_m_s=vitesse, hauteur_cm=100 * hauteur,
                                              deviation_deg=deviation_deg), reinitialiser=True)
        self.phys.placer_balle(x, y, hauteur, vx, vy, 0.0)
        self.phys.lire_decolles_periode()          # remet le compteur de la période à zéro
        self.sim.derniere_mesure = (x, y)

    def centrer(self) -> None:
        """Balle posée immobile au centre."""
        self.lancer(0.0, 0.0, action="centre")

    def tentative_libre(self) -> None:
        """Enregistre sans toucher à la balle (pour la pousser à la souris, changer de consigne...)."""
        if self.etat == "tombée":
            self.message = "La balle est tombée : utilise Lancer, Lâcher ou Centre."
            return
        self._nouvelle_tentative("libre", {})

    # --- Tentatives --------------------------------------------------------------------------
    def _nouvelle_tentative(self, action: str, parametres: dict, avant: Resultats = None,
                            reinitialiser: bool = False) -> None:
        if self.tentative is not None:
            self._terminer("interrompue")
        self.numero += 1
        if reinitialiser:
            self.sim.reinitialiser_controle()
        self.sim.r = avant if avant is not None else Resultats()
        self.tentative = dict(numero=self.numero, action=action, parametres=parametres, t0=self.t,
                              heure=datetime.datetime.now().strftime("%H:%M:%S"),
                              consigne=self.consigne, capteur=self.capteur,
                              latence_ms=round(1000 * self.cfg.retard * self.cfg.periode_controle),
                              compensation=self.cfg.compensation_retard,
                              debut_stable=None, atterrissages=0, rebond_max=0.0)
        self.etat = "tentative"
        self._pas_en_vol = 0
        self.message = f"Tentative {self.numero} ({action}) en cours."

    def _terminer(self, resultat: str) -> dict:
        tv = self.tentative
        r = self.sim.r
        t0 = tv["t0"]
        apres = [i for i, t in enumerate(r.t) if t >= t0]
        dist = [math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) for i in apres]
        bord = [math.hypot(r.x[i], r.y[i]) for i in apres]
        incl = [max(abs(r.theta_x[i]), abs(r.theta_y[i])) for i in apres]
        couples = [max(abs(c) for c in r.couples[i]) for i in apres if r.couples and r.couples[i]]
        decollee = [r.decollee[i] for i in apres] if r.decollee else []
        ligne = dict(
            numero=tv["numero"], heure=tv["heure"], action=tv["action"], **tv["parametres"],
            consigne=tv["consigne"], capteur=tv["capteur"], latence_ms=tv["latence_ms"],
            compensation=tv["compensation"],
            resultat=resultat, duree_s=self.t - t0,
            stabilisee_en_s=(tv["debut_stable"] - t0) if resultat == "stabilisée" else None,
            ecart_max_mm=1000 * max(dist) if dist else None,
            marge_bord_min_mm=1000 * (self.cfg.plateau.rayon_plateau - max(bord)) if bord else None,
            atterrissages=tv["atterrissages"], rebond_max_mm=1000 * tv["rebond_max"],
            inclinaison_max_deg=math.degrees(max(incl)) if incl else None,
            couple_max_Nm=float(max(couples)) if couples else None,
            decollee_pct=100 * float(np.mean(decollee)) if decollee else None,
            fichier=f"tentative_{tv['numero']:03d}.csv")
        self.resume.append(ligne)
        if self.enregistrer:
            self._ecrire(r, ligne)
        self.tentative = None
        self.etat = "tombée" if resultat == "tombée" else "attente"
        self.message = self._decrire(ligne)
        self.sim.r = Resultats()
        return ligne

    @staticmethod
    def _decrire(l: dict) -> str:
        texte = f"Tentative {l['numero']} ({l['action']}) : {l['resultat']}"
        if l["stabilisee_en_s"] is not None:
            texte += f" en {l['stabilisee_en_s']:.2f} s"
        else:
            texte += f" après {l['duree_s']:.2f} s"
        if l["atterrissages"]:
            texte += f", {l['atterrissages']} atterrissage(s), rebond max {l['rebond_max_mm']:.0f} mm"
        if l["marge_bord_min_mm"] is not None and l["resultat"] != "tombée":
            texte += f", passée à {l['marge_bord_min_mm']:.0f} mm du bord"
        return texte + "."

    # --- Fichiers ----------------------------------------------------------------------------
    COLONNES = ["t_s", "x_mm", "y_mm", "hauteur_mm", "vx_m_s", "vy_m_s", "x_consigne_mm", "y_consigne_mm",
                "distance_consigne_mm", "theta_x_commande_deg", "theta_y_commande_deg", "theta_x_reel_deg",
                "theta_y_reel_deg", "moteur1_deg", "moteur2_deg", "moteur3_deg", "couple1_Nm", "couple2_Nm",
                "couple3_Nm", "decollee_pct", "erreur_mesure_mm"]

    def _ecrire(self, r: Resultats, ligne: dict) -> None:
        self.dossier.mkdir(parents=True, exist_ok=True)
        t0 = self.tentative["t0"]
        with open(self.dossier / ligne["fichier"], "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(self.COLONNES)
            for i in range(len(r.t)):
                tx, ty = r.inclinaison_reelle[i]
                alphas = r.alphas[i] or (math.nan,) * 3
                couples = r.couples[i] if r.couples and r.couples[i] else (math.nan,) * 3
                valeurs = [r.t[i] - t0, 1000 * r.x[i], 1000 * r.y[i], 1000 * r.hauteur[i], *r.vitesse[i],
                           1000 * r.x_ref[i], 1000 * r.y_ref[i],
                           1000 * math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]),
                           math.degrees(r.theta_x[i]), math.degrees(r.theta_y[i]),
                           math.degrees(tx), math.degrees(ty), *(math.degrees(a) for a in alphas),
                           *couples, 100 * r.decollee[i] if r.decollee else math.nan,
                           1000 * r.erreur_mesure[i]]
                w.writerow([_nombre(v, 4) for v in valeurs])
        cles = list(ligne.keys())
        for l in self.resume:                  # colonnes de toutes les tentatives (lancer ⊃ libre)
            cles += [k for k in l if k not in cles]
        with open(self.dossier / "resume.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cles)
            for l in self.resume:
                w.writerow([_nombre(l.get(k), 3) if isinstance(l.get(k), float) else
                            ("oui" if l.get(k) is True else "non" if l.get(k) is False else
                             "" if l.get(k) is None else l.get(k)) for k in cles])

    # --- Pas de simulation -------------------------------------------------------------------
    def pas(self) -> None:
        """Avance d'un pas de dt_physique, quoi qu'il arrive à la balle."""
        sim, phys = self.sim, self.phys
        if self.etat == "tombée":
            # Plus de balle à suivre : plateau ramené à plat, rien n'est enregistré
            if sim.n % self.ppc == 0:
                phys.commander(0.0, 0.0)
                phys.lire_decolles_periode()     # non enregistré
            phys.pas(self.dt)
            sim.n += 1
            return
        controle = sim.n % self.ppc == 0
        if not sim.pas():                       # balle tombée (détectée avant le pas)
            if self.tentative is not None:
                self._terminer("tombée")
            else:
                self.etat = "tombée"
                self.message = "La balle est tombée."
            return
        h = phys.hauteur_balle()
        if self.tentative is not None:
            self._suivre_pas(h)
        if controle:
            self._apres_controle(h)

    def _suivre_pas(self, h: float) -> None:
        """Atterrissages (après un vol de plus de VOL_MIN au-dessus de POSEE) et hauteur maximale des
        rebonds, vérifiés à chaque pas : un échantillonnage à 50 Hz les raterait."""
        tv = self.tentative
        if h > POSEE:
            self._pas_en_vol += 1
            if tv["atterrissages"]:
                tv["rebond_max"] = max(tv["rebond_max"], h)
        else:
            if self._pas_en_vol * self.dt >= VOL_MIN:
                tv["atterrissages"] += 1
            self._pas_en_vol = 0

    def _apres_controle(self, h: float) -> None:
        r = self.sim.r
        d = math.hypot(r.x[-1] - r.x_ref[-1], r.y[-1] - r.y_ref[-1])
        self.courbes.append((r.t[-1], d, r.theta_x[-1], r.theta_y[-1], h, r.x[-1], r.y[-1],
                             r.x_ref[-1], r.y_ref[-1]))
        if self.tentative is None:
            # Attente : on garde 0,5 s d'historique, et une perturbation démarre une tentative
            n_avant = round(AVANT_PERTURBATION / self.cfg.periode_controle)
            if len(r.t) > 20 * n_avant:
                self.sim.r = _garder_fin(r, n_avant)
            if d > SEUIL_PERTURBATION:
                self._nouvelle_tentative("perturbation", {}, avant=_garder_fin(self.sim.r, n_avant))
                self.tentative["t0"] = self.sim.r.t[-1]
            return
        tv = self.tentative
        if d < SEUIL_STABLE and h < POSEE:
            if tv["debut_stable"] is None:
                tv["debut_stable"] = r.t[-1]
            elif r.t[-1] - tv["debut_stable"] >= DUREE_STABLE:
                self._terminer("stabilisée")
                return
        else:
            tv["debut_stable"] = None
        if self.t - tv["t0"] >= self.duree_max:
            self._terminer("non stabilisée")

    def fermer(self) -> list:
        """Termine la tentative en cours ; renvoie le résumé de toutes les tentatives."""
        if self.tentative is not None:
            self._terminer("interrompue")
        self.fermer_camera()
        return self.resume
