"""Bac à sable 3D : le robot complet dans MuJoCo, en temps réel, piloté depuis un panneau de commandes.

Deux fenêtres :
    - la vue 3D MuJoCo (souris : clic gauche glissé = tourner, clic droit glissé = déplacer,
      molette = zoom ; double-clic sur la balle puis Ctrl + clic droit glissé = la pousser ;
      Tab et Maj + Tab : menus de MuJoCo) ;
    - le panneau de commandes et de mesures (matplotlib).

Actions :
    Lancer   balle posée au point de départ, à la hauteur choisie, lancée vers le centre à la vitesse
             choisie (tournée de la déviation). Hauteur > 0 : elle arrive en rebondissant.
    Lâcher   même point de départ et même hauteur, sans vitesse.
    Centre   balle posée immobile au centre.
    Tentative libre : enregistre sans toucher la balle (avant de la pousser à la souris par exemple).
    Clic dans la vue de dessus : choisit le point de départ.
    Barre d'espace (dans la vue 3D) ou bouton Pause : met en pause / reprend.

Chaque tentative est enregistrée dans results/attempts/<date_heure>/ (voir simulation/session.py).

Lancement (depuis ball-plate-sim, environnement activé) :
    python sandbox.py
    python sandbox.py --camera      # mesure par la caméra virtuelle au lieu du capteur fictif
"""
import argparse
import math
import queue
import threading
import time
import matplotlib.pyplot as plt
plt.rcParams["keymap.quit"] = ["ctrl+w"]     # pas de fermeture accidentelle par la touche q
from matplotlib.patches import Circle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import mujoco
import mujoco.viewer

from simulation.session import CONSIGNES, Session, config_bac_a_sable

TOUCHE_ESPACE = 32
IMAGES_PAR_SECONDE = 60
RETARD_MAX = 0.05       # (s) au-delà, la simulation ralentit au lieu de rattraper le temps réel
LATENCES = {"20 ms": 1, "60 ms": 3, "100 ms": 5}


class BacASable:
    def __init__(self, session: Session, camera: bool = False):
        self.s = session
        self.camera = camera
        self.actions = queue.Queue()        # actions du panneau, exécutées par la simulation
        self.pause = False
        self.arret = threading.Event()
        self.erreur = None
        self.vitesse = 1.0                  # temps simulé / temps réel, mesuré
        self._construire_panneau()

    # --- Panneau -----------------------------------------------------------------------------
    def _construire_panneau(self) -> None:
        self.fig = fig = plt.figure("Bac à sable : commandes et mesures", figsize=(13, 7.2))
        fig.text(0.02, 0.965, "Lancer ou lâcher la balle", fontsize=12, weight="bold")

        def curseur(y, texte, vmin, vmax, v0, pas=None):
            ax = fig.add_axes((0.15, y, 0.16, 0.03))
            return Slider(ax, texte, vmin, vmax, valinit=v0, valstep=pas)

        rayon_max = 100 * (self.s.cfg.plateau.rayon_plateau - self.s.cfg.plateau.rayon_balle)
        self.c_distance = curseur(0.90, "Départ : distance (cm)", 0, rayon_max, 10, 0.5)
        self.c_angle = curseur(0.85, "Départ : angle (°)", 0, 355, 180, 5)
        self.c_vitesse = curseur(0.80, "Vitesse (m/s)", 0, 1.5, 0.4, 0.05)
        self.c_deviation = curseur(0.75, "Déviation (°)", -90, 90, 0, 5)
        self.c_hauteur = curseur(0.70, "Hauteur (cm)", 0, 50, 0, 1)

        def bouton(x, y, l, texte, action):
            b = Button(fig.add_axes((x, y, l, 0.05)), texte)
            b.on_clicked(lambda _: action())
            return b

        self.boutons = [
            bouton(0.02, 0.61, 0.095, "Lancer", self._lancer),
            bouton(0.12, 0.61, 0.095, "Lâcher", self._lacher),
            bouton(0.22, 0.61, 0.095, "Centre", lambda: self.actions.put(self.s.centrer)),
            bouton(0.02, 0.55, 0.145, "Tentative libre", lambda: self.actions.put(self.s.tentative_libre)),
            bouton(0.17, 0.55, 0.145, "Pause / reprendre", self._basculer_pause),
        ]

        fig.text(0.02, 0.50, "Consigne", weight="bold")
        self.r_consigne = RadioButtons(fig.add_axes((0.02, 0.36, 0.09, 0.13)), list(CONSIGNES))
        self.r_consigne.on_clicked(lambda nom: self.actions.put(lambda: self.s.changer_consigne(nom)))
        fig.text(0.13, 0.50, "Latence de mesure", weight="bold")
        self.r_latence = RadioButtons(fig.add_axes((0.13, 0.36, 0.09, 0.13)), list(LATENCES))
        self.r_latence.on_clicked(lambda nom: self.actions.put(
            lambda: self.s.changer_latence(LATENCES[nom])))
        self.c_compensation = CheckButtons(fig.add_axes((0.02, 0.29, 0.29, 0.05)),
                                           ["Compensation de la latence"], [False])
        self.c_compensation.on_clicked(lambda _: self.actions.put(
            lambda a=self.c_compensation.get_status()[0]: self.s.changer_compensation(a)))

        self.texte = fig.text(0.02, 0.24, "", va="top", fontsize=9, family="monospace")

        # Vue de dessus (même orientation que la vue 3D au démarrage : x vers la droite)
        self.ax_dessus = ax = fig.add_axes((0.36, 0.42, 0.27, 0.53))
        R = 1000 * self.s.cfg.plateau.rayon_plateau
        ax.add_patch(Circle((0, 0), R, fill=False, lw=1.5, color="0.4"))
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect("equal")
        ax.set_title("Vue de dessus (mm) : clic = point de départ", fontsize=10)
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        self.trace, = ax.plot([], [], color="tab:orange", lw=1, alpha=0.6)
        self.balle = Circle((0, 0), 1000 * self.s.cfg.plateau.rayon_balle, color="tab:orange", zorder=3)
        ax.add_patch(self.balle)
        self.point_consigne, = ax.plot([], [], "+", color="tab:green", ms=14, mew=2, zorder=4)
        self.depart, = ax.plot([], [], "o", mfc="none", mec="tab:blue", ms=10, zorder=4)
        self.fleche = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
        fig.canvas.mpl_connect("button_press_event", self._clic)

        # Courbes en direct (10 dernières secondes)
        self.ax_courbes = []
        titres = ["Distance balle - consigne (mm)", "Inclinaison commandée (°)", "Hauteur de la balle (mm)"]
        for i, titre in enumerate(titres):
            a = fig.add_axes((0.69, 0.70 - 0.31 * i, 0.29, 0.25))
            a.set_title(titre, fontsize=10)
            a.set_xlim(-10, 0)
            a.grid(alpha=0.3)
            self.ax_courbes.append(a)
        self.ax_courbes[-1].set_xlabel("temps (s), 0 = maintenant")
        self.l_distance, = self.ax_courbes[0].plot([], [], color="tab:red")
        self.l_tx, = self.ax_courbes[1].plot([], [], label="θx")
        self.l_ty, = self.ax_courbes[1].plot([], [], label="θy")
        self.ax_courbes[1].legend(loc="upper left", fontsize=8)
        self.l_hauteur, = self.ax_courbes[2].plot([], [], color="tab:purple")
        for c in (self.c_distance, self.c_angle, self.c_vitesse, self.c_deviation, self.c_hauteur):
            c.on_changed(lambda _: self._apercu_lancer())
        self._apercu_lancer()

    def _parametres(self):
        return dict(distance=self.c_distance.val / 100, angle_deg=self.c_angle.val,
                    vitesse=self.c_vitesse.val, hauteur=self.c_hauteur.val / 100,
                    deviation_deg=self.c_deviation.val)

    def _lancer(self) -> None:
        p = self._parametres()
        self.actions.put(lambda: self.s.lancer(**p))

    def _lacher(self) -> None:
        p = dict(self._parametres(), vitesse=0.0, deviation_deg=0.0)
        self.actions.put(lambda: self.s.lancer(**p))

    def _basculer_pause(self) -> None:
        self.pause = not self.pause

    def _clic(self, ev) -> None:
        if ev.inaxes is not self.ax_dessus or ev.xdata is None:
            return
        d = min(math.hypot(ev.xdata, ev.ydata) / 10, self.c_distance.valmax)
        self.c_distance.set_val(round(d * 2) / 2)
        self.c_angle.set_val(round(math.degrees(math.atan2(ev.ydata, ev.xdata)) % 360 / 5) * 5 % 360)

    def _apercu_lancer(self) -> None:
        p = self._parametres()
        a = math.radians(p["angle_deg"])
        x, y = 1000 * p["distance"] * math.cos(a), 1000 * p["distance"] * math.sin(a)
        visee = a + math.pi + math.radians(p["deviation_deg"])
        l = 100 * p["vitesse"]                       # flèche : 1 m/s = 10 cm
        self.depart.set_data([x], [y])
        self.fleche.xy = (x + l * math.cos(visee), y + l * math.sin(visee))
        self.fleche.set_position((x, y))
        self.fig.canvas.draw_idle()

    def _ajuster_a_l_ecran(self) -> None:
        """Réduit le panneau s'il est trop grand pour l'écran et le place à droite (fenêtre Tk,
        par défaut sous Windows ; sans effet avec une autre bibliothèque graphique)."""
        try:
            fenetre = self.fig.canvas.manager.window
            largeur, hauteur = fenetre.winfo_screenwidth(), fenetre.winfo_screenheight()
        except AttributeError:
            return
        w, h = self.fig.get_size_inches() * self.fig.dpi
        f = min(1.0, 0.7 * largeur / w, 0.8 * hauteur / h)
        if f < 1.0:
            self.fig.set_dpi(self.fig.dpi * f)    # tout est réduit, textes compris
        fenetre.update()
        fenetre.geometry(f"+{max(0, largeur - fenetre.winfo_width() - 10)}+10")

    def _mettre_a_jour(self) -> None:
        try:
            donnees = list(self.s.courbes)
        except RuntimeError:                         # modifiées pendant la copie : image suivante
            return
        if donnees:
            t_fin = donnees[-1][0]
            t = [d[0] - t_fin for d in donnees]
            self.l_distance.set_data(t, [1000 * d[1] for d in donnees])
            self.l_tx.set_data(t, [math.degrees(d[2]) for d in donnees])
            self.l_ty.set_data(t, [math.degrees(d[3]) for d in donnees])
            self.l_hauteur.set_data(t, [1000 * d[4] for d in donnees])
            for a in self.ax_courbes:
                a.relim()
                a.autoscale_view(scalex=False)
            recent = [d for d in donnees if d[0] > t_fin - 3.0]
            self.trace.set_data([1000 * d[5] for d in recent], [1000 * d[6] for d in recent])
            self.balle.center = (1000 * donnees[-1][5], 1000 * donnees[-1][6])
            self.point_consigne.set_data([1000 * donnees[-1][7]], [1000 * donnees[-1][8]])
            self.balle.set_visible(self.s.etat != "tombée")
        etat = "PAUSE" if self.pause else {"tentative": "enregistrement", "attente": "en attente",
                                           "tombée": "balle tombée"}[self.s.etat]
        cfg = self.s.cfg
        lignes = [f"Temps simulé : {self.s.t:6.1f} s ({self.vitesse:.2f} × temps réel)   État : {etat}",
                  f"Capteur : {self.s.capteur}",
                  f"Consigne : {self.s.consigne}   Latence : {round(1000 * cfg.retard * cfg.periode_controle)} ms"
                  f"   Compensation : {'oui' if cfg.compensation_retard else 'non'}",
                  "", *_couper(self.s.message, 62), "",
                  f"Tentatives terminées : {len(self.s.resume)}",
                  f"Dossier : {self.s.dossier}"]
        self.texte.set_text("\n".join(lignes))
        self.fig.canvas.draw_idle()

    # --- Simulation (fil séparé) -------------------------------------------------------------
    def _touche(self, code: int) -> None:
        if code == TOUCHE_ESPACE:
            self.pause = not self.pause

    def _simuler(self, viewer) -> None:
        """Avance la simulation au rythme du temps réel et rafraîchit la vue 3D."""
        try:
            if self.camera:
                self.s.activer_camera()             # créée dans ce fil, qui seul l'utilise
            horloge = time.perf_counter()
            mesure_reel, mesure_simule = horloge, self.s.t
            while not self.arret.is_set() and viewer.is_running():
                debut_image = time.perf_counter()
                ecoule, horloge = min(debut_image - horloge, RETARD_MAX), debut_image
                while not self.actions.empty():
                    self.actions.get()()
                if self.pause:
                    mesure_reel, mesure_simule = debut_image, self.s.t
                else:
                    cible = self.s.t + ecoule
                    while self.s.t < cible:
                        self.s.pas()
                    if debut_image - mesure_reel > 1.0:      # vitesse mesurée sur 1 s
                        self.vitesse = (self.s.t - mesure_simule) / (debut_image - mesure_reel)
                        mesure_reel, mesure_simule = debut_image, self.s.t
                viewer.sync()
                reste = 1 / IMAGES_PAR_SECONDE - (time.perf_counter() - debut_image)
                if reste > 0:
                    time.sleep(reste)
        except Exception as e:                      # affichée par le fil principal
            self.erreur = e
        finally:
            self.s.fermer_camera()
            self.arret.set()

    def executer(self) -> None:
        model, data = self.s.phys.model, self.s.phys.data
        deja_la = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=self._touche,
                                              show_left_ui=False, show_right_ui=False)
        fils_vue = [f for f in threading.enumerate() if f not in deja_la]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.6, -35, 90
        viewer.cam.lookat[:] = (0, 0, self.s.cfg.geometrie.hauteur)
        fil = threading.Thread(target=self._simuler, args=(viewer,), daemon=True)
        fil.start()
        plt.show(block=False)
        self._ajuster_a_l_ecran()
        try:
            while not self.arret.is_set() and plt.fignum_exists(self.fig.number):
                self._mettre_a_jour()
                self.fig.canvas.start_event_loop(0.2)
        finally:
            self.arret.set()
            fil.join(timeout=5)
            viewer.close()
            for f in fils_vue:      # attendre la fermeture complète de la vue 3D, sinon plantage en sortie
                f.join(timeout=5)
            plt.close(self.fig)
        if self.erreur is not None:
            raise self.erreur


def _couper(texte: str, largeur: int):
    """Découpe un texte en lignes de largeur maximale donnée."""
    lignes, ligne = [], ""
    for mot in texte.split():
        if ligne and len(ligne) + 1 + len(mot) > largeur:
            lignes.append(ligne)
            ligne = mot
        else:
            ligne = f"{ligne} {mot}".strip()
    return lignes + [ligne] if ligne else lignes


def main() -> None:
    parser = argparse.ArgumentParser(description="Bac à sable 3D du robot équilibriste")
    parser.add_argument("--camera", action="store_true", help="mesure par la caméra virtuelle")
    parser.add_argument("--duree-max", type=float, default=30.0, help="durée maximale d'une tentative (s)")
    args = parser.parse_args()
    session = Session(config_bac_a_sable(), duree_max=args.duree_max)
    print("Bac à sable lancé. Ferme une des deux fenêtres pour arrêter.")
    try:
        BacASable(session, camera=args.camera).executer()
    finally:
        resume = session.fermer()
        if resume:
            print(f"{len(resume)} tentative(s) enregistrée(s) dans {session.dossier}")
            for ligne in resume:
                print(" ", Session._decrire(ligne))


if __name__ == "__main__":
    main()
