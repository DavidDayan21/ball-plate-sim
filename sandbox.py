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

from simulation.session import SETPOINTS, Session, sandbox_config

SPACE_KEY = 32
FRAMES_PER_SECOND = 60
MAX_LAG = 0.05       # (s) au-delà, la simulation ralentit au lieu de rattraper le temps réel
LATENCIES = {"20 ms": 1, "60 ms": 3, "100 ms": 5}


class Sandbox:
    def __init__(self, session: Session, camera: bool = False):
        self.s = session
        self.camera = camera
        self.actions = queue.Queue()        # actions du panneau, exécutées par la simulation
        self.pause = False
        self.stop_event = threading.Event()
        self.error = None
        self.speed = 1.0                  # temps simulé / temps réel, mesuré
        self._build_panel()

    # --- Panneau -----------------------------------------------------------------------------
    def _build_panel(self) -> None:
        self.fig = fig = plt.figure("Bac à sable : commandes et mesures", figsize=(13, 7.2))
        fig.text(0.02, 0.965, "Lancer ou lâcher la balle", fontsize=12, weight="bold")

        def slider(y, text, vmin, vmax, v0, step=None):
            ax = fig.add_axes((0.15, y, 0.16, 0.03))
            return Slider(ax, text, vmin, vmax, valinit=v0, valstep=step)

        max_radius = 100 * (self.s.cfg.plate.plate_radius - self.s.cfg.plate.ball_radius)
        self.c_distance = slider(0.90, "Départ : distance (cm)", 0, max_radius, 10, 0.5)
        self.c_angle = slider(0.85, "Départ : angle (°)", 0, 355, 180, 5)
        self.c_speed = slider(0.80, "Vitesse (m/s)", 0, 1.5, 0.4, 0.05)
        self.c_deviation = slider(0.75, "Déviation (°)", -90, 90, 0, 5)
        self.c_height = slider(0.70, "Hauteur (cm)", 0, 50, 0, 1)

        def button(x, y, l, text, action):
            b = Button(fig.add_axes((x, y, l, 0.05)), text)
            b.on_clicked(lambda _: action())
            return b

        self.buttons = [
            button(0.02, 0.61, 0.095, "Lancer", self._throw),
            button(0.12, 0.61, 0.095, "Lâcher", self._drop),
            button(0.22, 0.61, 0.095, "Centre", lambda: self.actions.put(self.s.center_ball)),
            button(0.02, 0.55, 0.145, "Tentative libre", lambda: self.actions.put(self.s.free_attempt)),
            button(0.17, 0.55, 0.145, "Pause / reprendre", self._toggle_pause),
        ]

        fig.text(0.02, 0.50, "Consigne", weight="bold")
        self.r_setpoint = RadioButtons(fig.add_axes((0.02, 0.36, 0.09, 0.13)), list(SETPOINTS))
        self.r_setpoint.on_clicked(lambda name: self.actions.put(lambda: self.s.set_setpoint(name)))
        fig.text(0.13, 0.50, "Latence de mesure", weight="bold")
        self.r_latency = RadioButtons(fig.add_axes((0.13, 0.36, 0.09, 0.13)), list(LATENCIES))
        self.r_latency.on_clicked(lambda name: self.actions.put(
            lambda: self.s.set_latency(LATENCIES[name])))
        self.c_compensation = CheckButtons(fig.add_axes((0.02, 0.29, 0.29, 0.05)),
                                           ["Compensation de la latence"], [False])
        self.c_compensation.on_clicked(lambda _: self.actions.put(
            lambda a=self.c_compensation.get_status()[0]: self.s.set_compensation(a)))

        self.text = fig.text(0.02, 0.24, "", va="top", fontsize=9, family="monospace")

        # Vue de dessus (même orientation que la vue 3D au démarrage : x vers la droite)
        self.ax_top = ax = fig.add_axes((0.36, 0.42, 0.27, 0.53))
        R = 1000 * self.s.cfg.plate.plate_radius
        ax.add_patch(Circle((0, 0), R, fill=False, lw=1.5, color="0.4"))
        ax.set_xlim(-R * 1.1, R * 1.1)
        ax.set_ylim(-R * 1.1, R * 1.1)
        ax.set_aspect("equal")
        ax.set_title("Vue de dessus (mm) : clic = point de départ", fontsize=10)
        ax.axhline(0, color="0.9", lw=0.8, zorder=0)
        ax.axvline(0, color="0.9", lw=0.8, zorder=0)
        self.trace, = ax.plot([], [], color="tab:orange", lw=1, alpha=0.6)
        self.ball = Circle((0, 0), 1000 * self.s.cfg.plate.ball_radius, color="tab:orange", zorder=3)
        ax.add_patch(self.ball)
        self.setpoint_marker, = ax.plot([], [], "+", color="tab:green", ms=14, mew=2, zorder=4)
        self.start_marker, = ax.plot([], [], "o", mfc="none", mec="tab:blue", ms=10, zorder=4)
        self.arrow = ax.annotate("", (0, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
        fig.canvas.mpl_connect("button_press_event", self._click)

        # Courbes en direct (10 dernières secondes)
        self.ax_curves = []
        titles = ["Distance balle - consigne (mm)", "Inclinaison commandée (°)", "Hauteur de la balle (mm)"]
        for i, title in enumerate(titles):
            a = fig.add_axes((0.69, 0.70 - 0.31 * i, 0.29, 0.25))
            a.set_title(title, fontsize=10)
            a.set_xlim(-10, 0)
            a.grid(alpha=0.3)
            self.ax_curves.append(a)
        self.ax_curves[-1].set_xlabel("temps (s), 0 = maintenant")
        self.l_distance, = self.ax_curves[0].plot([], [], color="tab:red")
        self.l_tx, = self.ax_curves[1].plot([], [], label="θx")
        self.l_ty, = self.ax_curves[1].plot([], [], label="θy")
        self.ax_curves[1].legend(loc="upper left", fontsize=8)
        self.l_height, = self.ax_curves[2].plot([], [], color="tab:purple")
        for c in (self.c_distance, self.c_angle, self.c_speed, self.c_deviation, self.c_height):
            c.on_changed(lambda _: self._preview_throw())
        self._preview_throw()

    def _params(self):
        return dict(distance=self.c_distance.val / 100, angle_deg=self.c_angle.val,
                    speed=self.c_speed.val, height=self.c_height.val / 100,
                    deviation_deg=self.c_deviation.val)

    def _throw(self) -> None:
        p = self._params()
        self.actions.put(lambda: self.s.throw(**p))

    def _drop(self) -> None:
        p = dict(self._params(), speed=0.0, deviation_deg=0.0)
        self.actions.put(lambda: self.s.throw(**p))

    def _toggle_pause(self) -> None:
        self.pause = not self.pause

    def _click(self, ev) -> None:
        if ev.inaxes is not self.ax_top or ev.xdata is None:
            return
        d = min(math.hypot(ev.xdata, ev.ydata) / 10, self.c_distance.valmax)
        self.c_distance.set_val(round(d * 2) / 2)
        self.c_angle.set_val(round(math.degrees(math.atan2(ev.ydata, ev.xdata)) % 360 / 5) * 5 % 360)

    def _preview_throw(self) -> None:
        p = self._params()
        a = math.radians(p["angle_deg"])
        x, y = 1000 * p["distance"] * math.cos(a), 1000 * p["distance"] * math.sin(a)
        aim = a + math.pi + math.radians(p["deviation_deg"])
        l = 100 * p["speed"]                       # flèche : 1 m/s = 10 cm
        self.start_marker.set_data([x], [y])
        self.arrow.xy = (x + l * math.cos(aim), y + l * math.sin(aim))
        self.arrow.set_position((x, y))
        self.fig.canvas.draw_idle()

    def _fit_to_screen(self) -> None:
        """Réduit le panneau s'il est trop grand pour l'écran et le place à droite (fenêtre Tk,
        par défaut sous Windows ; sans effet avec une autre bibliothèque graphique)."""
        try:
            window = self.fig.canvas.manager.window
            width, height = window.winfo_screenwidth(), window.winfo_screenheight()
        except AttributeError:
            return
        w, h = self.fig.get_size_inches() * self.fig.dpi
        f = min(1.0, 0.7 * width / w, 0.8 * height / h)
        if f < 1.0:
            self.fig.set_dpi(self.fig.dpi * f)    # tout est réduit, textes compris
        window.update()
        window.geometry(f"+{max(0, width - window.winfo_width() - 10)}+10")

    def _update(self) -> None:
        try:
            data = list(self.s.curves)
        except RuntimeError:                         # modifiées pendant la copie : image suivante
            return
        if data:
            t_end = data[-1][0]
            t = [d[0] - t_end for d in data]
            self.l_distance.set_data(t, [1000 * d[1] for d in data])
            self.l_tx.set_data(t, [math.degrees(d[2]) for d in data])
            self.l_ty.set_data(t, [math.degrees(d[3]) for d in data])
            self.l_height.set_data(t, [1000 * d[4] for d in data])
            for a in self.ax_curves:
                a.relim()
                a.autoscale_view(scalex=False)
            recent = [d for d in data if d[0] > t_end - 3.0]
            self.trace.set_data([1000 * d[5] for d in recent], [1000 * d[6] for d in recent])
            self.ball.center = (1000 * data[-1][5], 1000 * data[-1][6])
            self.setpoint_marker.set_data([1000 * data[-1][7]], [1000 * data[-1][8]])
            self.ball.set_visible(self.s.state != "fallen")
        state = "PAUSE" if self.pause else {"attempt": "enregistrement", "waiting": "en attente",
                                           "fallen": "balle tombée"}[self.s.state]
        cfg = self.s.cfg
        lines = [f"Temps simulé : {self.s.t:6.1f} s ({self.speed:.2f} × temps réel)   État : {state}",
                  f"Capteur : {self.s.sensor}",
                  f"Consigne : {self.s.setpoint}   Latence : {round(1000 * cfg.latency * cfg.control_period)} ms"
                  f"   Compensation : {'oui' if cfg.latency_compensation else 'non'}",
                  "", *_wrap(self.s.message, 62), "",
                  f"Tentatives terminées : {len(self.s.summary)}",
                  f"Dossier : {self.s.folder}"]
        self.text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    # --- Simulation (fil séparé) -------------------------------------------------------------
    def _key(self, code: int) -> None:
        if code == SPACE_KEY:
            self.pause = not self.pause

    def _simulate(self, viewer) -> None:
        """Avance la simulation au rythme du temps réel et rafraîchit la vue 3D."""
        try:
            if self.camera:
                self.s.enable_camera()             # créée dans ce fil, qui seul l'utilise
            clock = time.perf_counter()
            real_mark, sim_mark = clock, self.s.t
            while not self.stop_event.is_set() and viewer.is_running():
                frame_start = time.perf_counter()
                elapsed, clock = min(frame_start - clock, MAX_LAG), frame_start
                while not self.actions.empty():
                    self.actions.get()()
                if self.pause:
                    real_mark, sim_mark = frame_start, self.s.t
                else:
                    target = self.s.t + elapsed
                    while self.s.t < target:
                        self.s.step()
                    if frame_start - real_mark > 1.0:      # vitesse mesurée sur 1 s
                        self.speed = (self.s.t - sim_mark) / (frame_start - real_mark)
                        real_mark, sim_mark = frame_start, self.s.t
                viewer.sync()
                remaining = 1 / FRAMES_PER_SECOND - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e:                      # affichée par le fil principal
            self.error = e
        finally:
            self.s.close_camera()
            self.stop_event.set()

    def run(self) -> None:
        model, data = self.s.phys.model, self.s.phys.data
        existing = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=self._key,
                                              show_left_ui=False, show_right_ui=False)
        viewer_threads = [f for f in threading.enumerate() if f not in existing]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.6, -35, 90
        viewer.cam.lookat[:] = (0, 0, self.s.cfg.geometry.height)
        thread = threading.Thread(target=self._simulate, args=(viewer,), daemon=True)
        thread.start()
        plt.show(block=False)
        self._fit_to_screen()
        try:
            while not self.stop_event.is_set() and plt.fignum_exists(self.fig.number):
                self._update()
                self.fig.canvas.start_event_loop(0.2)
        finally:
            self.stop_event.set()
            thread.join(timeout=5)
            viewer.close()
            for f in viewer_threads:      # attendre la fermeture complète de la vue 3D, sinon plantage en sortie
                f.join(timeout=5)
            plt.close(self.fig)
        if self.error is not None:
            raise self.error


def _wrap(text: str, width: int):
    """Découpe un texte en lignes de largeur maximale donnée."""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Bac à sable 3D du robot équilibriste")
    parser.add_argument("--camera", action="store_true", help="mesure par la caméra virtuelle")
    parser.add_argument("--max-duration", type=float, default=30.0, help="durée maximale d'une tentative (s)")
    args = parser.parse_args()
    session = Session(sandbox_config(), max_duration=args.max_duration)
    print("Bac à sable lancé. Ferme une des deux fenêtres pour arrêter.")
    try:
        Sandbox(session, camera=args.camera).run()
    finally:
        summary = session.close()
        if summary:
            print(f"{len(summary)} tentative(s) enregistrée(s) dans {session.folder}")
            for row in summary:
                print(" ", Session._describe(row))


if __name__ == "__main__":
    main()
