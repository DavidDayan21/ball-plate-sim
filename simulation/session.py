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

Données écrites dans results/attempts/<date_heure>/ :
    attempt_001.csv ...  toutes les mesures de la tentative, à 50 Hz
    summary.csv             une ligne par tentative (paramètres et résultat)
Format adapté à Excel en français : séparateur « ; », virgule décimale.
"""
from collections import deque
from dataclasses import fields, replace
import csv
import datetime
import math
from pathlib import Path

import numpy as np

from simulation.loop import Results, Simulation
from simulation.config import Config, square, center, circle

SETPOINTS = {"center": center, "circle": circle, "square": square}
STABLE_THRESHOLD = 0.01          # (m) distance à la consigne
STABLE_DURATION = 2.0           # (s)
DISTURBANCE_THRESHOLD = 0.02    # (m) en attente : au-delà, une tentative démarre toute seule
PRE_DISTURBANCE = 0.5     # (s) de mesures conservées avant le déclenchement
MIN_FLIGHT = 0.010              # (s) un vol plus court n'est pas compté comme un atterrissage
RESTING_HEIGHT = 0.001                # (m) balle considérée posée sous cette hauteur


def sandbox_config() -> Config:
    """Robot complet, balle au repos au centre, consigne au centre."""
    return Config(trajectory=center, start_position=(0.0, 0.0), duration=math.inf)


def _shift(function, t0: float):
    """Consigne qui démarre à t0 (un cercle commence à son point de départ)."""
    return lambda t: function(t - t0)


def _keep_last(r: Results, n: int) -> Results:
    """Copie des n derniers échantillons de r."""
    tail = Results()
    for f in fields(Results):
        value = getattr(r, f.name)
        if isinstance(value, list):
            setattr(tail, f.name, value[-n:] if n > 0 else [])
    return tail


def _format_number(v: float, decimals: int = 3) -> str:
    """Nombre au format français (virgule décimale)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.{decimals}f}".replace(".", ",")


class Session:
    def __init__(self, cfg: Config = None, folder="results/attempts", max_duration: float = 30.0,
                 ground: bool = True, record: bool = True):
        self.cfg = cfg or sandbox_config()
        if self.cfg.physics != "mujoco":
            raise ValueError("Le bac à sable utilise la physique MuJoCo")
        self.sim = Simulation(self.cfg)
        self.phys = self.sim.phys
        self.dt = self.cfg.dt_physics
        self.ppc = self.sim.steps_per_control
        self.max_duration = max_duration
        self.setpoint = "center"
        self.t_setpoint = 0.0
        self._apply_config()
        if ground:
            self._enable_ground()
        # Enregistrement
        self.record = record
        self.folder = Path(folder) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.summary = []                   # une ligne (dict) par tentative terminée
        self.number = 0
        self.attempt = None              # dict de la tentative en cours, None en attente
        self.state = "waiting"              # "tentative", "attente" ou "tombée"
        self.message = "Prêt."
        # Affichage en direct, 10 s d'historique : (t, distance à la consigne, θx, θy, hauteur, x, y,
        # x_consigne, y_consigne)
        self.curves = deque(maxlen=round(10.0 / self.cfg.control_period))
        # Suivi pas à pas (atterrissages)
        self._steps_in_flight = 0

    # --- Réglages ----------------------------------------------------------------------------
    def _enable_ground(self) -> None:
        """Le sol arrête la balle tombée (sans lui, elle tomberait indéfiniment).
        Bit de collision 2 : le sol ne touche que la balle, jamais le plateau."""
        m = self.phys.model
        ground, ball = m.geom("ground").id, m.geom("ball").id
        m.geom_contype[ground] = 2
        m.geom_conaffinity[ground] = 2
        m.geom_conaffinity[ball] |= 2

    def _apply_config(self, **changes) -> None:
        self.cfg = replace(self.cfg, **changes)
        self.sim.cfg = replace(self.cfg, trajectory=_shift(SETPOINTS[self.setpoint], self.t_setpoint))

    def set_setpoint(self, name: str) -> None:
        self.setpoint, self.t_setpoint = name, self.sim.t
        self._apply_config()

    def set_latency(self, periods: int) -> None:
        """Latence de mesure en périodes de contrôle (1 = 20 ms). Remet le contrôleur à zéro."""
        self._apply_config(latency=periods)
        self.sim.reset_control()

    def set_compensation(self, active: bool) -> None:
        self._apply_config(latency_compensation=active)
        self.sim.reset_control()

    def enable_camera(self) -> None:
        """Mesure par la caméra virtuelle, à appeler depuis le fil qui fait avancer la simulation."""
        if self.sim.camera is None:
            self.sim.create_camera()
            self.sim.reset_control()

    def close_camera(self) -> None:
        """Libère la caméra (depuis le fil qui l'a créée) ; retour au capteur fictif."""
        if self.sim.camera is not None:
            self.sim.camera.close()
            self.sim.camera = None
            self.sim.reset_control()

    @property
    def sensor(self) -> str:
        return "camera" if self.sim.camera is not None else "synthetic"

    @property
    def t(self) -> float:
        return self.sim.t

    # --- Actions -----------------------------------------------------------------------------
    def throw(self, distance: float, angle_deg: float, speed: float = 0.0, height: float = 0.0,
               deviation_deg: float = 0.0, action: str = None) -> None:
        """Place la balle au point de départ (distance, angle) du centre, à `hauteur` au-dessus du
        plateau, avec une vitesse horizontale dirigée vers le centre, tournée de `deviation_deg`.
        vitesse = 0 : simple lâcher. hauteur = 0 : balle lancée en roulant."""
        a = math.radians(angle_deg)
        x, y = distance * math.cos(a), distance * math.sin(a)
        aim = a + math.pi + math.radians(deviation_deg)
        vx, vy = speed * math.cos(aim), speed * math.sin(aim)
        action = action or ("drop" if speed == 0 else "throw")
        self._new_attempt(action, dict(distance_cm=100 * distance, angle_deg=angle_deg,
                                              speed_m_s=speed, height_cm=100 * height,
                                              deviation_deg=deviation_deg), reset=True)
        self.phys.place_ball(x, y, height, vx, vy, 0.0)
        self.phys.read_period_airborne_steps()          # remet le compteur de la période à zéro
        self.sim.last_measurement = (x, y)

    def center_ball(self) -> None:
        """Balle posée immobile au centre."""
        self.throw(0.0, 0.0, action="center")

    def free_attempt(self) -> None:
        """Enregistre sans toucher à la balle (pour la pousser à la souris, changer de consigne...)."""
        if self.state == "fallen":
            self.message = "La balle est tombée : utilise Lancer, Lâcher ou Centre."
            return
        self._new_attempt("free", {})

    # --- Tentatives --------------------------------------------------------------------------
    def _new_attempt(self, action: str, params: dict, before: Results = None,
                            reset: bool = False) -> None:
        if self.attempt is not None:
            self._finish("interrupted")
        self.number += 1
        if reset:
            self.sim.reset_control()
        self.sim.r = before if before is not None else Results()
        self.attempt = dict(number=self.number, action=action, params=params, t0=self.t,
                              time=datetime.datetime.now().strftime("%H:%M:%S"),
                              setpoint=self.setpoint, sensor=self.sensor,
                              latency_ms=round(1000 * self.cfg.latency * self.cfg.control_period),
                              compensation=self.cfg.latency_compensation,
                              stable_since=None, landings=0, max_bounce=0.0)
        self.state = "attempt"
        self._steps_in_flight = 0
        self.message = f"Tentative {self.number} ({action}) en cours."

    def _finish(self, result: str) -> dict:
        att = self.attempt
        r = self.sim.r
        t0 = att["t0"]
        after = [i for i, t in enumerate(r.t) if t >= t0]
        dist = [math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]) for i in after]
        edge = [math.hypot(r.x[i], r.y[i]) for i in after]
        tilt = [max(abs(r.theta_x[i]), abs(r.theta_y[i])) for i in after]
        torques = [max(abs(c) for c in r.torques[i]) for i in after if r.torques and r.torques[i]]
        airborne = [r.airborne[i] for i in after] if r.airborne else []
        row = dict(
            number=att["number"], time=att["time"], action=att["action"], **att["params"],
            setpoint=att["setpoint"], sensor=att["sensor"], latency_ms=att["latency_ms"],
            compensation=att["compensation"],
            result=result, duration_s=self.t - t0,
            stabilized_in_s=(att["stable_since"] - t0) if result == "stabilized" else None,
            max_error_mm=1000 * max(dist) if dist else None,
            min_edge_margin_mm=1000 * (self.cfg.plate.plate_radius - max(edge)) if edge else None,
            landings=att["landings"], max_bounce_mm=1000 * att["max_bounce"],
            max_tilt_deg=math.degrees(max(tilt)) if tilt else None,
            max_torque_Nm=float(max(torques)) if torques else None,
            airborne_pct=100 * float(np.mean(airborne)) if airborne else None,
            file=f"attempt_{att['number']:03d}.csv")
        self.summary.append(row)
        if self.record:
            self._write(r, row)
        self.attempt = None
        self.state = "fallen" if result == "fallen" else "waiting"
        self.message = self._describe(row)
        self.sim.r = Results()
        return row

    @staticmethod
    def _describe(l: dict) -> str:
        text = f"Tentative {l['number']} ({l['action']}) : {l['result']}"
        if l["stabilized_in_s"] is not None:
            text += f" en {l['stabilized_in_s']:.2f} s"
        else:
            text += f" après {l['duration_s']:.2f} s"
        if l["landings"]:
            text += f", {l['landings']} atterrissage(s), rebond max {l['max_bounce_mm']:.0f} mm"
        if l["min_edge_margin_mm"] is not None and l["result"] != "fallen":
            text += f", passée à {l['min_edge_margin_mm']:.0f} mm du bord"
        return text + "."

    # --- Fichiers ----------------------------------------------------------------------------
    COLUMNS = ["t_s", "x_mm", "y_mm", "height_mm", "vx_m_s", "vy_m_s", "x_setpoint_mm", "y_setpoint_mm",
                "setpoint_distance_mm", "theta_x_command_deg", "theta_y_command_deg", "theta_x_actual_deg",
                "theta_y_actual_deg", "motor1_deg", "motor2_deg", "motor3_deg", "torque1_Nm", "torque2_Nm",
                "torque3_Nm", "airborne_pct", "measurement_error_mm"]

    def _write(self, r: Results, row: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        t0 = self.attempt["t0"]
        with open(self.folder / row["file"], "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(self.COLUMNS)
            for i in range(len(r.t)):
                tx, ty = r.actual_tilt[i]
                alphas = r.alphas[i] or (math.nan,) * 3
                torques = r.torques[i] if r.torques and r.torques[i] else (math.nan,) * 3
                values = [r.t[i] - t0, 1000 * r.x[i], 1000 * r.y[i], 1000 * r.height[i], *r.velocity[i],
                           1000 * r.x_ref[i], 1000 * r.y_ref[i],
                           1000 * math.hypot(r.x[i] - r.x_ref[i], r.y[i] - r.y_ref[i]),
                           math.degrees(r.theta_x[i]), math.degrees(r.theta_y[i]),
                           math.degrees(tx), math.degrees(ty), *(math.degrees(a) for a in alphas),
                           *torques, 100 * r.airborne[i] if r.airborne else math.nan,
                           1000 * r.measurement_error[i]]
                w.writerow([_format_number(v, 4) for v in values])
        keys = list(row.keys())
        for l in self.summary:                  # colonnes de toutes les tentatives (lancer ⊃ libre)
            keys += [k for k in l if k not in keys]
        with open(self.folder / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(keys)
            for l in self.summary:
                w.writerow([_format_number(l.get(k), 3) if isinstance(l.get(k), float) else
                            ("yes" if l.get(k) is True else "no" if l.get(k) is False else
                             "" if l.get(k) is None else l.get(k)) for k in keys])

    # --- Pas de simulation -------------------------------------------------------------------
    def step(self) -> None:
        """Avance d'un pas de dt_physique, quoi qu'il arrive à la balle."""
        sim, phys = self.sim, self.phys
        if self.state == "fallen":
            # Plus de balle à suivre : plateau ramené à plat, rien n'est enregistré
            if sim.n % self.ppc == 0:
                phys.command(0.0, 0.0)
                phys.read_period_airborne_steps()     # non enregistré
            phys.step(self.dt)
            sim.n += 1
            return
        control = sim.n % self.ppc == 0
        if not sim.step():                       # balle tombée (détectée avant le pas)
            if self.attempt is not None:
                self._finish("fallen")
            else:
                self.state = "fallen"
                self.message = "La balle est tombée."
            return
        h = phys.ball_height()
        if self.attempt is not None:
            self._track_step(h)
        if control:
            self._after_control(h)

    def _track_step(self, h: float) -> None:
        """Atterrissages (après un vol de plus de VOL_MIN au-dessus de POSEE) et hauteur maximale des
        rebonds, vérifiés à chaque pas : un échantillonnage à 50 Hz les raterait."""
        att = self.attempt
        if h > RESTING_HEIGHT:
            self._steps_in_flight += 1
            if att["landings"]:
                att["max_bounce"] = max(att["max_bounce"], h)
        else:
            if self._steps_in_flight * self.dt >= MIN_FLIGHT:
                att["landings"] += 1
            self._steps_in_flight = 0

    def _after_control(self, h: float) -> None:
        r = self.sim.r
        d = math.hypot(r.x[-1] - r.x_ref[-1], r.y[-1] - r.y_ref[-1])
        self.curves.append((r.t[-1], d, r.theta_x[-1], r.theta_y[-1], h, r.x[-1], r.y[-1],
                             r.x_ref[-1], r.y_ref[-1]))
        if self.attempt is None:
            # Attente : on garde 0,5 s d'historique, et une perturbation démarre une tentative
            n_before = round(PRE_DISTURBANCE / self.cfg.control_period)
            if len(r.t) > 20 * n_before:
                self.sim.r = _keep_last(r, n_before)
            if d > DISTURBANCE_THRESHOLD:
                self._new_attempt("disturbance", {}, before=_keep_last(self.sim.r, n_before))
                self.attempt["t0"] = self.sim.r.t[-1]
            return
        att = self.attempt
        if d < STABLE_THRESHOLD and h < RESTING_HEIGHT:
            if att["stable_since"] is None:
                att["stable_since"] = r.t[-1]
            elif r.t[-1] - att["stable_since"] >= STABLE_DURATION:
                self._finish("stabilized")
                return
        else:
            att["stable_since"] = None
        if self.t - att["t0"] >= self.max_duration:
            self._finish("not stabilized")

    def close(self) -> list:
        """Termine la tentative en cours ; renvoie le résumé de toutes les tentatives."""
        if self.attempt is not None:
            self._finish("interrupted")
        self.close_camera()
        return self.summary
