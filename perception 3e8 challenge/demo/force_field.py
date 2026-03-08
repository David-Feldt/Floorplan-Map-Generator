from __future__ import annotations
import math
from dataclasses import dataclass, field

from config import (
    GOAL_STRENGTH,
    REPULSION_STRENGTH,
    DRAG,
    MAX_VELOCITY,
    MIN_DISTANCE,
    ROBOT_MASS,
    STALL_TIMEOUT,
)
from perception import SceneFrame


# ---------------------------------------------------------------------------
# LLM-assigned context modifier for a single scene object
# ---------------------------------------------------------------------------

@dataclass
class ForceModifier:
    repulsion_scale: float = 1.0   # 0 = ignore … 5 = extreme danger
    label: str = ""                # semantic tag ("threat", "waving_through", …)
    speak_text: str = ""           # message robot should display


# ---------------------------------------------------------------------------
# Force field engine
# ---------------------------------------------------------------------------

class ForceField:
    def __init__(self):
        self.velocity: float = 0.0
        self.position: float = 0.0          # normalised forward axis
        self.modifiers: dict[str, ForceModifier] = {}
        self.last_net_force: float = 0.0
        self.last_forces: list[dict] = []   # for UI visualisation
        self._stall_time: float = 0.0       # accumulated time near zero velocity

    # -- called by LLM thread --------------------------------------------------

    def set_modifier(self, object_id: str, modifier: ForceModifier):
        self.modifiers[object_id] = modifier

    # -- called every render frame ---------------------------------------------

    def compute(self, scene: SceneFrame, dt: float) -> float:
        forces: list[dict] = []

        # constant goal attraction (forward = positive)
        f_goal = GOAL_STRENGTH

        # reduce goal when floor hazard detected
        if not scene.floor_clear:
            f_goal *= 0.1
            forces.append({"source": "floor_hazard", "force": 0.0, "modifier": "hazard"})

        forces.append({"source": "goal", "force": f_goal})

        # repulsion from each detected object
        for obj in scene.objects:
            dist = max(obj.distance_m, MIN_DISTANCE)

            # inverse-square base repulsion (pushes backward = negative)
            base = -REPULSION_STRENGTH / (dist * dist)

            # larger bounding-box → stronger repulsion (sqrt to avoid crushing small objects)
            size_ratio = (obj.bbox[2] * obj.bbox[3]) / (640 * 480)
            size_factor = max(math.sqrt(size_ratio), 0.2)

            # objects dead-ahead matter more than peripheral ones
            angle_factor = max(0.0, math.cos(obj.angle))

            # LLM context modifier (default 1.0)
            mod = self.modifiers.get(obj.id, ForceModifier())

            f = base * size_factor * angle_factor * mod.repulsion_scale
            forces.append({
                "source": f"{obj.obj_class}@{dist:.1f}m",
                "force": f,
                "modifier": mod.label,
            })

        # net force → acceleration → velocity → position
        net = sum(f["force"] for f in forces)
        acceleration = net / ROBOT_MASS
        self.velocity += acceleration * dt
        # framerate-independent drag: v *= e^(-drag * dt)
        self.velocity *= math.exp(-DRAG * dt)
        self.velocity = max(-MAX_VELOCITY, min(MAX_VELOCITY, self.velocity))

        # equilibrium escape: if stalled with obstacles, slowly retreat
        if abs(self.velocity) < 0.1 and scene.objects:
            self._stall_time += dt
            if self._stall_time > STALL_TIMEOUT:
                self.velocity = -0.6  # nudge backward to break equilibrium
                self._stall_time = 0.0
        else:
            self._stall_time = 0.0

        self.position += self.velocity * dt

        self.last_net_force = net
        self.last_forces = forces
        return self.position

    # -- UI helpers ------------------------------------------------------------

    def get_display_action(self) -> str:
        speak_texts = [m.speak_text for m in self.modifiers.values() if m.speak_text]
        if speak_texts:
            return f"SPEAK: {speak_texts[0]}"
        if self._stall_time > STALL_TIMEOUT * 0.5:
            return "STALL ESCAPE"
        if self.velocity > 0.5:
            return "MOVING FORWARD"
        if self.velocity < -0.5:
            return "RETREATING"
        return "WAITING"

    def get_display_reason(self) -> str:
        significant = [
            f for f in self.last_forces
            if abs(f["force"]) > 0.1 and f["source"] != "goal"
        ]
        if not significant:
            return "Path is clear"
        biggest = max(significant, key=lambda f: abs(f["force"]))
        label = biggest.get("modifier", "")
        if label:
            return f"{biggest['source']} — {label}"
        return f"{biggest['source']} in path"
