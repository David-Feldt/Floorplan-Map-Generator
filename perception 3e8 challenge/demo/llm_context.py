"""
LLM Context Interpreter — Gemma (via Ollama) reads scene snapshots
and outputs force modifiers.  Non-blocking on the robot: if the LLM
is slow or crashes, the robot keeps navigating on default forces.
"""
from __future__ import annotations
import json
import threading
import time

from config import LLM_MODEL, LLM_HISTORY_SIZE, LLM_LOOP_SLEEP
from force_field import ForceField, ForceModifier
from perception import SceneFrame

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are interpreting a scene for a delivery robot's navigation system.

Given objects detected by the robot's cameras, assess each object's impact on navigation safety.

For each object, output:
- repulsion_scale: 0.0 (ignore / safe to pass) to 5.0 (extreme danger, retreat immediately). Default is 1.0 (normal obstacle).
- label: short semantic tag (e.g. "blocking", "waving_through", "threat", "injured", "hazard")
- speak: optional short message the robot should display. Only include if the situation warrants communication.

Guidelines:
- Person waving or gesturing "go ahead" → repulsion_scale 0.1, label "waving_through"
- Person standing normally in path → repulsion_scale 1.0, label "blocking"
- Person on the ground / fallen → repulsion_scale 2.0, label "injured", speak "Person on ground, waiting for safety"
- Threatening posture / aggressive gesture → repulsion_scale 3.0-5.0, label "threat", speak "Potential threat, maintaining distance"
- Floor hazard (liquid, debris) → repulsion_scale 1.5, label "hazard"
- Object far away or off to side → repulsion_scale 0.5, label "peripheral"

Respond ONLY with valid JSON. Example:
{
  "objects": [
    {"id": "person_320_200", "repulsion_scale": 0.1, "label": "waving_through", "speak": ""},
    {"id": "bottle_100_400", "repulsion_scale": 1.5, "label": "hazard", "speak": "Obstacle detected on floor"}
  ]
}"""


# ---------------------------------------------------------------------------
# LLM context class
# ---------------------------------------------------------------------------

class LLMContext:
    def __init__(self, force_field: ForceField):
        self.ff = force_field
        self.history: list[str] = []

    def interpret(self, scene: SceneFrame):
        if ollama is None:
            return
        if not scene.objects:
            self.ff.modifiers.clear()
            return

        # build snapshot
        objs = []
        for o in scene.objects:
            objs.append({
                "id": o.id,
                "class": o.obj_class,
                "distance_m": o.distance_m,
                "position": (
                    "center" if abs(o.angle) < 0.3
                    else ("left" if o.angle < 0 else "right")
                ),
                "pose": o.pose,
                "floor_hazard_nearby": o.floor_hazard,
            })

        snapshot = json.dumps({"objects": objs, "floor_clear": scene.floor_clear})
        self.history.append(snapshot)
        if len(self.history) > LLM_HISTORY_SIZE:
            self.history.pop(0)

        user_msg = "Scene history (oldest first):\n"
        for i, s in enumerate(self.history):
            user_msg += f"  {i + 1}. {s}\n"
        user_msg += "\nInterpret each object for robot navigation."

        try:
            resp = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            self._apply(resp["message"]["content"])
        except Exception as exc:
            # LLM failure is non-critical
            print(f"[llm] error: {exc}")

    def _apply(self, raw: str):
        text = raw.strip()
        # strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return

        for obj_data in data.get("objects", []):
            obj_id = obj_data.get("id", "")
            mod = ForceModifier(
                repulsion_scale=float(obj_data.get("repulsion_scale", 1.0)),
                label=obj_data.get("label", ""),
                speak_text=obj_data.get("speak", ""),
            )
            self.ff.set_modifier(obj_id, mod)


# ---------------------------------------------------------------------------
# Thread target
# ---------------------------------------------------------------------------

def llm_loop(shared: dict, force_field: ForceField, stop: threading.Event):
    ctx = LLMContext(force_field)
    while not stop.is_set():
        scene: SceneFrame | None = shared.get("scene")
        if scene and scene.objects:
            ctx.interpret(scene)
        time.sleep(LLM_LOOP_SLEEP)
