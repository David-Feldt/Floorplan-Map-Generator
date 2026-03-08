"""
End-to-end navigation simulation tests.

Runs the robot through every environment condition, verifying it actually moves
correctly: advances on clear paths, stops for obstacles, retreats from threats,
handles dynamic transitions, floor hazards, LLM overrides, sensor noise, and
edge cases.

No camera or display required — uses SceneFrame objects directly.

Run:  cd demo && python -m pytest test_navigation.py -v
"""
from __future__ import annotations
import math
import time

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from perception import SceneObject, SceneFrame
from force_field import ForceField, ForceModifier
from config import (
    GOAL_STRENGTH, REPULSION_STRENGTH, DRAG, MAX_VELOCITY, MIN_DISTANCE,
    STALL_TIMEOUT, FPS,
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

DT = 1.0 / FPS  # 33ms per frame at 30fps


def simulate(ff: ForceField, scene: SceneFrame, seconds: float) -> list[dict]:
    """Run the force field for `seconds` of simulated time, return frame log."""
    frames = int(seconds / DT)
    log = []
    for i in range(frames):
        ff.compute(scene, DT)
        log.append({
            "t": i * DT,
            "pos": ff.position,
            "vel": ff.velocity,
            "net": ff.last_net_force,
            "action": ff.get_display_action(),
        })
    return log


def simulate_dynamic(ff: ForceField, scene_fn, seconds: float) -> list[dict]:
    """Run with a scene that changes each frame (callable returning SceneFrame)."""
    frames = int(seconds / DT)
    log = []
    for i in range(frames):
        scene = scene_fn(i, i * DT)
        ff.compute(scene, DT)
        log.append({
            "t": i * DT,
            "pos": ff.position,
            "vel": ff.velocity,
            "net": ff.last_net_force,
            "action": ff.get_display_action(),
        })
    return log


def make_scene(objects=None, floor_clear=True):
    return SceneFrame(
        objects=objects or [],
        floor_clear=floor_clear,
        timestamp=time.time(),
    )


def person(distance, angle=0.0, pose="standing", bbox_wh=(100, 300), oid="person_0"):
    cx = int((angle / 1.047 + 0.5) * 640)
    cy = 240
    return SceneObject(oid, "person", distance, angle, (cx, cy, *bbox_wh), pose)


def bottle(distance, angle=0.0, oid="bottle_0"):
    cx = int((angle / 1.047 + 0.5) * 640)
    return SceneObject(oid, "bottle", distance, angle, (cx, 400, 30, 60))


# ---------------------------------------------------------------------------
# Canonical scene builders (replace former mock scenarios)
# ---------------------------------------------------------------------------

def scene_clear():
    return make_scene()

def scene_person_blocking():
    return make_scene([person(1.2, 0.0, "standing", (100, 300))])

def scene_person_close():
    return make_scene([person(0.5, 0.0, "standing", (140, 350))])

def scene_person_fallen():
    return make_scene([person(1.5, 0.0, "on_ground", (200, 80))])

def scene_threat():
    return make_scene([person(0.8, 0.0, "arms_raised", (120, 320))])

def scene_floor_hazard():
    return make_scene(
        [SceneObject("bottle_0", "bottle", 2.0, -0.3, (100, 400, 30, 60), None, True)],
        floor_clear=False,
    )

def scene_waving():
    return make_scene([person(2.5, 0.4, "waving", (100, 300))])

def scene_two_people():
    return make_scene([
        person(1.0, -0.2, "standing", (100, 300), "person_0"),
        person(3.0, 0.5, "standing", (90, 280), "person_1"),
    ])

ALL_SCENES = {
    "clear": scene_clear,
    "person_blocking": scene_person_blocking,
    "person_close": scene_person_close,
    "person_fallen": scene_person_fallen,
    "threat": scene_threat,
    "floor_hazard": scene_floor_hazard,
    "waving": scene_waving,
    "two_people": scene_two_people,
}


# ===========================================================================
# 1. CLEAR PATH — robot must move forward and keep accelerating
# ===========================================================================

class TestClearPath:
    def test_moves_forward(self):
        """On empty scene, robot should move forward steadily."""
        ff = ForceField()
        log = simulate(ff, make_scene(), 3.0)
        assert log[-1]["vel"] > 0.5, f"Should be moving forward, vel={log[-1]['vel']}"
        assert log[-1]["pos"] > 0.0, f"Should have advanced, pos={log[-1]['pos']}"

    def test_reaches_terminal_velocity(self):
        """Drag should cap velocity below MAX_VELOCITY."""
        ff = ForceField()
        log = simulate(ff, make_scene(), 10.0)
        assert log[-1]["vel"] < MAX_VELOCITY
        last_vels = [f["vel"] for f in log[-30:]]
        assert max(last_vels) - min(last_vels) < 0.05, "Velocity should converge"

    def test_position_always_increases(self):
        """Position should be monotonically increasing on clear path."""
        ff = ForceField()
        log = simulate(ff, make_scene(), 5.0)
        for i in range(1, len(log)):
            assert log[i]["pos"] >= log[i-1]["pos"] - 0.001

    def test_action_is_moving_forward(self):
        """Display should say MOVING FORWARD after ramp-up."""
        ff = ForceField()
        log = simulate(ff, make_scene(), 2.0)
        assert log[-1]["action"] == "MOVING FORWARD"

    def test_makes_significant_progress(self):
        """Robot should cover meaningful distance in 10 seconds."""
        ff = ForceField()
        log = simulate(ff, make_scene(), 10.0)
        assert log[-1]["pos"] > 5.0, f"Only moved {log[-1]['pos']:.1f} units in 10s"


# ===========================================================================
# 2. PERSON BLOCKING — robot must slow/stop, not crash
# ===========================================================================

class TestPersonBlocking:
    def test_slows_down(self):
        ff_clear = ForceField()
        simulate(ff_clear, make_scene(), 3.0)
        ff_blocked = ForceField()
        simulate(ff_blocked, scene_person_blocking(), 3.0)
        assert ff_blocked.velocity < ff_clear.velocity

    def test_does_not_advance_past_close_person(self):
        ff = ForceField()
        log = simulate(ff, scene_person_close(), 5.0)
        assert log[-1]["vel"] < 0.5, f"Should not push through, vel={log[-1]['vel']}"

    def test_close_person_causes_retreat(self):
        ff = ForceField()
        scene = make_scene([person(0.3, bbox_wh=(180, 400))])
        log = simulate(ff, scene, 3.0)
        assert log[-1]["vel"] < 0.0
        assert log[-1]["action"] == "RETREATING"


# ===========================================================================
# 3. DISTANCE RESPONSE — proportional
# ===========================================================================

class TestDistanceResponse:
    def test_closer_means_more_repulsion(self):
        vels = {}
        for dist in [0.5, 1.0, 2.0, 4.0]:
            ff = ForceField()
            simulate(ff, make_scene([person(dist)]), 3.0)
            vels[dist] = ff.velocity
        assert vels[0.5] < vels[1.0] < vels[2.0] < vels[4.0], f"Closer=slower: {vels}"

    def test_very_far_person_minimal_effect(self):
        ff = ForceField()
        log = simulate(ff, make_scene([person(10.0, bbox_wh=(30, 90))]), 3.0)
        assert log[-1]["vel"] > 0.8

    def test_inverse_square_law(self):
        forces = {}
        for dist in [1.0, 2.0]:
            ff = ForceField()
            ff.compute(make_scene([person(dist)]), DT)
            obj_forces = [f for f in ff.last_forces if f["source"] != "goal"]
            if obj_forces:
                forces[dist] = abs(obj_forces[0]["force"])
        if 1.0 in forces and 2.0 in forces:
            ratio = forces[1.0] / forces[2.0]
            assert 3.0 < ratio < 5.0, f"Inverse-square ratio should be ~4, got {ratio}"


# ===========================================================================
# 4. ANGLE RESPONSE
# ===========================================================================

class TestAngleResponse:
    def test_dead_ahead_strongest(self):
        ff_center = ForceField()
        ff_center.compute(make_scene([person(1.5, angle=0.0)]), DT)
        cf = [f for f in ff_center.last_forces if f["source"] != "goal"]
        ff_side = ForceField()
        ff_side.compute(make_scene([person(1.5, angle=0.4)]), DT)
        sf = [f for f in ff_side.last_forces if f["source"] != "goal"]
        if cf and sf:
            assert abs(cf[0]["force"]) > abs(sf[0]["force"])

    def test_peripheral_object_allows_movement(self):
        ff = ForceField()
        log = simulate(ff, make_scene([person(1.0, angle=0.5)]), 3.0)
        assert log[-1]["vel"] > 0.0

    def test_symmetric_angles_produce_same_force(self):
        ff_l = ForceField()
        ff_l.compute(make_scene([person(1.5, angle=-0.3)]), DT)
        ff_r = ForceField()
        ff_r.compute(make_scene([person(1.5, angle=0.3)]), DT)
        lf = [f for f in ff_l.last_forces if f["source"] != "goal"]
        rf = [f for f in ff_r.last_forces if f["source"] != "goal"]
        if lf and rf:
            assert abs(abs(lf[0]["force"]) - abs(rf[0]["force"])) < 0.01


# ===========================================================================
# 5. FLOOR HAZARD
# ===========================================================================

class TestFloorHazard:
    def test_floor_hazard_slows_robot(self):
        ff_safe = ForceField()
        simulate(ff_safe, make_scene(floor_clear=True), 3.0)
        ff_hazard = ForceField()
        simulate(ff_hazard, make_scene(floor_clear=False), 3.0)
        assert ff_hazard.velocity < ff_safe.velocity * 0.3

    def test_floor_hazard_with_object(self):
        ff = ForceField()
        log = simulate(ff, scene_floor_hazard(), 3.0)
        assert log[-1]["vel"] < 0.3

    def test_floor_hazard_reduces_goal_force(self):
        ff = ForceField()
        ff.compute(make_scene(floor_clear=False), DT)
        goal_force = [f for f in ff.last_forces if f["source"] == "goal"][0]["force"]
        assert abs(goal_force - GOAL_STRENGTH * 0.1) < 0.01

    def test_floor_clear_recovery(self):
        ff = ForceField()
        simulate(ff, make_scene(floor_clear=False), 2.0)
        assert ff.velocity < 0.3
        log = simulate(ff, make_scene(floor_clear=True), 3.0)
        assert log[-1]["vel"] > 0.5


# ===========================================================================
# 6. THREAT — with LLM modifier, robot must retreat
# ===========================================================================

class TestThreatResponse:
    def test_threat_with_llm_modifier_retreats(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=4.0, label="threat"))
        log = simulate(ff, scene_threat(), 3.0)
        assert log[-1]["vel"] < -0.5

    def test_extreme_threat_rapid_retreat(self):
        ff_normal = ForceField()
        scene = scene_person_close()
        simulate(ff_normal, scene, 2.0)
        ff_threat = ForceField()
        ff_threat.set_modifier("person_0", ForceModifier(repulsion_scale=5.0, label="threat"))
        simulate(ff_threat, scene, 2.0)
        assert ff_threat.velocity < ff_normal.velocity

    def test_threat_display_reason(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=3.0, label="threat"))
        ff.compute(make_scene([person(1.0)]), DT)
        assert "threat" in ff.get_display_reason()


# ===========================================================================
# 7. WAVING THROUGH — LLM reduces repulsion, robot passes
# ===========================================================================

class TestWavingThrough:
    def test_waving_person_low_repulsion(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=0.1, label="waving_through"))
        log = simulate(ff, scene_waving(), 3.0)
        assert log[-1]["vel"] > 0.5

    def test_waving_vs_blocking_comparison(self):
        ff_block = ForceField()
        simulate(ff_block, make_scene([person(1.5)]), 3.0)
        ff_wave = ForceField()
        ff_wave.set_modifier("person_0", ForceModifier(repulsion_scale=0.1, label="waving"))
        simulate(ff_wave, make_scene([person(1.5, pose="waving")]), 3.0)
        assert ff_wave.velocity > ff_block.velocity


# ===========================================================================
# 8. PERSON FALLEN — moderate caution
# ===========================================================================

class TestPersonFallen:
    def test_fallen_person_with_modifier(self):
        """Modifier increases repulsion vs same scene without modifier."""
        ff_no_mod = ForceField()
        simulate(ff_no_mod, scene_person_fallen(), 3.0)
        ff_mod = ForceField()
        ff_mod.set_modifier("person_0", ForceModifier(repulsion_scale=2.0, label="injured"))
        simulate(ff_mod, scene_person_fallen(), 3.0)
        assert ff_mod.velocity < ff_no_mod.velocity

    def test_speak_text_appears(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(
            repulsion_scale=2.0, label="injured",
            speak_text="Person on ground, waiting for safety"
        ))
        ff.compute(scene_person_fallen(), DT)
        action = ff.get_display_action()
        assert "SPEAK:" in action
        assert "Person on ground" in action


# ===========================================================================
# 9. MULTIPLE OBJECTS — combined repulsion
# ===========================================================================

class TestMultipleObjects:
    def test_two_people_more_repulsion_than_one(self):
        ff_one = ForceField()
        simulate(ff_one, make_scene([person(1.5)]), 3.0)
        ff_two = ForceField()
        simulate(ff_two, scene_two_people(), 3.0)
        assert ff_two.velocity < ff_one.velocity

    def test_crowd_causes_full_stop(self):
        ff = ForceField()
        objs = [
            person(0.8, angle=-0.3, oid="p0"),
            person(0.6, angle=0.0, oid="p1"),
            person(0.9, angle=0.3, oid="p2"),
        ]
        log = simulate(ff, make_scene(objs), 3.0)
        assert log[-1]["vel"] < 0.0

    def test_mixed_objects(self):
        ff_person = ForceField()
        simulate(ff_person, make_scene([person(1.5)]), 3.0)
        ff_mixed = ForceField()
        simulate(ff_mixed, make_scene([person(1.5), bottle(1.0, 0.1)]), 3.0)
        assert ff_mixed.velocity < ff_person.velocity


# ===========================================================================
# 10. EQUILIBRIUM ESCAPE
# ===========================================================================

class TestEquilibriumEscape:
    def test_stall_triggers_escape(self):
        ff = ForceField()
        scene = make_scene([person(0.8, bbox_wh=(200, 400))])
        log = simulate(ff, scene, STALL_TIMEOUT + 3.0)
        escaped = any(f["vel"] < -0.3 for f in log)
        assert escaped, "Robot should trigger escape nudge after stall"

    def test_no_stall_on_clear_path(self):
        ff = ForceField()
        log = simulate(ff, make_scene(), 10.0)
        for frame in log:
            assert frame["action"] != "STALL ESCAPE"

    def test_stall_counter_resets_on_movement(self):
        ff = ForceField()
        simulate(ff, make_scene([person(0.8, bbox_wh=(200, 400))]), STALL_TIMEOUT + 1.0)
        simulate(ff, make_scene(), 1.0)
        assert ff._stall_time == 0.0


# ===========================================================================
# 11. DYNAMIC ENVIRONMENT — scene changes over time
# ===========================================================================

class TestDynamicEnvironment:
    def test_person_appears_then_disappears(self):
        ff = ForceField()
        log1 = simulate(ff, make_scene(), 2.0)
        vel_clear = log1[-1]["vel"]
        assert vel_clear > 0.5
        log2 = simulate(ff, make_scene([person(1.0)]), 3.0)
        vel_blocked = log2[-1]["vel"]
        assert vel_blocked < vel_clear
        log3 = simulate(ff, make_scene(), 3.0)
        assert log3[-1]["vel"] > vel_blocked

    def test_approaching_object(self):
        ff = ForceField()
        def scene_fn(frame_i, t):
            return make_scene([person(max(0.3, 5.0 - t * 0.5))])
        log = simulate_dynamic(ff, scene_fn, 8.0)
        assert log[-1]["vel"] < log[30]["vel"]

    def test_object_receding(self):
        ff = ForceField()
        def scene_fn(frame_i, t):
            return make_scene([person(0.5 + t * 0.5)])
        log = simulate_dynamic(ff, scene_fn, 6.0)
        assert log[-1]["vel"] > log[10]["vel"]

    def test_floor_hazard_appears_mid_motion(self):
        ff = ForceField()
        simulate(ff, make_scene(floor_clear=True), 2.0)
        vel_before = ff.velocity
        assert vel_before > 0.5
        simulate(ff, make_scene(floor_clear=False), 2.0)
        assert ff.velocity < vel_before * 0.3

    def test_blocking_to_waving_to_clear(self):
        """Cycling through blocking → waving → clear should show proper transitions."""
        ff = ForceField()
        simulate(ff, scene_person_blocking(), 2.0)
        vel_block = ff.velocity
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=0.1, label="waving"))
        simulate(ff, scene_waving(), 2.0)
        vel_wave = ff.velocity
        ff.modifiers.clear()
        simulate(ff, scene_clear(), 2.0)
        vel_clear = ff.velocity
        assert vel_wave > vel_block
        assert vel_clear > vel_wave or abs(vel_clear - vel_wave) < 0.3


# ===========================================================================
# 12. ALL ENVIRONMENT CONDITIONS
# ===========================================================================

class TestAllEnvironments:
    def test_clear_path(self):
        ff = ForceField()
        log = simulate(ff, scene_clear(), 5.0)
        assert log[-1]["vel"] > 0.5
        assert log[-1]["action"] == "MOVING FORWARD"

    def test_person_blocking(self):
        ff = ForceField()
        log = simulate(ff, scene_person_blocking(), 5.0)
        assert log[-1]["vel"] < 1.5

    def test_person_close(self):
        ff = ForceField()
        log = simulate(ff, scene_person_close(), 5.0)
        assert log[-1]["vel"] < 0.0

    def test_person_fallen(self):
        ff = ForceField()
        log = simulate(ff, scene_person_fallen(), 5.0)
        assert isinstance(log[-1]["vel"], float)

    def test_threat(self):
        ff = ForceField()
        log = simulate(ff, scene_threat(), 5.0)
        assert log[-1]["vel"] < 0.5

    def test_threat_with_modifier(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=4.0, label="threat"))
        log = simulate(ff, scene_threat(), 5.0)
        assert log[-1]["vel"] < -0.5

    def test_floor_hazard(self):
        ff = ForceField()
        log = simulate(ff, scene_floor_hazard(), 5.0)
        assert log[-1]["vel"] < 0.3

    def test_waving_through(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=0.1, label="waving"))
        log = simulate(ff, scene_waving(), 5.0)
        assert log[-1]["vel"] > 0.5

    def test_two_people(self):
        ff = ForceField()
        log = simulate(ff, scene_two_people(), 5.0)
        assert log[-1]["vel"] < 1.0


# ===========================================================================
# 13. LLM MODIFIER LIFECYCLE
# ===========================================================================

class TestLLMModifierLifecycle:
    def test_modifier_changes_behavior(self):
        ff = ForceField()
        scene = make_scene([person(1.5)])
        simulate(ff, scene, 2.0)
        vel_before = ff.velocity
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=3.0, label="threat"))
        simulate(ff, scene, 2.0)
        assert ff.velocity < vel_before

    def test_removing_modifier_restores_behavior(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=5.0, label="threat"))
        scene = make_scene([person(1.5)])
        simulate(ff, scene, 2.0)
        vel_threat = ff.velocity
        ff.modifiers.clear()
        simulate(ff, scene, 3.0)
        assert ff.velocity > vel_threat

    def test_scale_zero_ignores_object(self):
        ff = ForceField()
        ff.set_modifier("person_0", ForceModifier(repulsion_scale=0.0, label="ignore"))
        log = simulate(ff, make_scene([person(1.0)]), 3.0)
        assert log[-1]["vel"] > 0.5

    def test_modifier_on_wrong_id_has_no_effect(self):
        ff = ForceField()
        ff.set_modifier("person_99", ForceModifier(repulsion_scale=5.0))
        scene = make_scene([person(1.0)])
        log1 = simulate(ff, scene, 3.0)
        ff2 = ForceField()
        log2 = simulate(ff2, scene, 3.0)
        assert abs(log1[-1]["vel"] - log2[-1]["vel"]) < 0.01


# ===========================================================================
# 14. BBOX SIZE EFFECT
# ===========================================================================

class TestBboxSizeEffect:
    def test_bigger_bbox_more_repulsion(self):
        ff_s = ForceField()
        ff_s.compute(make_scene([person(1.5, bbox_wh=(50, 100))]), DT)
        sf = sum(abs(f["force"]) for f in ff_s.last_forces if f["source"] != "goal")
        ff_b = ForceField()
        ff_b.compute(make_scene([person(1.5, bbox_wh=(200, 400))]), DT)
        bf = sum(abs(f["force"]) for f in ff_b.last_forces if f["source"] != "goal")
        assert bf > sf

    def test_tiny_bbox_has_minimum_effect(self):
        ff = ForceField()
        ff.compute(make_scene([SceneObject("x_0", "person", 1.0, 0.0, (320, 240, 1, 1))]), DT)
        obj_forces = [f for f in ff.last_forces if f["source"] != "goal"]
        assert len(obj_forces) > 0
        assert abs(obj_forces[0]["force"]) > 0.1


# ===========================================================================
# 15. STRESS TESTS — extreme/edge environments
# ===========================================================================

class TestStressEnvironments:
    def test_many_objects(self):
        ff = ForceField()
        objs = [person(1.0 + i*0.1, angle=(i-10)*0.05, oid=f"p{i}") for i in range(20)]
        log = simulate(ff, make_scene(objs), 3.0)
        assert log[-1]["vel"] < 0.0

    def test_object_at_minimum_distance(self):
        ff = ForceField()
        log = simulate(ff, make_scene([person(MIN_DISTANCE, bbox_wh=(200, 400))]), 3.0)
        assert not math.isnan(log[-1]["vel"]) and not math.isinf(log[-1]["vel"])
        assert log[-1]["vel"] < 0.0

    def test_object_below_minimum_distance(self):
        ff = ForceField()
        log = simulate(ff, make_scene([person(0.1, bbox_wh=(300, 450))]), 3.0)
        assert not math.isnan(log[-1]["vel"]) and not math.isinf(log[-1]["vel"])
        assert abs(log[-1]["vel"]) < MAX_VELOCITY + 0.1

    def test_rapid_scene_changes(self):
        ff = ForceField()
        clear = make_scene()
        blocked = make_scene([person(0.5, bbox_wh=(140, 350))])
        for i in range(300):
            ff.compute(clear if i % 2 == 0 else blocked, DT)
        assert not math.isnan(ff.velocity) and not math.isinf(ff.velocity)

    def test_all_zeros_scene(self):
        ff = ForceField()
        ff.compute(make_scene([SceneObject("z_0", "person", 0.0, 0.0, (0, 0, 0, 0))]), DT)
        assert not math.isnan(ff.velocity) and not math.isinf(ff.velocity)

    def test_very_high_dt(self):
        ff = ForceField()
        ff.compute(make_scene([person(1.0)]), 1.0)
        assert not math.isnan(ff.velocity)
        assert abs(ff.velocity) <= MAX_VELOCITY + 0.1

    def test_very_small_dt(self):
        ff = ForceField()
        scene = make_scene([person(1.0)])
        for _ in range(10000):
            ff.compute(scene, 0.0001)
        assert not math.isnan(ff.velocity)

    def test_long_simulation_stability(self):
        ff = ForceField()
        log = simulate(ff, make_scene([person(2.0)]), 60.0)
        for frame in log:
            assert not math.isnan(frame["vel"]) and not math.isinf(frame["vel"])
            assert abs(frame["vel"]) <= MAX_VELOCITY + 0.1


# ===========================================================================
# 16. FULL PIPELINE INVARIANTS — across every environment
# ===========================================================================

class TestPipelineInvariants:
    def test_velocity_never_exceeds_max(self):
        for name, fn in ALL_SCENES.items():
            ff = ForceField()
            log = simulate(ff, fn(), 10.0)
            for frame in log:
                assert abs(frame["vel"]) <= MAX_VELOCITY + 0.01, \
                    f"Velocity exceeded max in {name}: {frame['vel']}"

    def test_no_nan_in_any_environment(self):
        for name, fn in ALL_SCENES.items():
            ff = ForceField()
            log = simulate(ff, fn(), 10.0)
            for frame in log:
                assert not math.isnan(frame["vel"]), f"NaN vel in {name}"
                assert not math.isnan(frame["pos"]), f"NaN pos in {name}"
                assert not math.isnan(frame["net"]), f"NaN net in {name}"

    def test_display_action_always_valid(self):
        valid = {"MOVING FORWARD", "RETREATING", "WAITING", "STALL ESCAPE"}
        for name, fn in ALL_SCENES.items():
            ff = ForceField()
            log = simulate(ff, fn(), 5.0)
            for frame in log:
                assert frame["action"] in valid or frame["action"].startswith("SPEAK:"), \
                    f"Unknown action '{frame['action']}' in {name}"

    def test_display_reason_never_empty(self):
        for name, fn in ALL_SCENES.items():
            ff = ForceField()
            ff.compute(fn(), DT)
            reason = ff.get_display_reason()
            assert isinstance(reason, str) and len(reason) > 0

    def test_goal_force_present_in_every_frame(self):
        for name, fn in ALL_SCENES.items():
            ff = ForceField()
            ff.compute(fn(), DT)
            goal_forces = [f for f in ff.last_forces if f["source"] == "goal"]
            assert len(goal_forces) == 1, f"Missing goal force in {name}"
