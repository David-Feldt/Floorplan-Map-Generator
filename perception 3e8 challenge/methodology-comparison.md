# Methodology Comparison: Challenge Spec vs Implementation

This document details how the implemented system diverges from the original challenge specification and why each change was made.

---

## 1. Perception Model: YOLO + MediaPipe vs SAM

**Challenge spec:** SAM (Segment Anything Model) as primary segmentation, with YOLO/OpenCV as alternatives.

**Implementation:** YOLOv8-nano for object detection + MediaPipe Pose for human pose classification. SAM is not used.

**Why the change:**

SAM produces pixel-level segmentation masks but does not classify what objects are. It answers "where are the boundaries" but not "what is this." The robot needs to know "person at 1.2m, crouching" — not a silhouette mask. Using SAM would require a second classification step on top of segmentation, adding latency and complexity with no navigation benefit.

YOLOv8-nano directly outputs classified bounding boxes with confidence scores in a single pass. At ~6MB it runs comfortably on CPU at real-time rates. MediaPipe adds skeletal landmark detection specifically for human poses, enabling the five-pose classification (standing, crouching, on_ground, arms_raised, waving) that the challenge scenarios require.

The combined YOLO + MediaPipe pipeline gives the robot exactly what it needs — object identity, location, and human pose — without the overhead of a general-purpose segmentation model.

---

## 2. Navigation Engine: Potential Fields vs Direct LLM Actions

**Challenge spec:** The LLM (Gemma) directly chooses from four discrete actions: WAIT, MOVE FORWARD, MOVE BACK, SPEAK.

**Implementation:** A physics-based potential field engine computes continuous velocity from force equations. The LLM provides contextual modifiers (repulsion scaling) rather than direct commands.

**Why the change:**

Direct LLM action selection has three problems:

1. **Latency gap.** Gemma runs at ~3Hz. Between LLM calls the robot is blind — it cannot react to a person suddenly stepping into its path. The potential field engine runs every render frame (30fps), giving 10x faster reaction time.

2. **No proportional response.** Discrete actions cannot express "slow down slightly" vs "stop immediately" vs "back up fast." The force field produces continuous velocity that scales naturally with distance, object size, and angle — a person at 3m causes gentle slowing while a person at 0.5m causes immediate retreat.

3. **Equilibrium handling.** When goal attraction and obstacle repulsion balance out, a discrete system oscillates between MOVE FORWARD and WAIT. The force field reaches a smooth equilibrium, and the stall-escape mechanism (3-second timeout, backward nudge) breaks deadlocks gracefully.

The LLM still participates — it reads the scene and assigns semantic force modifiers (e.g., "this person is waving you through, set repulsion to 0.1" or "this is a threat, set repulsion to 5.0"). This gives the LLM interpretive power over the scene without making it responsible for frame-by-frame control.

---

## 3. Depth Integration: Force Equations vs Text Descriptions

**Challenge spec:** Depth + segmentation data is compressed into a short text description before the LLM call. The depth map is used for scene description only.

**Implementation:** Depth data drives the force field directly through inverse-square repulsion. Distance in meters is extracted per-object via median depth sampling and fed into the physics engine every frame.

**Why the change:**

Converting depth to text ("person at 1.2m") and having an LLM decide what to do with that number discards the precision. The LLM might treat 1.1m and 1.3m identically because it is a language model, not a physics engine. The force equation `f = -8.0 / dist^2` gives mathematically correct distance-proportional responses with no ambiguity.

The depth map also drives floor hazard detection directly — the bottom 30% of the depth frame is thresholded for close objects, providing a hardware-level safety check that operates independently of both YOLO and the LLM.

---

## 4. Object Tracking: Centroid Tracker vs No Tracking

**Challenge spec:** No mention of cross-frame object tracking. Each scene snapshot is independent.

**Implementation:** A centroid tracker maintains stable object IDs across frames using bbox-center proximity matching.

**Why the addition:**

Without tracking, the LLM assigns a modifier to "person_0" in frame N, but in frame N+1 the same person might be detected as "person_2" (different YOLO detection order). The modifier is lost. The centroid tracker ensures "person_3" stays "person_3" as long as they remain visible, so LLM modifiers persist correctly and the force field sees consistent objects.

---

## 5. Floor Hazard: EMA Smoothing vs Binary Detection

**Challenge spec:** Floor hazard is a scenario ("Liquid / obstacle on floor") with no specified detection method.

**Implementation:** Depth thresholding on the bottom 30% of the frame, smoothed with an exponential moving average (alpha=0.3) to prevent oscillation.

**Why this approach:**

Raw depth thresholding produces binary flicker — one frame says hazard, the next says clear, the next says hazard again. This causes the robot to stutter between MOVING and WAITING. The EMA smooths the signal so the robot commits to stopping when a real hazard appears and does not react to single-frame noise.

When floor hazard is active, goal attraction drops to 10% of normal, effectively halting the robot regardless of other forces. This is a hardware-level safety override that does not depend on the LLM interpreting the hazard correctly.

---

## 6. UI Display Actions vs Challenge Actions

**Challenge spec:** Four actions — WAIT, MOVE FORWARD, MOVE BACK, SPEAK.

**Implementation:** Five actions — WAITING, MOVING FORWARD, RETREATING, STALL ESCAPE, SPEAK.

**Mapping:**

| Challenge | Implementation | Difference |
| --- | --- | --- |
| WAIT | WAITING | Velocity between -0.5 and 0.5 |
| MOVE FORWARD | MOVING FORWARD | Velocity > 0.5 |
| MOVE BACK | RETREATING | Velocity < -0.5 |
| (none) | STALL ESCAPE | New: equilibrium deadlock breaker |
| SPEAK | SPEAK: [text] | Same: LLM-assigned speak_text displayed on robot |

STALL ESCAPE is new — it activates when the robot is stuck in equilibrium for 3+ seconds, nudging backward to break free. The challenge spec did not anticipate this edge case because discrete actions do not create equilibrium states.

---

## 7. LLM Role: Interpreter vs Controller

**Challenge spec:** Gemma receives scene descriptions and outputs the robot's action directly.

**Implementation:** Gemma receives scene descriptions and outputs force modifiers (repulsion_scale, label, speak_text) that tune the physics engine.

**Why the change:**

This separation has two benefits:

1. **Graceful degradation.** If Gemma is slow, crashes, or produces invalid JSON, the robot continues navigating on default physics. The `--no-llm` flag runs the full system without any LLM. In the challenge spec, losing the LLM means losing all decision-making.

2. **Additive intelligence.** The LLM adds semantic understanding on top of working physics rather than being the sole decision-maker. A waving person gets repulsion_scale=0.1 (the robot passes through). A threat gets repulsion_scale=5.0 (aggressive retreat). The physics handles everything in between without LLM involvement.

---

## 8. File Structure

**Challenge spec:**
```
/demo
  main.py
  perception.py
  llm_agent.py
  ui.py
  scenarios.md
  README.md
```

**Implementation:**
```
/demo
  main.py            # main loop, pygame render
  perception.py      # Orbbec camera + YOLO + MediaPipe + tracker
  force_field.py     # potential field navigation engine (new)
  llm_context.py     # Gemma/Ollama modifier interface (renamed)
  config.py          # all tunable parameters (new)
  ui.py              # 3-panel pygame renderer
  test_perception.py # 58 perception tests
  test_navigation.py # 64 navigation tests
```

Key differences:
- `force_field.py` is entirely new — the challenge had no physics engine
- `config.py` centralizes all parameters — the challenge had no configuration layer
- `llm_agent.py` became `llm_context.py` to reflect its role as context provider, not agent
- `scenarios.md` replaced by 122 automated tests that verify every scenario programmatically
- No README — this documentation and the technical documentation serve that purpose

---

## Summary

The core challenge vision is preserved: an Orbbec depth camera feeds perception data to a local LLM, which influences a robot's navigation decisions displayed in a Pygame UI. The six demo scenarios all work as specified.

The methodology changes — potential fields instead of direct LLM control, YOLO+MediaPipe instead of SAM, continuous physics instead of discrete actions — solve real-time responsiveness, proportional distance response, and fault tolerance problems that the original architecture would encounter in practice.
