# Delivery Robot — Potential Field Navigation: Technical Documentation

## Architecture Overview

The system is a real-time delivery robot navigation pipeline with four concurrent subsystems:

```
Orbbec Camera (RGB+Depth @ 30fps)
        │
        ▼
  Perception Thread ──────────────► shared["scene"] (SceneFrame)
  (YOLO + MediaPipe + Tracker)              │
                                            ├──► Force Field Engine (every render frame)
                                            │         │
                                            │         ▼
                                            │    velocity / position
                                            │
                                            └──► LLM Context Thread (~3Hz)
                                                      │
                                                      ▼
                                                ForceModifiers → Force Field
```

**Threads:**
1. **Main thread** — Pygame render loop at 30 FPS, reads `shared["scene"]`, calls `ForceField.compute()`, draws UI
2. **Perception thread** — Captures frames, runs YOLO + MediaPipe, updates `shared["scene"]`
3. **LLM thread** (optional) — Reads scene snapshots, queries Gemma:2b via Ollama, writes `ForceModifier` entries

All communication is through two shared structures: `shared["scene"]` (perception → main/LLM) and `ForceField.modifiers` (LLM → force engine). No locks are needed — Python's GIL ensures atomic dict assignment, and stale reads are acceptable.

---

## Module Reference

### `config.py` — All Tunable Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `GOAL_STRENGTH` | 3.0 | Constant forward attraction force |
| `REPULSION_STRENGTH` | 8.0 | Base repulsive force at 1m (inverse-square) |
| `ROBOT_MASS` | 1.0 | F=ma denominator; higher = slower response |
| `DRAG` | 3.0 | Exponential velocity decay coefficient |
| `MAX_VELOCITY` | 4.0 | Hard velocity clamp (both directions) |
| `MIN_DISTANCE` | 0.3 | Distance floor to prevent force singularity |
| `YOLO_CONFIDENCE` | 0.5 | Detection threshold |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8-nano (~6MB, CPU-viable) |
| `CAMERA_WIDTH` / `HEIGHT` | 640 / 480 | Orbbec stream resolution |
| `MEDIAPIPE_COMPLEXITY` | 0 | Fastest pose model (suitable for embedded) |
| `FLOOR_ANOMALY_THRESHOLD` | 1000 | Pixel count triggering floor hazard |
| `FLOOR_CLOSE_MM` | 500 | Depth threshold for close floor objects |
| `TRACKER_MAX_DISAPPEARED` | 10 | Frames before dropping a tracked object |
| `DEPTH_SAMPLE_RADIUS` | 10 | Pixels for median depth sampling region |
| `DEPTH_FALLBACK_MM` | 500.0 | Conservative fallback distance (0.5m) |
| `STALL_TIMEOUT` | 3.0 | Seconds before equilibrium escape triggers |
| `FLOOR_HAZARD_SMOOTH` | 0.3 | EMA alpha for floor hazard smoothing |
| `HFOV_RAD` | 1.047 | Horizontal FOV (~60 degrees) |
| `LLM_MODEL` | `gemma:2b` | Ollama model for semantic interpretation |
| `LLM_LOOP_SLEEP` | 0.3 | LLM query interval (~3Hz) |

---

### `perception.py` — Sensor Fusion Pipeline

#### Data Structures

**`SceneObject`** — A single detected entity:
- `id: str` — Stable tracking ID (e.g., `"person_3"`)
- `obj_class: str` — YOLO class name (`"person"`, `"bottle"`, etc.)
- `distance_m: float` — Median depth at bbox center, in meters
- `angle: float` — Horizontal angle from image center in radians
- `bbox: (cx, cy, w, h)` — Bounding box center and dimensions
- `pose: str | None` — For persons: `"standing"`, `"crouching"`, `"on_ground"`, `"arms_raised"`, `"waving"`
- `floor_hazard: bool` — Whether object is near a floor anomaly

**`SceneFrame`** — Complete snapshot for one frame:
- `objects: list[SceneObject]` — All detected objects with stable IDs
- `floor_clear: bool` — EMA-smoothed floor safety flag
- `timestamp: float` — `time.time()` of capture
- `rgb: np.ndarray` — Raw 640x480x3 color frame
- `depth_colorized: np.ndarray` — JET-colorized depth for UI display

#### Camera Class (Orbbec via pyorbbecsdk)

Initializes an Orbbec depth camera pipeline:
- Color stream: 640x480, RGB888, 30fps
- Depth stream: 640x480, Y16 (uint16 millimeters), 30fps
- Depth-to-color alignment: software mode (`OBAlignMode.SW_MODE`)
- `get_frames()` returns `(rgb, depth)` or `(None, None)` on timeout

#### `detect_objects(rgb, depth)` — YOLOv8-nano Detection

1. Runs YOLOv8-nano inference on the RGB frame (lazy-loaded singleton)
2. For each detection bounding box:
   - Extracts center coordinates and dimensions
   - Samples a 20x20 pixel region around bbox center from the depth map
   - Takes the **median** of non-zero depth values (robust to noise, holes)
   - Falls back to 500mm (0.5m) if no valid depth — conservative, treats unknown as close
   - Computes horizontal angle: `((cx / width) - 0.5) * HFOV_RAD`

#### `infer_pose(rgb)` — MediaPipe Pose Classification

Classifies a cropped person image into one of five poses using MediaPipe landmark positions:

| Pose | Condition |
|---|---|
| `on_ground` | Average hip Y > 0.85 (hips near bottom of frame) |
| `crouching` | Average hip Y > 0.70 |
| `arms_raised` | Both wrists above both shoulders |
| `waving` | Exactly one wrist above its shoulder |
| `standing` | Default |

MediaPipe is lazy-loaded with `model_complexity=0` for minimal latency.

#### `check_floor(depth)` — Floor Hazard Detection

Examines the bottom 30% of the depth image. Counts pixels with depth > 0 and < 500mm. If count exceeds 1000 pixels, floor is flagged as hazardous.

#### `CentroidTracker` — Cross-Frame ID Persistence

Maintains stable object IDs across frames using bbox-center proximity matching:
- Computes Euclidean distance matrix between previous and current centroids
- Greedily assigns matches (closest first), max 150px match threshold
- Unmatched old IDs increment a disappearance counter (dropped after 10 frames)
- Unmatched new detections get fresh IDs
- Result: LLM modifiers and force history apply to the correct physical objects

#### `perception_loop()` — Main Perception Thread

Per-frame pipeline:
1. Capture RGB + depth from Orbbec camera
2. Run YOLO detection → list of `SceneObject`
3. Update centroid tracker → stable IDs
4. For each person: crop bbox region, run MediaPipe pose → assign `obj.pose`
5. Check floor hazard → apply EMA smoothing (alpha=0.3) to prevent oscillation
6. Package into `SceneFrame`, write to `shared["scene"]`

---

### `force_field.py` — Potential Field Navigation Engine

#### Force Computation (`compute()`)

Every render frame (30 FPS), computes net force and integrates to velocity/position:

**1. Goal Attraction (constant forward pull)**
```
f_goal = GOAL_STRENGTH (3.0)
```
If floor is not clear: `f_goal *= 0.1` (90% reduction — near-stop for floor hazards)

**2. Per-Object Repulsion (inverse-square)**
```
f = (-REPULSION_STRENGTH / dist^2) * size_factor * angle_factor * llm_scale
```

Where:
- `dist` = object distance in meters, clamped to MIN_DISTANCE (0.3m) to prevent singularity
- `size_factor` = `max(sqrt(bbox_area / image_area), 0.2)` — larger objects repel more, sqrt prevents small-object crushing
- `angle_factor` = `max(0, cos(angle))` — objects dead-ahead have full effect, peripheral objects fade to zero
- `llm_scale` = `ForceModifier.repulsion_scale` (default 1.0, range 0.0–5.0)

**3. Physics Integration**
```
acceleration = net_force / ROBOT_MASS
velocity += acceleration * dt
velocity *= e^(-DRAG * dt)          # framerate-independent exponential drag
velocity = clamp(velocity, -MAX_VELOCITY, MAX_VELOCITY)
position += velocity * dt
```

**4. Equilibrium Escape**

When velocity stays below 0.1 for 3 consecutive seconds with obstacles present, the robot nudges backward (velocity = -0.6) to break the equilibrium deadlock, then the stall timer resets.

#### `ForceModifier` — LLM-Assigned Context

Each modifier applies to one tracked object:
- `repulsion_scale: float` — Multiplier on repulsion (0 = ignore, 1 = normal, 5 = extreme danger)
- `label: str` — Semantic tag for UI display (`"threat"`, `"waving_through"`, `"injured"`)
- `speak_text: str` — Message the robot displays (shown as `SPEAK: ...` action)

---

### `llm_context.py` — Semantic Scene Interpretation

#### Pipeline

1. Builds a JSON snapshot of all detected objects (class, distance, position, pose, floor hazard)
2. Maintains a sliding window of last 3 snapshots for temporal context
3. Sends to Gemma:2b via Ollama with a system prompt defining response semantics
4. Parses JSON response, creates `ForceModifier` for each object
5. Writes modifiers to `ForceField.modifiers` dict

#### LLM Prompt Guidelines

The system prompt instructs the LLM to interpret poses:
- Person waving → `repulsion_scale: 0.1` (nearly ignore — let robot pass)
- Person blocking → `repulsion_scale: 1.0` (normal obstacle)
- Person fallen/injured → `repulsion_scale: 2.0` + speak text
- Threatening posture → `repulsion_scale: 3.0–5.0` + speak text
- Floor hazard → `repulsion_scale: 1.5`
- Peripheral object → `repulsion_scale: 0.5`

#### Fault Tolerance

- If Ollama is not installed: module imports gracefully, `interpret()` becomes a no-op
- If LLM call fails: exception is caught, logged, robot continues with default forces
- The `--no-llm` flag skips the LLM thread entirely

---

### `ui.py` — Pygame Visualization (3-Panel Layout)

**Left panel (320px):** Depth camera feed (JET colormap) with bounding box overlays. Red boxes for `arms_raised`/`on_ground` poses, green for others. Labels show class name + distance.

**Center panel:** Force field visualization. Robot shown as colored square (green = forward, red = retreating, yellow = waiting). Net force arrow above. Object dots along track. Velocity readout. Action label on robot, reason below.

**Right panel (240px):** Decision log showing action transitions with reasons.

---

## Navigation Behavior Summary

| Scenario | Behavior |
|---|---|
| Clear path | Accelerates forward to ~3.0 velocity, drag-limited |
| Person at 1.2m, center | Repulsion counters goal — slows to ~0.4 velocity |
| Person at 0.5m, center | Strong repulsion — retreats (negative velocity) |
| Person at 2.0m, side | Minimal effect — cos(angle) reduces repulsion |
| Floor hazard | Goal reduced 90% — near-stop regardless of obstacles |
| LLM: waving through | repulsion_scale=0.1 — robot passes person |
| LLM: threat detected | repulsion_scale=3-5 — aggressive retreat |
| LLM: person fallen | repulsion_scale=2.0 — cautious, displays message |
| Equilibrium deadlock | After 3s stall, nudges backward to break free |
| Multiple close objects | Repulsions sum — crowd causes full stop or retreat |

---

## Test Coverage

**122 tests, 0 failures.** All tests use real YOLO and MediaPipe models — no mocks.

### `test_perception.py` — 58 Tests

| Suite | Count | Coverage |
|---|---|---|
| YOLO Loading | 3 | Singleton loading, model names |
| Object Detection | 9 | Blank/real images, depth conversion, angle math, multi-object, confidence filter |
| MediaPipe Loading | 2 | Singleton loading |
| Pose Inference | 5 | Blank/noise/real images, valid pose strings |
| Floor Check | 7 | Clear/hazard/boundary/zero-depth edge cases |
| Scene Structures | 5 | SceneObject/SceneFrame creation, field assignment |
| Perception Loop | 4 | Shared state population, depth visualization, stop event, pose assignment |
| Camera Class | 3 | Init behavior, method signature |
| Edge Cases | 6 | Tiny/large images, wrong channels, max depth, single pixel |
| Performance | 3 | YOLO <500ms, MediaPipe <200ms, floor check <10ms |
| Centroid Tracker | 6 | ID assignment, persistence, new objects, disappearance, empty frames |
| Depth Sampling | 2 | Median sampling, zero-depth region handling |
| Force Field Fixes | 3 | Floor hazard integration, equilibrium escape, stall reset |

### `test_navigation.py` — 64 Tests

| Suite | Count | Coverage |
|---|---|---|
| Clear Path | 5 | Forward motion, positive velocity, net force, display states |
| Person Blocking | 3 | Reduced velocity, repulsion presence, reason display |
| Distance Response | 3 | Inverse-square verification across 0.5m, 1.5m, 3.0m |
| Angle Response | 3 | Center vs peripheral vs behind (cos factor) |
| Floor Hazard | 4 | Goal reduction, velocity decrease, display reason |
| Threat Response | 3 | High repulsion_scale increases retreat |
| Waving Through | 2 | Low repulsion_scale allows passage |
| Person Fallen | 2 | Modifier effect, speak text display |
| Multiple Objects | 3 | Additive repulsion, crowd full-stop, mixed objects |
| Equilibrium Escape | 3 | Stall trigger, no false triggers, counter reset |
| Dynamic Environment | 5 | Appear/disappear, approach/recede, hazard mid-motion, state transitions |
| All Environments | 9 | Every scenario produces valid physics output |
| LLM Modifier Lifecycle | 4 | Apply, remove, scale=0 ignore, wrong-ID no-effect |
| Bbox Size Effect | 2 | Larger bbox → more repulsion, tiny bbox minimum |
| Stress Tests | 8 | 20 objects, min/sub-min distance, rapid changes, zero scene, extreme dt, long sim |
| Pipeline Invariants | 5 | Velocity bounds, no NaN, valid display strings, goal force always present |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pyorbbecsdk` | latest | Orbbec camera RGB+depth capture |
| `ultralytics` | >=8.0 | YOLOv8-nano object detection |
| `mediapipe` | 0.10.18 | Pose landmark detection |
| `opencv-python` | >=4.0 | Image processing, colormap |
| `numpy` | >=1.24 | Array operations |
| `pygame` | >=2.0 | UI rendering |
| `ollama` | latest | LLM inference (optional) |
| `protobuf` | 4.x | Required by MediaPipe |

---

## Running

```bash
# Full system (camera + LLM)
python main.py

# Camera only, no LLM context
python main.py --no-llm

# Run all tests
SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())") \
    python3 -m pytest test_perception.py test_navigation.py -v
```
