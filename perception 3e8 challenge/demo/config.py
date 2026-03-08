# === Force Field ===
GOAL_STRENGTH = 3.0          # constant forward pull
REPULSION_STRENGTH = 8.0     # base repulsive force at 1m
ROBOT_MASS = 1.0             # higher = slower response
DRAG = 3.0                   # velocity drag coefficient (per-second, framerate-independent)
MAX_VELOCITY = 4.0           # max speed (normalized units/sec)
MIN_DISTANCE = 0.3           # clamp to avoid division blowup

# === Perception ===
YOLO_CONFIDENCE = 0.5
YOLO_MODEL = "yolov8n.pt"       # ~6MB, CPU-friendly
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MEDIAPIPE_COMPLEXITY = 0        # 0 = fastest (Pi-friendly)
FLOOR_ANOMALY_THRESHOLD = 1000  # pixel count
FLOOR_CLOSE_MM = 500            # millimeters
TRACKER_MAX_DISAPPEARED = 10    # frames before dropping tracked object
DEPTH_SAMPLE_RADIUS = 10        # pixels around bbox center for median depth
DEPTH_FALLBACK_MM = 500.0       # conservative fallback when depth invalid (0.5m)
STALL_TIMEOUT = 3.0             # seconds before equilibrium escape triggers
FLOOR_HAZARD_SMOOTH = 0.3       # EMA alpha for floor hazard smoothing
HFOV_RAD = 1.047                # ~60° horizontal FOV (Orbbec default)

# === LLM ===
LLM_MODEL = "gemma:2b"
LLM_HISTORY_SIZE = 3
LLM_LOOP_SLEEP = 0.3  # ~3Hz

# === UI ===
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 600
ROBOT_SIZE = 80
FPS = 30
