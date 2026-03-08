from __future__ import annotations
import math
import time
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (
    YOLO_CONFIDENCE, YOLO_MODEL, FLOOR_ANOMALY_THRESHOLD, FLOOR_CLOSE_MM,
    CAMERA_WIDTH, CAMERA_HEIGHT, MEDIAPIPE_COMPLEXITY, HFOV_RAD,
    DEPTH_SAMPLE_RADIUS, DEPTH_FALLBACK_MM, TRACKER_MAX_DISAPPEARED,
    FLOOR_HAZARD_SMOOTH,
)


# ---------------------------------------------------------------------------
# Data structures (used by every other module)
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    id: str                             # tracking id
    obj_class: str                      # "person", "bottle", …
    distance_m: float                   # depth at bbox center (meters)
    angle: float                        # horizontal angle from centre (radians)
    bbox: tuple[int, int, int, int]     # cx, cy, w, h
    pose: str | None = None             # persons only
    floor_hazard: bool = False


@dataclass
class SceneFrame:
    objects: list[SceneObject]
    floor_clear: bool
    timestamp: float
    rgb: np.ndarray | None = None
    depth_colorized: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Camera (Orbbec) — filled in at integration step
# ---------------------------------------------------------------------------

class Camera:
    """Orbbec depth camera wrapper using pyorbbecsdk."""

    def __init__(self):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
        self.pipeline = Pipeline()
        config = Config()
        # Enable color stream (640x480, RGB)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profiles.get_video_stream_profile(
            CAMERA_WIDTH, CAMERA_HEIGHT, OBFormat.RGB888, 30
        )
        config.enable_stream(color_profile)
        # Enable depth stream (640x480)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_video_stream_profile(
            CAMERA_WIDTH, CAMERA_HEIGHT, OBFormat.Y16, 30
        )
        config.enable_stream(depth_profile)
        # Align depth to color
        config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.start(config)

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        frameset = self.pipeline.wait_for_frames(100)  # 100ms timeout
        if frameset is None:
            return None, None
        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None
        rgb = np.frombuffer(
            color_frame.get_data(), dtype=np.uint8
        ).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 3))
        depth = np.frombuffer(
            depth_frame.get_data(), dtype=np.uint16
        ).reshape((CAMERA_HEIGHT, CAMERA_WIDTH))
        return rgb, depth


# ---------------------------------------------------------------------------
# Detection helpers — filled in at integration step (YOLO + MediaPipe)
# ---------------------------------------------------------------------------

_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def detect_objects(rgb: np.ndarray, depth: np.ndarray) -> list[SceneObject]:
    """YOLOv8-nano object detection with depth-based distance and angle."""
    model = _get_yolo()
    results = model(rgb, conf=YOLO_CONFIDENCE, verbose=False)[0]
    objects = []
    h_img, w_img = rgb.shape[:2]
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        cls_name = model.names[int(box.cls[0])]
        # median depth in a small region around bbox center (robust to noise)
        r = DEPTH_SAMPLE_RADIUS
        region = depth[max(0, cy - r):min(h_img, cy + r),
                       max(0, cx - r):min(w_img, cx + r)]
        valid = region[region > 0]
        d_mm = float(np.median(valid)) if len(valid) > 0 else DEPTH_FALLBACK_MM
        distance_m = d_mm / 1000.0
        # horizontal angle from image center
        angle = ((cx / w_img) - 0.5) * HFOV_RAD
        objects.append(SceneObject(
            id=f"{cls_name}_{i}",
            obj_class=cls_name,
            distance_m=distance_m,
            angle=angle,
            bbox=(cx, cy, w, h),
        ))
    return objects


_mp_pose = None


def _get_pose():
    global _mp_pose
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_COMPLEXITY,
            min_detection_confidence=0.5,
        )
    return _mp_pose


def infer_pose(rgb: np.ndarray) -> str | None:
    """MediaPipe pose inference — classifies standing/crouching/on_ground/arms_raised/waving."""
    pose = _get_pose()
    result = pose.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    # Key landmark Y-coordinates (normalized 0-1, top=0)
    l_hip_y, r_hip_y = lm[23].y, lm[24].y
    l_wrist_y, r_wrist_y = lm[15].y, lm[16].y
    l_shoulder_y, r_shoulder_y = lm[11].y, lm[12].y
    hip_avg = (l_hip_y + r_hip_y) / 2

    if hip_avg > 0.85:
        return "on_ground"
    if hip_avg > 0.7:
        return "crouching"
    if l_wrist_y < l_shoulder_y and r_wrist_y < r_shoulder_y:
        return "arms_raised"
    if (l_wrist_y < l_shoulder_y) != (r_wrist_y < r_shoulder_y):
        return "waving"
    return "standing"


def check_floor(depth: np.ndarray) -> bool:
    """Floor hazard check via depth thresholding."""
    floor_region = depth[int(depth.shape[0] * 0.7):, :]
    anomaly_count = int(np.count_nonzero((floor_region > 0) & (floor_region < FLOOR_CLOSE_MM)))
    return anomaly_count < FLOOR_ANOMALY_THRESHOLD


# ---------------------------------------------------------------------------
# Simple centroid tracker — preserves IDs across frames
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Matches detections across frames by bbox-center proximity."""

    def __init__(self, max_disappeared: int = TRACKER_MAX_DISAPPEARED):
        self._next_id: int = 0
        self._centroids: dict[str, tuple[int, int]] = {}   # id → (cx, cy)
        self._classes: dict[str, str] = {}                  # id → class name
        self._disappeared: dict[str, int] = {}
        self._max_disappeared = max_disappeared

    def update(self, objects: list[SceneObject]) -> list[SceneObject]:
        """Assign stable IDs to objects. Modifies objects in-place and returns them."""
        if not objects:
            # mark all existing as disappeared
            for oid in list(self._disappeared):
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self._max_disappeared:
                    self._deregister(oid)
            return objects

        new_centroids = [(o.bbox[0], o.bbox[1]) for o in objects]

        if not self._centroids:
            # register all
            for i, obj in enumerate(objects):
                self._register(obj, new_centroids[i])
            return objects

        # match existing IDs to new detections by minimum distance
        old_ids = list(self._centroids.keys())
        old_pts = [self._centroids[oid] for oid in old_ids]

        # compute distance matrix
        used_new = set()
        used_old = set()
        pairs = []
        for oi, (ox, oy) in enumerate(old_pts):
            for ni, (nx, ny) in enumerate(new_centroids):
                d = math.hypot(ox - nx, oy - ny)
                pairs.append((d, oi, ni))
        pairs.sort()

        for d, oi, ni in pairs:
            if oi in used_old or ni in used_new:
                continue
            if d > 150:  # max match distance in pixels
                break
            used_old.add(oi)
            used_new.add(ni)
            oid = old_ids[oi]
            objects[ni].id = oid
            self._centroids[oid] = new_centroids[ni]
            self._classes[oid] = objects[ni].obj_class
            self._disappeared[oid] = 0

        # mark unmatched old IDs as disappeared
        for oi in range(len(old_ids)):
            if oi not in used_old:
                oid = old_ids[oi]
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self._max_disappeared:
                    self._deregister(oid)

        # register unmatched new detections
        for ni in range(len(objects)):
            if ni not in used_new:
                self._register(objects[ni], new_centroids[ni])

        return objects

    def _register(self, obj: SceneObject, centroid: tuple[int, int]):
        oid = f"{obj.obj_class}_{self._next_id}"
        self._next_id += 1
        obj.id = oid
        self._centroids[oid] = centroid
        self._classes[oid] = obj.obj_class
        self._disappeared[oid] = 0

    def _deregister(self, oid: str):
        del self._centroids[oid]
        del self._classes[oid]
        del self._disappeared[oid]


# ---------------------------------------------------------------------------
# Perception thread
# ---------------------------------------------------------------------------

def perception_loop(shared: dict, camera: Camera, stop: threading.Event):
    tracker = CentroidTracker()
    floor_score = 0.0  # EMA-smoothed floor hazard score

    while not stop.is_set():
        rgb, depth = camera.get_frames()
        if rgb is None or depth is None:
            time.sleep(0.01)
            continue

        objects = detect_objects(rgb, depth)
        objects = tracker.update(objects)

        # per-person pose inference (crop each person's bbox)
        for obj in objects:
            if obj.obj_class == "person":
                cx, cy, w, h = obj.bbox
                x1 = max(0, cx - w // 2)
                y1 = max(0, cy - h // 2)
                x2 = min(rgb.shape[1], x1 + w)
                y2 = min(rgb.shape[0], y1 + h)
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    obj.pose = infer_pose(crop)

        # smoothed floor hazard detection (EMA to prevent oscillation)
        raw_clear = check_floor(depth)
        floor_score = (1 - FLOOR_HAZARD_SMOOTH) * floor_score + FLOOR_HAZARD_SMOOTH * (0.0 if raw_clear else 1.0)
        floor_clear = floor_score < 0.5

        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET,
        )

        shared["scene"] = SceneFrame(
            objects=objects,
            floor_clear=floor_clear,
            timestamp=time.time(),
            rgb=rgb,
            depth_colorized=depth_vis,
        )
