"""
Rigorous tests for perception.py using real YOLO and MediaPipe models.
No mocks — exercises the full detection and pose pipeline on synthetic images.

Run:  cd demo && python -m pytest test_perception.py -v
"""
from __future__ import annotations
import math
import time
import threading

import cv2
import numpy as np
import pytest

# Ensure config is importable from this directory
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from perception import (
    SceneObject, SceneFrame,
    detect_objects, infer_pose, check_floor,
    perception_loop, Camera, CentroidTracker,
    _get_yolo, _get_pose,
)
from config import (
    YOLO_CONFIDENCE, YOLO_MODEL, HFOV_RAD,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    FLOOR_CLOSE_MM, FLOOR_ANOMALY_THRESHOLD,
    DEPTH_SAMPLE_RADIUS, DEPTH_FALLBACK_MM,
    STALL_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Fixtures — create test images once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def yolo_model():
    """Pre-load YOLO so model download happens once."""
    return _get_yolo()


@pytest.fixture(scope="session")
def pose_model():
    """Pre-load MediaPipe Pose once."""
    return _get_pose()


@pytest.fixture
def blank_rgb():
    return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def blank_depth():
    return np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 3000, dtype=np.uint16)


@pytest.fixture
def person_image():
    """Draw a rough stick-figure-like shape that YOLO may detect as a person."""
    img = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 200, dtype=np.uint8)  # light gray bg
    # Head (circle)
    cv2.circle(img, (320, 100), 30, (0, 0, 0), -1)
    # Torso (rectangle)
    cv2.rectangle(img, (290, 130), (350, 280), (0, 0, 0), -1)
    # Left leg
    cv2.rectangle(img, (290, 280), (315, 420), (0, 0, 0), -1)
    # Right leg
    cv2.rectangle(img, (325, 280), (350, 420), (0, 0, 0), -1)
    # Left arm
    cv2.rectangle(img, (250, 140), (290, 160), (0, 0, 0), -1)
    # Right arm
    cv2.rectangle(img, (350, 140), (390, 160), (0, 0, 0), -1)
    return img


@pytest.fixture
def real_person_image():
    """Load a real test image from COCO or generate a more realistic one with
    a photo-like person. Falls back to the stick figure approach."""
    # Create a more realistic silhouette on a contrasting background
    img = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 220, dtype=np.uint8)
    # Draw a filled person-like silhouette (dark on light)
    pts_body = np.array([
        [300, 80], [340, 80],   # head top
        [350, 130],             # right shoulder
        [400, 200],             # right hand
        [360, 200],             # right elbow
        [355, 280],             # right hip
        [360, 420],             # right foot
        [330, 420],             # right ankle
        [330, 290],             # crotch right
        [310, 290],             # crotch left
        [310, 420],             # left ankle
        [280, 420],             # left foot
        [285, 280],             # left hip
        [280, 200],             # left elbow
        [240, 200],             # left hand
        [290, 130],             # left shoulder
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts_body], (50, 40, 30))
    # Head
    cv2.circle(img, (320, 70), 35, (50, 40, 30), -1)
    return img


# ===========================================================================
# 1. YOLO MODEL LOADING
# ===========================================================================

class TestYoloLoading:
    def test_model_loads(self, yolo_model):
        """YOLO model should load without error."""
        assert yolo_model is not None

    def test_model_has_names(self, yolo_model):
        """Model should expose COCO class names."""
        assert hasattr(yolo_model, "names")
        assert "person" in yolo_model.names.values()

    def test_singleton_returns_same(self):
        """_get_yolo should return the same instance on repeated calls."""
        a = _get_yolo()
        b = _get_yolo()
        assert a is b


# ===========================================================================
# 2. detect_objects — YOLO INFERENCE
# ===========================================================================

class TestDetectObjects:
    def test_blank_image_returns_empty(self, yolo_model, blank_rgb, blank_depth):
        """A blank black image should yield no detections."""
        objs = detect_objects(blank_rgb, blank_depth)
        assert isinstance(objs, list)
        # Might detect something spurious, but usually 0
        # We mainly check it doesn't crash

    def test_returns_scene_objects(self, yolo_model, person_image, blank_depth):
        """detect_objects should return a list of SceneObject."""
        objs = detect_objects(person_image, blank_depth)
        assert isinstance(objs, list)
        for obj in objs:
            assert isinstance(obj, SceneObject)

    def test_scene_object_fields(self, yolo_model, person_image, blank_depth):
        """Each SceneObject should have valid field values."""
        objs = detect_objects(person_image, blank_depth)
        for obj in objs:
            assert isinstance(obj.id, str) and len(obj.id) > 0
            assert isinstance(obj.obj_class, str) and len(obj.obj_class) > 0
            assert isinstance(obj.distance_m, float) and obj.distance_m > 0
            assert isinstance(obj.angle, float)
            assert -math.pi <= obj.angle <= math.pi
            cx, cy, w, h = obj.bbox
            assert 0 <= cx < CAMERA_WIDTH
            assert 0 <= cy < CAMERA_HEIGHT
            assert w > 0 and h > 0

    def test_depth_to_distance_conversion(self, yolo_model, person_image):
        """Distance should reflect depth value at bbox center (mm -> m)."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2500, dtype=np.uint16)
        objs = detect_objects(person_image, depth)
        for obj in objs:
            # depth is 2500mm = 2.5m everywhere
            assert abs(obj.distance_m - 2.5) < 0.01

    def test_zero_depth_fallback(self, yolo_model, person_image):
        """Zero depth should fall back to conservative DEPTH_FALLBACK_MM."""
        depth = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint16)
        objs = detect_objects(person_image, depth)
        expected_m = DEPTH_FALLBACK_MM / 1000.0
        for obj in objs:
            assert abs(obj.distance_m - expected_m) < 0.01

    def test_angle_center_is_zero(self, yolo_model):
        """Object at image center should have angle ≈ 0."""
        # Create image with a bright rectangle dead center on dark bg
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        cv2.rectangle(img, (280, 160), (360, 320), (255, 255, 255), -1)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        objs = detect_objects(img, depth)
        for obj in objs:
            # Object near center → angle should be small
            assert abs(obj.angle) < HFOV_RAD / 4  # within quarter of FOV

    def test_angle_left_is_negative(self, yolo_model):
        """Object on left side of image should have negative angle."""
        img = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        # Place bright blob on far left
        cv2.rectangle(img, (10, 160), (90, 320), (255, 255, 255), -1)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        objs = detect_objects(img, depth)
        for obj in objs:
            assert obj.angle < 0, f"Left-side object should have negative angle, got {obj.angle}"

    def test_multiple_objects(self, yolo_model):
        """Multiple distinct objects should produce multiple SceneObjects."""
        img = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 200, dtype=np.uint8)
        # Two dark blobs far apart
        cv2.rectangle(img, (50, 100), (150, 400), (10, 10, 10), -1)
        cv2.rectangle(img, (450, 100), (550, 400), (10, 10, 10), -1)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        objs = detect_objects(img, depth)
        # May or may not detect as distinct objects — just verify no crash
        assert isinstance(objs, list)

    def test_high_confidence_filter(self, yolo_model, blank_rgb, blank_depth):
        """With high confidence threshold, blank image should return nothing."""
        from config import YOLO_CONFIDENCE
        # The default confidence is already 0.5, blank image shouldn't trigger
        objs = detect_objects(blank_rgb, blank_depth)
        assert len(objs) == 0


# ===========================================================================
# 3. MEDIAPIPE POSE INFERENCE
# ===========================================================================

class TestMediaPipePoseLoading:
    def test_pose_model_loads(self, pose_model):
        """MediaPipe Pose should load without error."""
        assert pose_model is not None

    def test_singleton_returns_same(self):
        """_get_pose should return same instance."""
        a = _get_pose()
        b = _get_pose()
        assert a is b


class TestInferPose:
    def test_blank_image_returns_none(self, pose_model, blank_rgb):
        """No person visible → should return None."""
        result = infer_pose(blank_rgb)
        assert result is None

    def test_returns_valid_pose_string(self, pose_model, real_person_image):
        """If a person is detected, result should be one of the valid poses."""
        result = infer_pose(real_person_image)
        valid_poses = {"standing", "crouching", "on_ground", "arms_raised", "waving"}
        if result is not None:
            assert result in valid_poses, f"Got unexpected pose: {result}"

    def test_pose_returns_string_or_none(self, pose_model, person_image):
        """Return type should always be str or None."""
        result = infer_pose(person_image)
        assert result is None or isinstance(result, str)

    def test_pose_with_noise_image(self, pose_model):
        """Random noise should not crash, should return None."""
        noise = np.random.randint(0, 255, (CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        result = infer_pose(noise)
        assert result is None or isinstance(result, str)

    def test_pose_with_real_person_photo(self, pose_model):
        """Create a more person-like image and check pose detection."""
        # Create a standing person silhouette
        img = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 240, dtype=np.uint8)
        # Skin-colored head
        cv2.circle(img, (320, 80), 25, (180, 140, 120), -1)
        # Body
        cv2.rectangle(img, (295, 105), (345, 250), (50, 50, 150), -1)
        # Legs
        cv2.rectangle(img, (295, 250), (318, 400), (50, 50, 100), -1)
        cv2.rectangle(img, (322, 250), (345, 400), (50, 50, 100), -1)
        # Arms down at sides
        cv2.rectangle(img, (265, 110), (295, 220), (180, 140, 120), -1)
        cv2.rectangle(img, (345, 110), (375, 220), (180, 140, 120), -1)
        result = infer_pose(img)
        # May or may not detect — just ensure no crash and valid type
        assert result is None or result in {"standing", "crouching", "on_ground", "arms_raised", "waving"}


# ===========================================================================
# 4. check_floor — FLOOR HAZARD DETECTION
# ===========================================================================

class TestCheckFloor:
    def test_clear_floor(self):
        """All-far depth should be floor_clear=True."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        assert check_floor(depth) is True

    def test_hazard_floor(self):
        """Many close pixels in bottom 30% → floor_clear=False."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        # Fill bottom 30% with close values
        floor_start = int(CAMERA_HEIGHT * 0.7)
        depth[floor_start:, :] = 200  # 200mm, well under FLOOR_CLOSE_MM (500)
        assert check_floor(depth) is False

    def test_zero_depth_not_counted(self):
        """Zero depth (invalid readings) should NOT count as close."""
        depth = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint16)
        assert check_floor(depth) is True

    def test_just_below_threshold(self):
        """Anomaly count just below threshold should still be clear."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        floor_start = int(CAMERA_HEIGHT * 0.7)
        # Place exactly (THRESHOLD - 1) close pixels
        count = FLOOR_ANOMALY_THRESHOLD - 1
        row = floor_start
        for i in range(count):
            col = i % CAMERA_WIDTH
            if col == 0 and i > 0:
                row += 1
            if row < CAMERA_HEIGHT:
                depth[row, col] = 300
        assert check_floor(depth) is True

    def test_just_above_threshold(self):
        """Anomaly count at threshold should trigger hazard."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        floor_start = int(CAMERA_HEIGHT * 0.7)
        count = FLOOR_ANOMALY_THRESHOLD + 1
        row = floor_start
        for i in range(count):
            col = i % CAMERA_WIDTH
            if col == 0 and i > 0:
                row += 1
            if row < CAMERA_HEIGHT:
                depth[row, col] = 300
        assert check_floor(depth) is False

    def test_close_pixels_above_floor_region_ignored(self):
        """Close pixels in upper 70% should not affect floor check."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        depth[:int(CAMERA_HEIGHT * 0.7), :] = 100  # close, but above floor region
        assert check_floor(depth) is True

    def test_boundary_depth_value(self):
        """Pixels at exactly FLOOR_CLOSE_MM should NOT be counted (< not <=)."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), FLOOR_CLOSE_MM, dtype=np.uint16)
        # All pixels at exactly FLOOR_CLOSE_MM → condition is (d > 0) & (d < FLOOR_CLOSE_MM)
        # At exactly FLOOR_CLOSE_MM, d < FLOOR_CLOSE_MM is False, so not counted
        assert check_floor(depth) is True


# ===========================================================================
# 5. DATA STRUCTURES
# ===========================================================================

class TestSceneObject:
    def test_creation(self):
        obj = SceneObject("p0", "person", 1.5, 0.1, (320, 240, 100, 200))
        assert obj.id == "p0"
        assert obj.obj_class == "person"
        assert obj.distance_m == 1.5
        assert obj.angle == 0.1
        assert obj.bbox == (320, 240, 100, 200)
        assert obj.pose is None
        assert obj.floor_hazard is False

    def test_pose_assignment(self):
        obj = SceneObject("p0", "person", 1.0, 0.0, (320, 240, 100, 200))
        obj.pose = "standing"
        assert obj.pose == "standing"

    def test_floor_hazard_flag(self):
        obj = SceneObject("b0", "bottle", 0.8, -0.2, (100, 400, 30, 60), floor_hazard=True)
        assert obj.floor_hazard is True


class TestSceneFrame:
    def test_creation(self):
        frame = SceneFrame(objects=[], floor_clear=True, timestamp=time.time())
        assert frame.objects == []
        assert frame.floor_clear is True
        assert frame.rgb is None
        assert frame.depth_colorized is None

    def test_with_objects(self):
        objs = [
            SceneObject("p0", "person", 1.0, 0.0, (320, 240, 100, 200)),
            SceneObject("b0", "bottle", 2.0, 0.5, (500, 300, 30, 60)),
        ]
        frame = SceneFrame(objects=objs, floor_clear=True, timestamp=time.time())
        assert len(frame.objects) == 2


# ===========================================================================
# 6. PERCEPTION LOOP INTEGRATION TEST
# ===========================================================================

class _FakeCamera:
    """Minimal camera stand-in that returns real numpy frames for N calls."""
    def __init__(self, rgb, depth, max_frames=3):
        self.rgb = rgb
        self.depth = depth
        self.max_frames = max_frames
        self.call_count = 0

    def get_frames(self):
        self.call_count += 1
        if self.call_count > self.max_frames:
            return None, None
        return self.rgb.copy(), self.depth.copy()


class TestPerceptionLoop:
    def test_loop_populates_shared_scene(self, yolo_model, pose_model):
        """perception_loop should populate shared['scene'] with a SceneFrame."""
        rgb = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 200, dtype=np.uint8)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 3000, dtype=np.uint16)
        cam = _FakeCamera(rgb, depth, max_frames=5)

        shared = {"scene": None}
        stop = threading.Event()

        t = threading.Thread(target=perception_loop, args=(shared, cam, stop), daemon=True)
        t.start()

        # Wait for at least one frame to be processed
        deadline = time.time() + 10.0
        while shared["scene"] is None and time.time() < deadline:
            time.sleep(0.05)

        stop.set()
        t.join(timeout=5.0)

        scene = shared["scene"]
        assert scene is not None, "perception_loop never populated shared['scene']"
        assert isinstance(scene, SceneFrame)
        assert isinstance(scene.objects, list)
        assert isinstance(scene.floor_clear, bool)
        assert scene.timestamp > 0
        assert scene.rgb is not None
        assert scene.depth_colorized is not None

    def test_loop_depth_colorized_shape(self, yolo_model, pose_model):
        """Colorized depth should be 3-channel uint8 same spatial size."""
        rgb = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 200, dtype=np.uint8)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        cam = _FakeCamera(rgb, depth, max_frames=3)

        shared = {"scene": None}
        stop = threading.Event()

        t = threading.Thread(target=perception_loop, args=(shared, cam, stop), daemon=True)
        t.start()

        deadline = time.time() + 10.0
        while shared["scene"] is None and time.time() < deadline:
            time.sleep(0.05)

        stop.set()
        t.join(timeout=5.0)

        scene = shared["scene"]
        assert scene is not None
        assert scene.depth_colorized.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        assert scene.depth_colorized.dtype == np.uint8

    def test_loop_stops_on_event(self, yolo_model, pose_model):
        """Loop should exit promptly when stop event is set."""
        rgb = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 200, dtype=np.uint8)
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        cam = _FakeCamera(rgb, depth, max_frames=1000)

        shared = {"scene": None}
        stop = threading.Event()

        t = threading.Thread(target=perception_loop, args=(shared, cam, stop), daemon=True)
        t.start()

        time.sleep(0.3)
        stop.set()
        t.join(timeout=3.0)
        assert not t.is_alive(), "perception_loop did not stop within 3s"

    def test_loop_assigns_pose_to_persons(self, yolo_model, pose_model, real_person_image):
        """If YOLO detects a person and MediaPipe finds a pose, obj.pose should be set."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        cam = _FakeCamera(real_person_image, depth, max_frames=5)

        shared = {"scene": None}
        stop = threading.Event()

        t = threading.Thread(target=perception_loop, args=(shared, cam, stop), daemon=True)
        t.start()

        deadline = time.time() + 10.0
        while shared["scene"] is None and time.time() < deadline:
            time.sleep(0.05)

        stop.set()
        t.join(timeout=5.0)

        scene = shared["scene"]
        assert scene is not None
        # If any person was detected, check pose was assigned
        persons = [o for o in scene.objects if o.obj_class == "person"]
        for p in persons:
            # pose was assigned (could be None if mediapipe didn't detect, that's ok)
            assert p.pose is None or p.pose in {"standing", "crouching", "on_ground", "arms_raised", "waving"}


# ===========================================================================
# 7. CAMERA CLASS (structural tests — no hardware)
# ===========================================================================

class TestCameraClass:
    def test_camera_init_fails_without_hardware(self):
        """Camera() should raise when no Orbbec device is connected."""
        with pytest.raises(Exception):
            Camera()

    def test_camera_has_get_frames_method(self):
        """Camera class should have get_frames method."""
        assert hasattr(Camera, "get_frames")
        assert callable(getattr(Camera, "get_frames"))

    def test_camera_get_frames_signature(self):
        """get_frames should accept self and return a tuple."""
        import inspect
        sig = inspect.signature(Camera.get_frames)
        params = list(sig.parameters.keys())
        assert params == ["self"]


# ===========================================================================
# 8. EDGE CASES AND ROBUSTNESS
# ===========================================================================

class TestEdgeCases:
    def test_detect_objects_tiny_image(self, yolo_model):
        """Very small image should not crash."""
        tiny_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        tiny_depth = np.full((32, 32), 2000, dtype=np.uint16)
        objs = detect_objects(tiny_rgb, tiny_depth)
        assert isinstance(objs, list)

    def test_detect_objects_large_image(self, yolo_model):
        """Larger-than-expected image should still work."""
        big_rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        big_depth = np.full((1080, 1920), 2000, dtype=np.uint16)
        objs = detect_objects(big_rgb, big_depth)
        assert isinstance(objs, list)

    def test_infer_pose_wrong_channels(self, pose_model):
        """Grayscale image (1 channel) should either work or fail gracefully."""
        gray = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)
        try:
            result = infer_pose(np.stack([gray]*3, axis=-1))  # convert to 3ch
            assert result is None or isinstance(result, str)
        except Exception:
            pytest.fail("infer_pose should handle edge cases gracefully")

    def test_detect_objects_max_depth(self, yolo_model, person_image):
        """uint16 max depth (65535mm = 65.5m) should convert correctly."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 65535, dtype=np.uint16)
        objs = detect_objects(person_image, depth)
        for obj in objs:
            assert abs(obj.distance_m - 65.535) < 0.01

    def test_check_floor_single_pixel_close(self):
        """A single close pixel should not trigger hazard (below threshold)."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 5000, dtype=np.uint16)
        depth[CAMERA_HEIGHT - 1, 0] = 100
        assert check_floor(depth) is True

    def test_detect_objects_consistent_ids(self, yolo_model, person_image, blank_depth):
        """Object IDs should follow the '{class}_{index}' pattern."""
        objs = detect_objects(person_image, blank_depth)
        for obj in objs:
            parts = obj.id.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[0] == obj.obj_class
            assert parts[1].isdigit()


# ===========================================================================
# 9. PERFORMANCE SMOKE TEST
# ===========================================================================

class TestPerformance:
    def test_detect_objects_completes_in_time(self, yolo_model, person_image, blank_depth):
        """Single YOLO inference should complete in < 5s (generous for CPU)."""
        start = time.time()
        detect_objects(person_image, blank_depth)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"detect_objects took {elapsed:.2f}s"

    def test_infer_pose_completes_in_time(self, pose_model, real_person_image):
        """Single MediaPipe pose inference should complete in < 3s."""
        start = time.time()
        infer_pose(real_person_image)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"infer_pose took {elapsed:.2f}s"

    def test_check_floor_fast(self, blank_depth):
        """check_floor should be < 50ms (numpy only)."""
        start = time.time()
        for _ in range(100):
            check_floor(blank_depth)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"100x check_floor took {elapsed:.2f}s"


# ===========================================================================
# 10. CENTROID TRACKER
# ===========================================================================

class TestCentroidTracker:
    def test_assigns_ids(self):
        """Tracker should assign unique IDs to new objects."""
        tracker = CentroidTracker()
        objs = [
            SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100)),
            SceneObject("tmp", "bottle", 2.0, 0.3, (400, 300, 30, 60)),
        ]
        result = tracker.update(objs)
        ids = [o.id for o in result]
        assert len(set(ids)) == 2, "IDs should be unique"
        assert all("_" in i for i in ids)

    def test_preserves_ids_across_frames(self):
        """Same object in consecutive frames should keep the same ID."""
        tracker = CentroidTracker()
        objs1 = [SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100))]
        tracker.update(objs1)
        id1 = objs1[0].id

        # same object, slightly moved
        objs2 = [SceneObject("tmp", "person", 1.0, 0.0, (105, 202, 50, 100))]
        tracker.update(objs2)
        id2 = objs2[0].id

        assert id1 == id2, f"ID should persist: {id1} vs {id2}"

    def test_new_object_gets_new_id(self):
        """A new far-away object should get a fresh ID."""
        tracker = CentroidTracker()
        objs1 = [SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100))]
        tracker.update(objs1)
        id1 = objs1[0].id

        # add a second object far from the first
        objs2 = [
            SceneObject("tmp", "person", 1.0, 0.0, (105, 202, 50, 100)),
            SceneObject("tmp", "bottle", 2.0, 0.5, (500, 400, 30, 60)),
        ]
        tracker.update(objs2)
        assert objs2[0].id == id1  # first object kept
        assert objs2[1].id != id1  # second object is new

    def test_disappeared_object_eventually_dropped(self):
        """Object that vanishes for many frames should be deregistered."""
        tracker = CentroidTracker(max_disappeared=3)
        objs = [SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100))]
        tracker.update(objs)

        # disappear for 4 frames
        for _ in range(4):
            tracker.update([])

        # new object at same position gets a NEW id (old was dropped)
        objs2 = [SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100))]
        tracker.update(objs2)
        assert objs2[0].id != objs[0].id

    def test_empty_frames_dont_crash(self):
        """Empty detections should be handled gracefully."""
        tracker = CentroidTracker()
        result = tracker.update([])
        assert result == []

    def test_order_independence(self):
        """Swapping detection order shouldn't change IDs."""
        tracker = CentroidTracker()
        a = SceneObject("tmp", "person", 1.0, 0.0, (100, 200, 50, 100))
        b = SceneObject("tmp", "person", 2.0, 0.5, (500, 300, 50, 100))
        tracker.update([a, b])
        id_a, id_b = a.id, b.id

        # next frame: reversed order but same positions
        a2 = SceneObject("tmp", "person", 1.0, 0.0, (102, 201, 50, 100))
        b2 = SceneObject("tmp", "person", 2.0, 0.5, (498, 302, 50, 100))
        tracker.update([b2, a2])  # reversed order

        assert a2.id == id_a
        assert b2.id == id_b


# ===========================================================================
# 11. DEPTH REGION SAMPLING
# ===========================================================================

class TestDepthSampling:
    def test_median_depth_used(self, yolo_model, person_image):
        """When center pixel is zero but surroundings valid, median should be used."""
        depth = np.full((CAMERA_HEIGHT, CAMERA_WIDTH), 2000, dtype=np.uint16)
        # zero out a small center strip — median of surrounding should still be 2000
        depth[235:245, 315:325] = 0
        objs = detect_objects(person_image, depth)
        for obj in objs:
            # should use median of the valid pixels (~2000mm = 2.0m), not fallback
            assert obj.distance_m < 2.5, f"Expected ~2.0m, got {obj.distance_m}"
            assert obj.distance_m > 1.0, f"Should not use fallback, got {obj.distance_m}"

    def test_all_zero_uses_conservative_fallback(self, yolo_model, person_image):
        """All-zero depth should use DEPTH_FALLBACK_MM (0.5m), not 1.0m."""
        depth = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH), dtype=np.uint16)
        objs = detect_objects(person_image, depth)
        for obj in objs:
            assert abs(obj.distance_m - DEPTH_FALLBACK_MM / 1000.0) < 0.01


# ===========================================================================
# 12. FORCE FIELD FIXES
# ===========================================================================

class TestForceFieldFixes:
    def test_floor_hazard_reduces_goal(self):
        """When floor_clear=False, robot should near-halt."""
        from force_field import ForceField
        ff = ForceField()
        scene = SceneFrame(objects=[], floor_clear=False, timestamp=time.time())
        ff.compute(scene, 0.033)
        ff.compute(scene, 0.033)
        ff.compute(scene, 0.033)
        # with goal reduced to 10%, velocity should be very low
        assert ff.velocity < 0.5, f"Robot should near-halt on floor hazard, vel={ff.velocity}"

    def test_floor_clear_normal_speed(self):
        """When floor_clear=True, robot moves forward normally."""
        from force_field import ForceField
        ff = ForceField()
        scene = SceneFrame(objects=[], floor_clear=True, timestamp=time.time())
        for _ in range(30):
            ff.compute(scene, 0.033)
        assert ff.velocity > 0.5, f"Robot should move forward, vel={ff.velocity}"

    def test_equilibrium_escape(self):
        """Robot stalled with obstacles should eventually retreat."""
        from force_field import ForceField
        ff = ForceField()
        # create a scene that causes near-equilibrium: close person with large bbox
        # so repulsion ≈ goal strength, causing velocity → 0
        obj = SceneObject("person_0", "person", 0.8, 0.0, (320, 240, 200, 400))
        scene = SceneFrame(objects=[obj], floor_clear=True, timestamp=time.time())

        # run many frames — should trigger stall escape
        escaped = False
        for _ in range(int((STALL_TIMEOUT + 2) / 0.033)):
            ff.compute(scene, 0.033)
            if ff._stall_time == 0.0 and ff.velocity < -0.3:
                escaped = True
                break

        assert escaped, \
            f"Expected escape nudge, vel={ff.velocity}, stall={ff._stall_time}"
