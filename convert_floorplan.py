#!/usr/bin/env python3
"""
Hotel Floor Plan → ROS Map Dataset Pipeline

Converts hotel floor plan images into ROS-compatible 2D occupancy grid maps
(PGM + YAML) with optional OCR-based waypoint extraction.

Usage:
    python convert_floorplan.py --source sources.csv
    python convert_floorplan.py --source /path/to/images/
    python convert_floorplan.py --image single_floorplan.png --output dataset/Hotel/floor_1
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml
from PIL import Image
from skimage.morphology import thin


def download_image(url, dest):
    """Download a floor plan image from a URL.

    Handles Wikimedia Commons file pages by extracting the direct image link.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FloorplanPipeline/1.0)"
    }

    # If it's a Wikimedia file page, extract the direct image URL via the API
    if "commons.wikimedia.org/wiki/File:" in url:
        filename = url.split("File:")[-1]
        api_url = (
            "https://commons.wikimedia.org/w/api.php"
            f"?action=query&titles=File:{filename}"
            "&prop=imageinfo&iiprop=url&format=json"
        )
        resp = requests.get(api_url, timeout=30, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        direct_url = page["imageinfo"][0]["url"]
    else:
        direct_url = url

    print(f"  Downloading: {direct_url}")
    resp = requests.get(direct_url, timeout=60, stream=True, headers=headers)
    resp.raise_for_status()

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"  Saved to: {dest}")
    return dest


def process_image(input_path, output_pgm_path, wall_sensitivity=3.0):
    """Process a floor plan image into a ROS-compatible PGM occupancy grid.

    Uses shape-based connected-component filtering to keep structural walls
    (long, elongated features) while removing text labels, dotted patterns,
    and other compact/small details.

    Pipeline:
        1. Load image; compute resolution-adaptive scale factor
        2. Color pre-filter — mask high-saturation pixels to white
        3. Convert to grayscale, apply scale-adaptive Gaussian blur
        4. Otsu threshold (THRESH_BINARY_INV so walls=255 for morph ops)
        5. Light morphological open — remove isolated dots (1-2px noise)
        6. Connected-component shape filter — keep components that are
           large enough OR elongated enough to be walls; reject small
           compact blobs (text characters, dot clusters)
        7. Morphological close — seal small wall gaps (doorways, joints)
        8. Invert to ROS convention (white=free/255, black=wall/0)
        9. Save as PGM (P5 format)

    Args:
        input_path: Path to the source floor plan image.
        output_pgm_path: Destination path for the output PGM file.
        wall_sensitivity: Controls minimum feature size for filtering
            (default 3.0). Lower keeps more features; higher removes more.

    Returns (width, height) of the output map.
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not load image: {input_path}")

    h, w = img.shape[:2]

    # 1. Resolution-adaptive scale factor
    scale = np.clip(max(w, h) / 1000.0, 0.5, 5.0)

    # 2. Color pre-filter: set high-saturation pixels to white so colored
    #    fills (e.g. cyan Oval Office) don't become false walls in grayscale
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > 80
    img[sat_mask] = [255, 255, 255]

    # 3. Grayscale + light blur (avoid heavy blur so thin lines stay sharp)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_k = max(1, int(1.5 * scale) | 1)  # minimal blur to keep thin lines
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # 4. Otsu threshold (BINARY_INV: walls=255, free=0 for morph processing)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── 4b. OCR text removal using directional morphology ────────────────
    # Within each OCR phrase box, find line content via directional opening
    # (lines survive, text characters don't). Erase everything in the box
    # EXCEPT the line pixels. This keeps walls, removes all text.
    phrase_boxes = []
    ocr_word_heights = []
    try:
        from ocr_visualize import extract_words, group_nearby_words
        words = extract_words(input_path, min_conf=20)
        ocr_word_heights = [int(wd["h"]) for wd in words if int(wd["h"]) > 5]
        phrases = group_nearby_words(words)
        pad = max(12, int(8 * scale))
        for wd in phrases:
            bx, by, bw, bh = int(wd["x"]), int(wd["y"]), int(wd["w"]), int(wd["h"])
            phrase_boxes.append((bx - pad, by - pad, bx + bw + pad, by + bh + pad))
    except Exception:
        pass

    opened = binary.copy()

    def _diag_kernel(length, direction):
        k = np.zeros((length, length), dtype=np.uint8)
        for ii in range(length):
            jj = ii if direction == 1 else length - 1 - ii
            k[ii, jj] = 1
        return k

    base_line_k = max(15, int(10 * scale)) | 1

    def _directional_lines(region, lk):
        """Find pixels surviving directional opening in H/V/diagonal directions."""
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (lk, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, lk))
        dk = _diag_kernel(lk, 1)
        dk2 = _diag_kernel(lk, -1)
        return cv2.bitwise_or(
            cv2.bitwise_or(
                cv2.morphologyEx(region, cv2.MORPH_OPEN, kh),
                cv2.morphologyEx(region, cv2.MORPH_OPEN, kv)),
            cv2.bitwise_or(
                cv2.morphologyEx(region, cv2.MORPH_OPEN, dk),
                cv2.morphologyEx(region, cv2.MORPH_OPEN, dk2)))

    # Per-box text removal uses multiple signals:
    #  A) Directional opening with base_line_k — identifies line segments
    #  B) Boundary connectivity — CCs touching box edge are walls
    #  C) CC shape analysis — compact small CCs are text, elongated ones are walls
    text_char_max = max(200, int(100 * scale * scale))

    for (bx1, by1, bx2, by2) in phrase_boxes:
        bx1c, by1c = max(0, bx1), max(0, by1)
        bx2c, by2c = min(w, bx2), min(h, by2)
        if bx2c <= bx1c or by2c <= by1c:
            continue
        box_slice = opened[by1c:by2c, bx1c:bx2c]
        if box_slice.sum() == 0:
            continue
        box_h, box_w = box_slice.shape

        # A) Directional opening — features surviving are wall-like lines
        margin = base_line_k + 2
        rx1, ry1 = max(0, bx1c - margin), max(0, by1c - margin)
        rx2, ry2 = min(w, bx2c + margin), min(h, by2c + margin)
        region = opened[ry1:ry2, rx1:rx2].copy()
        lines_in_region = _directional_lines(region, base_line_k)
        line_safe = cv2.dilate(lines_in_region, np.ones((3, 3), np.uint8))
        oy1 = by1c - ry1
        oy2 = oy1 + box_h
        ox1 = bx1c - rx1
        ox2 = ox1 + box_w
        safe_directional = line_safe[oy1:oy2, ox1:ox2]

        # B) Boundary connectivity — CCs touching box edge are walls
        n_box, box_labels, box_stats, _ = cv2.connectedComponentsWithStats(
            box_slice, connectivity=8)
        safe_boundary = np.zeros_like(box_slice)
        for bi in range(1, n_box):
            cc_mask = (box_labels == bi)
            touches_edge = (cc_mask[0, :].any() or cc_mask[-1, :].any() or
                            cc_mask[:, 0].any() or cc_mask[:, -1].any())
            if touches_edge:
                bbw = box_stats[bi, cv2.CC_STAT_WIDTH]
                bbh = box_stats[bi, cv2.CC_STAT_HEIGHT]
                bmax = max(bbw, bbh)
                bmin = max(min(bbw, bbh), 1)
                if bmax / bmin >= 2.5 or (cc_mask & (safe_directional > 0)).any():
                    safe_boundary[cc_mask] = 255

        combined_safe = cv2.bitwise_or(safe_directional, safe_boundary)

        # C) CC shape filter on remaining candidates — only erase compact/small CCs
        candidate_text = cv2.subtract(box_slice, combined_safe)
        if candidate_text.sum() == 0:
            continue
        n_tc, tc_labels, tc_stats, _ = cv2.connectedComponentsWithStats(
            candidate_text, connectivity=8)
        for ti in range(1, n_tc):
            ta = tc_stats[ti, cv2.CC_STAT_AREA]
            tw = tc_stats[ti, cv2.CC_STAT_WIDTH]
            th = tc_stats[ti, cv2.CC_STAT_HEIGHT]
            tmax = max(tw, th)
            tmin = max(min(tw, th), 1)
            aspect = tmax / tmin
            is_text = (ta <= text_char_max and aspect < 4.0) or \
                      (ta <= text_char_max * 2 and aspect < 2.0)
            if is_text:
                box_slice[tc_labels == ti] = 0

    # ── 4c. Global text cleanup — wall-connectivity approach ────────────
    # Find core walls (very long linear features), then mark every CC that
    # touches the core as wall-connected.  Erase small isolated CCs that
    # don't belong to the wall network (remaining text remnants).
    core_lk = max(25, int(15 * scale)) | 1
    ck_h = cv2.getStructuringElement(cv2.MORPH_RECT, (core_lk, 1))
    ck_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, core_lk))
    ck_d1 = _diag_kernel(max(25, int(15 * scale)), 1)
    ck_d2 = _diag_kernel(max(25, int(15 * scale)), -1)
    core_walls = cv2.bitwise_or(
        cv2.bitwise_or(
            cv2.morphologyEx(opened, cv2.MORPH_OPEN, ck_h),
            cv2.morphologyEx(opened, cv2.MORPH_OPEN, ck_v)),
        cv2.bitwise_or(
            cv2.morphologyEx(opened, cv2.MORPH_OPEN, ck_d1),
            cv2.morphologyEx(opened, cv2.MORPH_OPEN, ck_d2)))
    # Dilate core to bridge small gaps at junctions
    bridge_k = max(3, int(2 * scale))
    core_bridged = cv2.dilate(core_walls, np.ones((bridge_k, bridge_k), np.uint8))
    # Mark CCs in opened that overlap with the core wall network
    n_cc_all, cc_labels_all, cc_stats_all, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8)
    max_isolated_area = max(800, int(500 * scale * scale))
    for i in range(1, n_cc_all):
        cc_mask = (cc_labels_all == i)
        if (cc_mask & (core_bridged > 0)).any():
            continue  # connected to wall network — keep
        area = cc_stats_all[i, cv2.CC_STAT_AREA]
        if area <= max_isolated_area:
            opened[cc_mask] = 0

    # ── 5. Dot removal ───────────────────────────────────────────────────
    dot_max_area = max(8, int(5 * scale * scale))
    dot_max_dim = max(5, int(3 * scale))
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    for i in range(1, n_cc):
        area = cc_stats[i, cv2.CC_STAT_AREA]
        bw_cc = cc_stats[i, cv2.CC_STAT_WIDTH]
        bh_cc = cc_stats[i, cv2.CC_STAT_HEIGHT]
        max_dim = max(bw_cc, bh_cc)
        min_dim = max(min(bw_cc, bh_cc), 1)
        if area <= dot_max_area and max_dim / min_dim < 2.5 and max_dim <= dot_max_dim:
            opened[cc_labels == i] = 0

    filtered = opened

    # ── 7a. Uniform thin walls ───────────────────────────────────────────
    # Skip skeletonization (it destroys thin segments). Instead:
    # erode thick walls to thin them, then re-dilate to uniform width.
    wall_target = max(1, int(1.0 * scale))
    # Erode to remove thickness, then dilate back to uniform
    erode_k = max(1, int(1 * scale))
    erode_kernel = np.ones((erode_k, erode_k), np.uint8)
    thinned = cv2.erode(filtered, erode_kernel, iterations=1)
    # Re-dilate to get uniform wall width
    dilate_kernel = np.ones((wall_target, wall_target), np.uint8)
    thin_walls_raw = cv2.dilate(thinned, dilate_kernel, iterations=1)
    # Restore any thin lines that erode might have killed: OR with
    # a skeleton of the original (catches 1px lines lost by erode)
    skel = thin(filtered > 0).astype(np.uint8) * 255
    skel_dilated = cv2.dilate(skel, dilate_kernel, iterations=1)
    thin_walls_raw = cv2.bitwise_or(thin_walls_raw, skel_dilated)

    # 7a-ii. Bridge tiny breaks for line completion
    min_door_px = max(3, int(2 * scale))
    noise_k = max(2, min_door_px - 1)
    h_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (noise_k, 1))
    v_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (1, noise_k))
    thin_walls = cv2.bitwise_or(
        cv2.morphologyEx(thin_walls_raw, cv2.MORPH_CLOSE, h_noise),
        cv2.morphologyEx(thin_walls_raw, cv2.MORPH_CLOSE, v_noise),
    )

    # ── 7b. Door gap detection ───────────────────────────────────────────
    DOOR_PIXEL = 128
    door_gap_k = max(5, int(8 * scale)) | 1
    h_close = cv2.getStructuringElement(cv2.MORPH_RECT, (door_gap_k, 1))
    v_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, door_gap_k))
    closed_h = cv2.morphologyEx(thin_walls, cv2.MORPH_CLOSE, h_close)
    closed_v = cv2.morphologyEx(thin_walls, cv2.MORPH_CLOSE, v_close)
    gaps_h = cv2.subtract(closed_h, thin_walls)
    gaps_v = cv2.subtract(closed_v, thin_walls)
    all_gaps = cv2.bitwise_or(gaps_h, gaps_v)

    def inside_phrase(cx, cy):
        for (x1, y1, x2, y2) in phrase_boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    h_im, w_im = thin_walls.shape
    door_mask = np.zeros_like(thin_walls)
    max_door_px = max(20, int(15 * scale))

    num_gap_labels, gap_labels, gap_stats, gap_cents = cv2.connectedComponentsWithStats(all_gaps)
    for i in range(1, num_gap_labels):
        area = gap_stats[i, cv2.CC_STAT_AREA]
        gw = gap_stats[i, cv2.CC_STAT_WIDTH]
        gh = gap_stats[i, cv2.CC_STAT_HEIGHT]
        max_dim = max(gw, gh)
        if max_dim < min_door_px or max_dim > max_door_px:
            continue
        if area > max_door_px * 6:
            continue
        gcx = int(gap_cents[i][0])
        gcy = int(gap_cents[i][1])
        if inside_phrase(gcx, gcy):
            continue
        door_mask[gap_labels == i] = 255

    # ── 8. Assemble result ───────────────────────────────────────────────
    result = np.where(thin_walls > 0, 0, 255).astype(np.uint8)
    # Door markers only where adjacent to wall
    wall_adjacent = cv2.dilate((thin_walls > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    door_on_border = (door_mask > 0) & (wall_adjacent > 0)
    result[door_on_border] = DOOR_PIXEL

    # 9. Save as PGM and walls-only PNG
    out_dir = os.path.dirname(output_pgm_path)
    os.makedirs(out_dir or ".", exist_ok=True)
    pil_img = Image.fromarray(result)
    pil_img.save(output_pgm_path, format="PPM")

    # Walls-only view: map_walls.png (no room labels)
    walls_png = os.path.join(out_dir, "map_walls.png")
    bgr = np.zeros((*result.shape, 3), dtype=np.uint8)
    bgr[:] = (255, 255, 255)
    bgr[result <= 1] = (0, 0, 0)
    bgr[(result >= 126) & (result <= 130)] = (42, 42, 139)
    cv2.imwrite(walls_png, bgr)

    h_out, w_out = result.shape
    print(f"  Processed: {w_out}x{h_out} → {output_pgm_path}")
    return w_out, h_out


def generate_map_yaml(output_dir, pgm_filename="map.pgm", resolution=0.05, origin=None):
    """Write a ROS map_server compatible map.yaml file.

    Args:
        output_dir: Directory where map.yaml will be written.
        pgm_filename: Name of the PGM file (relative to YAML location).
        resolution: Meters per pixel.
        origin: [x, y, theta] origin in meters. Defaults to [0.0, 0.0, 0.0].
    """
    if origin is None:
        origin = [0.0, 0.0, 0.0]

    map_config = {
        "image": pgm_filename,
        "resolution": float(resolution),
        "origin": origin,
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }

    yaml_path = os.path.join(output_dir, "map.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(map_config, f, default_flow_style=False, sort_keys=False)

    print(f"  YAML written: {yaml_path}")
    return yaml_path


def _polygon_for_phrase(pgm, phrase_center_px, resolution, phrase_bbox_px=None):
    """Find the free-space region containing the phrase center and return its outline as a polygon in meters.

    Uses PGM as-is (no dilation): free = (pgm > 200). Smallest containing contour = region. Fallback only.
    """
    free = (pgm > 200).astype(np.uint8) * 255
    h, w = pgm.shape[:2]
    cx, cy = int(phrase_center_px[0]), int(phrase_center_px[1])
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    contours, _ = cv2.findContours(free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    containing = [c for c in contours if cv2.pointPolygonTest(c, (cx, cy), False) >= 0]

    # If point is on closed door/wall, search nearby for a free pixel and use its contour
    if not containing and 0 <= cy < h and 0 <= cx < w:
        for r in range(1, 50):
            found = False
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and free[ny, nx] > 0:
                        containing = [c for c in contours if cv2.pointPolygonTest(c, (nx, ny), False) >= 0]
                        if containing:
                            found = True
                            break
                if found:
                    break
            if found:
                break

    if not containing:
        return None

    cnt = min(containing, key=cv2.contourArea)
    pts = cnt.reshape(-1, 2)
    if len(pts) < 3:
        return None

    def dist(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    poly = [[float(round(px * resolution, 3)), float(round(py * resolution, 3))] for px, py in pts]
    if len(poly) >= 3 and dist(poly[0], poly[-1]) > 1e-6:
        poly.append(poly[0][:])

    return poly


def generate_waypoints_with_polygons(raw_image_path, floor_dir, resolution=0.05, min_conf=60):
    """Extract grouped phrases from OCR, define polygon outlines from PGM for each, write waypoints.yaml (readme format).

    Partitions free space by Voronoi (nearest phrase center); no dilation of PGM. Each label gets its internal region.
    """
    try:
        from ocr_visualize import extract_words, group_nearby_words
    except ImportError:
        print("  [WARN] ocr_visualize not available — skipping waypoints with polygons.")
        return None

    try:
        # Extract at low confidence so short words (e.g. "West") participate
        # in grouping; the phrase-level min_conf filter below handles quality.
        words = extract_words(raw_image_path, min_conf=30)
        phrases = group_nearby_words(words)
    except Exception as e:
        print(f"  [WARN] OCR or grouping failed: {e}")
        return None

    pgm_path = os.path.join(floor_dir, "map.pgm")
    if not os.path.isfile(pgm_path):
        print("  [WARN] map.pgm not found — skipping polygon waypoints.")
        return None

    pgm = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if pgm is None:
        print("  [WARN] Could not load map.pgm.")
        return None

    map_yaml = os.path.join(floor_dir, "map.yaml")
    if os.path.isfile(map_yaml):
        with open(map_yaml) as f:
            map_config = yaml.safe_load(f)
            resolution = float(map_config.get("resolution", resolution))

    # Dilate walls slightly before computing free space so BFS doesn't
    # leak through 1-2px skeleton gaps between rooms.
    wall_pixels = (pgm <= 200).astype(np.uint8)
    seal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_sealed = cv2.dilate(wall_pixels, seal_k, iterations=1)
    free = ((wall_sealed == 0)).astype(np.uint8) * 255
    h, w = pgm.shape[:2]

    # Filter phrases and build seed list (pixel coords: x, y)
    filtered = []
    for wd in phrases:
        if wd["conf"] < min_conf:
            continue
        text = wd["text"]
        if len(text) < 2 or not re.search(r"[A-Za-z]", text):
            continue
        label = re.sub(r"[^A-Za-z0-9_ -]", "", text).strip()
        if len(label) < 3:
            continue
        cx_px = wd["x"] + wd["w"] / 2
        cy_px = wd["y"] + wd["h"] / 2
        filtered.append({"wd": wd, "label": label, "cx_px": cx_px, "cy_px": cy_px})

    if not filtered:
        print("  No phrases passed filters.")
        return None

    # Wall-aware multi-seed BFS: all labels expand simultaneously through free
    # space.  Each pixel is claimed by whichever label reaches it first (shortest
    # wall-respecting path), producing a Voronoi-like partition that follows walls.
    from collections import deque
    label_im = np.full((h, w), -1, dtype=np.int32)
    q = deque()

    # Seed each label's center (snap to nearest free pixel if needed)
    for i, f in enumerate(filtered):
        sx, sy = int(f["cx_px"]), int(f["cy_px"])
        if not (0 <= sy < h and 0 <= sx < w and free[sy, sx] > 0):
            found = False
            for r in range(1, 30):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < w and 0 <= ny < h and free[ny, nx] > 0:
                            sx, sy = nx, ny
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                continue
        if label_im[sy, sx] == -1:
            label_im[sy, sx] = i
            q.append((sx, sy))

    # Simultaneous BFS expansion — walls block propagation
    while q:
        cx, cy = q.popleft()
        cur = label_im[cy, cx]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and free[ny, nx] > 0 and label_im[ny, nx] == -1:
                label_im[ny, nx] = cur
                q.append((nx, ny))

    def contour_to_poly_m(c):
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            return None
        poly = [[float(round(px * resolution, 3)), float(round(py * resolution, 3))] for px, py in pts]
        if (poly[0][0] - poly[-1][0]) ** 2 + (poly[0][1] - poly[-1][1]) ** 2 > 1e-12:
            poly.append(poly[0][:])
        return poly

    by_label = {}
    for i, f in enumerate(filtered):
        region = (label_im == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = int(f["cx_px"]), int(f["cy_px"])
        containing = [c for c in contours if cv2.pointPolygonTest(c, (sx, sy), False) >= 0]
        if not containing:
            containing = [c for c in contours if cv2.contourArea(c) >= 4]
        polygon = None
        if containing:
            cnt = max(containing, key=cv2.contourArea)
            polygon = contour_to_poly_m(cnt)
        if polygon is None:
            wd = f["wd"]
            x0 = float(round(wd["x"] * resolution, 3))
            y0 = float(round(wd["y"] * resolution, 3))
            x1 = float(round((wd["x"] + wd["w"]) * resolution, 3))
            y1 = float(round((wd["y"] + wd["h"]) * resolution, 3))
            polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        x_m = float(round(f["cx_px"] * resolution, 3))
        y_m = float(round(f["cy_px"] * resolution, 3))
        cand = {"wd": f["wd"], "x_m": x_m, "y_m": y_m, "polygon": polygon}
        if f["label"] not in by_label:
            by_label[f["label"]] = []
        by_label[f["label"]].append(cand)

    # Pick best room for each label: prefer higher conf, then real contour (not 5-pt fallback), then room-like area
    waypoints_list = []
    h, w = pgm.shape[:2]
    map_area_m2 = (w * resolution) * (h * resolution)

    def polygon_area_m2(poly):
        if len(poly) < 3:
            return 0.0
        area = 0.0
        for i in range(len(poly) - 1):
            area += poly[i][0] * poly[i + 1][1] - poly[i + 1][0] * poly[i][1]
        return abs(area) * 0.5

    def _wp_score(c):
        conf = c["wd"]["conf"]
        poly = c["polygon"]
        area = polygon_area_m2(poly)
        s = conf * 1e6
        s += len(poly) * 1e4
        if 0.001 * map_area_m2 <= area <= 0.5 * map_area_m2:
            s += 1e3
        elif area <= 0.001 * map_area_m2:
            s -= 1e3
        return s

    for label, candidates in by_label.items():
        # Keep ALL instances of duplicate labels (e.g. multiple "Misc OFFICES")
        # sorted best-first, numbering duplicates.
        ranked = sorted(candidates, key=_wp_score, reverse=True)
        for idx, cand in enumerate(ranked):
            name = label if idx == 0 else f"{label} {idx + 1}"
            waypoints_list.append({
                "name": name,
                "x": cand["x_m"],
                "y": cand["y_m"],
                "yaw": float(0.0),
                "polygon": cand["polygon"],
            })

    if not waypoints_list:
        print("  No waypoints with polygons generated.")
        return None

    wp_path = os.path.join(floor_dir, "waypoints.yaml")
    with open(wp_path, "w") as f:
        yaml.dump({"waypoints": waypoints_list}, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"  Waypoints (with polygon outlines) written: {wp_path} ({len(waypoints_list)} labels)")
    return wp_path


def generate_waypoints(original_image_path, output_dir, resolution=0.05):
    """Use Tesseract OCR to find room labels and estimate positions.

    Uses multi-pass OCR (preprocessing variants, rotations, multiple PSM modes)
    when ocr_visualize is available for robust extraction from floor plans.
    Falls back to single-pass OCR if not. Writes waypoints.yaml with
    label → (x, y) in meters.
    """
    try:
        import pytesseract
    except ImportError:
        print("  [WARN] pytesseract not installed — skipping waypoint extraction.")
        return None

    try:
        from ocr_visualize import extract_words
        use_robust_ocr = True
    except ImportError:
        use_robust_ocr = False

    try:
        img = Image.open(original_image_path)
    except Exception as e:
        print(f"  [WARN] Could not open image for OCR: {e}")
        return None

    words = []
    if use_robust_ocr:
        try:
            words = extract_words(original_image_path)
        except Exception as e:
            print(f"  [WARN] Robust OCR failed, falling back to simple pass: {e}")
            use_robust_ocr = False

    if not use_robust_ocr:
        try:
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as e:
            print(f"  [WARN] Tesseract OCR failed (is tesseract installed?): {e}")
            return None
        n = len(ocr_data["text"])
        for i in range(n):
            text = ocr_data["text"][i].strip()
            conf = int(ocr_data["conf"][i])
            if not text or conf < 40:
                continue
            if len(text) < 2 or not re.search(r"[A-Za-z]", text):
                continue
            words.append({
                "text": text, "conf": conf,
                "x": ocr_data["left"][i], "y": ocr_data["top"][i],
                "w": ocr_data["width"][i], "h": ocr_data["height"][i],
            })

    waypoints_raw = {}
    for wd in words:
        text = wd["text"]
        if len(text) < 2 or not re.search(r"[A-Za-z]", text):
            continue
        label = re.sub(r"[^A-Za-z0-9_ -]", "", text).strip()
        # Require 3+ chars to filter OCR garbage; keep numerals like "101"
        if len(label) < 3:
            continue
        cx = (wd["x"] + wd["w"] / 2) * resolution
        cy = (wd["y"] + wd["h"] / 2) * resolution
        if label:
            conf = wd.get("conf", 0)
            if label not in waypoints_raw or conf > waypoints_raw[label][2]:
                waypoints_raw[label] = (round(cx, 3), round(cy, 3), conf)

    waypoints = {k: {"x": v[0], "y": v[1]} for k, v in waypoints_raw.items()}

    if not waypoints:
        print("  No waypoints detected via OCR.")
        return None

    wp_path = os.path.join(output_dir, "waypoints.yaml")
    with open(wp_path, "w") as f:
        yaml.dump({"waypoints": waypoints}, f, default_flow_style=False, sort_keys=False)

    print(f"  Waypoints written: {wp_path} ({len(waypoints)} labels)")
    return wp_path


def process_entry(hotel_name, floor, source_url, license_str, resolution=0.05,
                  dataset_root="dataset", local_path=None, wall_sensitivity=3.0):
    """Orchestrate a single floor plan: download → process → YAML → waypoints.

    Returns a dict suitable for a row in dataset_index.csv.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", hotel_name)
    floor_dir = os.path.join(dataset_root, safe_name, f"floor_{floor}")
    os.makedirs(floor_dir, exist_ok=True)

    print(f"\n[{hotel_name} / Floor {floor}]")

    # Step 1: Get image
    if local_path and os.path.isfile(local_path):
        raw_path = local_path
        print(f"  Using local file: {raw_path}")
    elif source_url:
        ext = _guess_extension(source_url)
        raw_path = os.path.join(floor_dir, f"raw{ext}")
        try:
            download_image(source_url, raw_path)
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            return None
    else:
        print("  [ERROR] No source URL or local path provided.")
        return None

    # Step 2: Process into PGM
    pgm_path = os.path.join(floor_dir, "map.pgm")
    try:
        w, h = process_image(raw_path, pgm_path, wall_sensitivity=wall_sensitivity)
    except Exception as e:
        print(f"  [ERROR] Image processing failed: {e}")
        return None

    # Step 3: Generate map.yaml
    generate_map_yaml(floor_dir, resolution=resolution)

    # Step 4: Waypoints with polygon outlines (Phase 2) — readme format: list of name, x, y, yaw, polygon
    if not generate_waypoints_with_polygons(raw_path, floor_dir, resolution=resolution):
        generate_waypoints(raw_path, floor_dir, resolution=resolution)

    # Step 5: Generate rooms-labeled visualization
    try:
        from view_pgm_rooms import draw_rooms_view
        draw_rooms_view(floor_dir, output_path=os.path.join(floor_dir, "map_rooms_labeled.png"))
    except Exception as e:
        print(f"  [WARN] Room visualization failed: {e}")

    return {
        "hotel_name": hotel_name,
        "floor": floor,
        "source_url": source_url or "",
        "license": license_str or "",
        "pgm_path": pgm_path,
        "width_px": w,
        "height_px": h,
        "resolution": resolution,
    }


def batch_process(source, dataset_root="dataset", resolution=0.05, wall_sensitivity=3.0):
    """Process a batch of floor plans from a CSV file or a folder of images.

    CSV columns: hotel_name, floor, source_url, license

    For folder mode, each image becomes its own entry with the filename as hotel name.
    """
    results = []

    source_path = Path(source)

    if source_path.is_dir():
        # Folder mode: process each image file
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
        images = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in image_exts
        )
        if not images:
            print(f"No image files found in {source}")
            return results

        print(f"Found {len(images)} images in {source}")
        for img_path in images:
            name = img_path.stem
            row = process_entry(
                hotel_name=name,
                floor=1,
                source_url="",
                license_str="",
                resolution=resolution,
                dataset_root=dataset_root,
                local_path=str(img_path),
                wall_sensitivity=wall_sensitivity,
            )
            if row:
                results.append(row)

    elif source_path.is_file() and source_path.suffix.lower() == ".csv":
        # CSV mode
        with open(source, newline="") as f:
            reader = csv.DictReader(f)
            entries = list(reader)

        if not entries:
            print(f"No entries found in {source}")
            return results

        print(f"Processing {len(entries)} entries from {source}")
        for entry in entries:
            row = process_entry(
                hotel_name=entry.get("hotel_name", "Unknown"),
                floor=entry.get("floor", 1),
                source_url=entry.get("source_url", ""),
                license_str=entry.get("license", ""),
                resolution=resolution,
                dataset_root=dataset_root,
                wall_sensitivity=wall_sensitivity,
            )
            if row:
                results.append(row)
    else:
        print(f"Source must be a CSV file or directory of images: {source}")
        return results

    # Write dataset index
    if results:
        index_path = os.path.join(dataset_root, "dataset_index.csv")
        fieldnames = ["hotel_name", "floor", "source_url", "license",
                      "pgm_path", "width_px", "height_px", "resolution"]
        with open(index_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nDataset index written: {index_path} ({len(results)} entries)")

    return results


def _guess_extension(url):
    """Guess file extension from a URL."""
    lower = url.lower()
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".svg"]:
        if ext in lower:
            return ext
    return ".jpg"


def main():
    parser = argparse.ArgumentParser(
        description="Convert hotel floor plan images to ROS-compatible occupancy grid maps."
    )
    parser.add_argument(
        "--source",
        help="Path to sources.csv or a folder of floor plan images.",
    )
    parser.add_argument(
        "--image",
        help="Path to a single floor plan image to process.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (for --image mode). Default: dataset/<image_name>/floor_1",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="Map resolution in meters/pixel (default: 0.05).",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Root directory for dataset output (default: dataset/).",
    )
    parser.add_argument(
        "--wall-sensitivity",
        type=float,
        default=3.0,
        help="Erosion aggressiveness for wall detection (default: 3.0). "
             "Lower (e.g. 2.0) keeps thinner features; higher (e.g. 4.0) "
             "removes more text/dots.",
    )

    args = parser.parse_args()

    if not args.source and not args.image:
        parser.print_help()
        print("\nError: Provide either --source or --image.")
        sys.exit(1)

    if args.image:
        # Single image mode
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"Image not found: {args.image}")
            sys.exit(1)

        if args.output:
            out_dir = args.output
        else:
            out_dir = os.path.join(args.dataset_root, img_path.stem, "floor_1")

        os.makedirs(out_dir, exist_ok=True)
        pgm_path = os.path.join(out_dir, "map.pgm")

        print(f"Processing single image: {args.image}")
        process_image(args.image, pgm_path, wall_sensitivity=args.wall_sensitivity)
        generate_map_yaml(out_dir, resolution=args.resolution)
        if not generate_waypoints_with_polygons(args.image, out_dir, resolution=args.resolution):
            generate_waypoints(args.image, out_dir, resolution=args.resolution)
        try:
            from view_pgm_rooms import draw_rooms_view
            draw_rooms_view(out_dir, output_path=os.path.join(out_dir, "map_rooms_labeled.png"))
        except Exception as e:
            print(f"  [WARN] Room visualization failed: {e}")
        print("\nDone.")
    else:
        # Batch mode
        results = batch_process(
            args.source,
            dataset_root=args.dataset_root,
            resolution=args.resolution,
            wall_sensitivity=args.wall_sensitivity,
        )
        if results:
            print(f"\nDone. Processed {len(results)} floor plans.")
        else:
            print("\nNo floor plans were successfully processed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
