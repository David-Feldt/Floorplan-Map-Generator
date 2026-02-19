#!/usr/bin/env python3
"""Run OCR on floor plan images using the OCR library (Tesseract), then visualize."""

import csv
import math
import os

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFilter


def _run_ocr_on_image(pil_img, min_conf=30, psm=11):
    """Run Tesseract on a PIL image; return list of word dicts (x,y,w,h in that image's coords)."""
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=f"--psm {psm} --oem 3")
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        # Strip surrounding pipe/bracket artifacts from OCR
        text = text.strip("|[]{}()")
        conf = int(data["conf"][i])
        if not text or conf < min_conf:
            continue
        words.append({
            "text": text,
            "conf": conf,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        })
    return words


def _unrotate_boxes(words, rot_w, rot_h, orig_w, orig_h, angle_deg):
    """Map bounding boxes from rotated image back to original image coordinates."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    ocx, ocy = orig_w / 2, orig_h / 2
    rcx, rcy = rot_w / 2, rot_h / 2
    out = []
    for wd in words:
        rx = wd["x"] + wd["w"] / 2 - rcx
        ry = wd["y"] + wd["h"] / 2 - rcy
        ox = rx * cos_a + ry * sin_a + ocx
        oy = -rx * sin_a + ry * cos_a + ocy
        bw = max(wd["w"], wd["h"])
        bh = min(wd["w"], wd["h"])
        out.append({
            "text": wd["text"], "conf": wd["conf"],
            "x": int(ox - bw / 2), "y": int(oy - bh / 2),
            "w": int(bw), "h": int(bh),
        })
    return out


def _dedupe_by_location(words, dist=25):
    """Keep one word per location (prefer longer text / higher conf)."""
    out = []
    for wd in sorted(words, key=lambda d: (-len(d["text"]), -d["conf"])):
        cx, cy = wd["x"] + wd["w"] / 2, wd["y"] + wd["h"] / 2
        if any(abs(cx - (k["x"] + k["w"] / 2)) < dist and abs(cy - (k["y"] + k["h"] / 2)) < dist for k in out):
            continue
        out.append(wd)
    return out


def _clear_lines_for_ocr(img_bgr, width, height):
    """Use PGM-style logic to detect wall lines and whiten them so OCR sees text only.

    Same pipeline as convert_floorplan process_image: threshold, connected components,
    keep elongated/large structures (walls). Then whiten those pixels in the original
    so the image has lines cleared out and text is easier for OCR.
    """
    scale = np.clip(max(width, height) / 1000.0, 0.5, 5.0)
    img = img_bgr.copy()

    # Color pre-filter: high-saturation pixels → white (so colored fills don't become lines)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mask = hsv[:, :, 1] > 80
    img[sat_mask] = [255, 255, 255]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_k = int(3 * scale) | 1
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    # Otsu threshold: BINARY_INV so dark (walls + text) = 255
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    open_k = max(2, int(1.5 * scale))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Connected components: keep only wall-like (elongated or large)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    line_mask = np.zeros_like(gray, dtype=np.uint8)
    min_area = int(50 * scale * scale * 3.0)
    min_span = int(15 * scale * 3.0)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        max_dim = max(bw, bh)
        min_dim = max(min(bw, bh), 1)
        aspect = max_dim / min_dim
        if area >= min_area or (aspect >= 3.0 and max_dim >= min_span):
            line_mask[labels == i] = 255

    # Whiten line pixels in original grayscale so OCR sees clean background where lines were
    result = gray.copy()
    result[line_mask > 0] = 255
    return result


def extract_words(image_path, min_conf=30):
    """Run Tesseract OCR (horizontal + vertical/diagonal passes) and return word list.

    Uses PGM-style step to clear wall lines for easier OCR, plus scale+CLAHE for 'g's,
    and 0°/90°/270° passes for different direction text. Returns list of dicts: text, conf, x, y, w, h.
    """
    try:
        orig_cv = cv2.imread(image_path)
        pil = Image.open(image_path)
    except Exception:
        return []

    w, h = pil.size
    if orig_cv is None or orig_cv.size == 0:
        orig_cv = np.array(pil.convert("RGB"))[:, :, ::-1].copy()

    # PGM-style: clear wall lines so OCR sees text on clean background
    cleared_gray = _clear_lines_for_ocr(orig_cv, w, h)
    cleared_pil = Image.fromarray(cleared_gray)

    scale = max(1.0, 1800 / max(w, h))
    if scale > 1:
        pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        cleared_pil = cleared_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    gray = pil.convert("L")
    arr = np.array(gray)

    # Preprocessing that helps 'g' and low-contrast text: CLAHE + sharpen
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(arr)
    enhanced_pil = Image.fromarray(enhanced).filter(ImageFilter.SHARPEN)

    pre_w, pre_h = enhanced_pil.size
    all_words = []

    # Horizontal: gray, enhanced, and line-cleared (lines out of the way for OCR)
    for img in [gray, enhanced_pil, cleared_pil]:
        for wd in _run_ocr_on_image(img, min_conf=min_conf, psm=11):
            all_words.append(wd)

    # Rotated text: vertical (90/270) and diagonal (30/45/60/315/330) passes
    for angle in (90, 270, 30, 45, 60, 315, 330):
        rotated = enhanced_pil.rotate(-angle, expand=True, fillcolor=255)
        rw, rh = rotated.size
        for wd in _run_ocr_on_image(rotated, min_conf=min_conf, psm=11):
            all_words.extend(_unrotate_boxes([wd], rw, rh, pre_w, pre_h, angle))

    # Scale back to original image coordinates
    orig_w, orig_h = w, h
    if scale > 1:
        for wd in all_words:
            wd["x"] = int(wd["x"] / scale)
            wd["y"] = int(wd["y"] / scale)
            wd["w"] = int(wd["w"] / scale)
            wd["h"] = int(wd["h"] / scale)

    # Drop rotation artifacts (boxes mostly outside image)
    margin = 50
    all_words = [wd for wd in all_words
                 if -margin <= wd["x"] <= orig_w + margin and -margin <= wd["y"] <= orig_h + margin]

    # Noise filtering: reject detections that are clearly not real words
    import re as _re
    cleaned = []
    for wd in all_words:
        text = wd["text"].strip()
        # Strip leading/trailing punctuation/symbols to get core text
        core = _re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', text)
        alpha_chars = sum(1 for c in core if c.isalpha())
        # Reject if core has fewer than 2 alpha characters
        if alpha_chars < 2:
            continue
        # Low-confidence words need at least 3 alpha chars to survive
        if wd["conf"] < 70 and alpha_chars < 3:
            continue
        # Reject if mostly non-alphanumeric (symbols, punctuation, line artifacts)
        alnum_count = sum(1 for c in text if c.isalnum())
        if alnum_count < len(text) * 0.5:
            continue
        # Reject very small bounding boxes (likely line/dot artifacts)
        if wd["w"] < 12 or wd["h"] < 10:
            continue
        # Reject if area is tiny (noise from wall intersections)
        if wd["w"] * wd["h"] < 200:
            continue
        cleaned.append(wd)
    all_words = cleaned

    all_words = _dedupe_by_location(all_words)
    return all_words


def _merge_group(group):
    """Turn a list of word dicts into one phrase dict (combined bbox, text joined by space)."""
    if len(group) == 1:
        return group[0]
    x_min = min(w["x"] for w in group)
    y_min = min(w["y"] for w in group)
    x_max = max(w["x"] + w["w"] for w in group)
    y_max = max(w["y"] + w["h"] for w in group)
    return {
        "text": " ".join(w["text"] for w in group),
        "conf": max(w["conf"] for w in group),
        "x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min,
    }


def _merge_hyphenated(text_parts):
    """Join text parts, merging hyphenated line breaks (e.g. ['PRESI-', 'DENT'] → 'PRESIDENT')."""
    if not text_parts:
        return ""
    result = text_parts[0]
    for part in text_parts[1:]:
        if result.endswith("-"):
            # Hyphenated word break — join without space, remove hyphen
            result = result[:-1] + part
        else:
            result = result + " " + part
    return result


def group_nearby_words(words, max_gap_ratio=1.8, max_y_ratio=0.6, max_vertical_gap_ratio=2.0):
    """Group words close together in the same direction into single phrases.

    - Horizontal: same line (similar y), small horizontal gap → one phrase (e.g. Press + Briefing + Room).
    - Vertical: same column (similar x), small vertical gap → one phrase (e.g. Oval above Office).
    - Hyphenation: words ending in '-' merge with the word below (PRESI- + DENT → PRESIDENT).
    """
    if not words:
        return []

    avg_h = sum(w["h"] for w in words) / len(words)
    avg_char_w = sum(w["w"] / max(len(w["text"]), 1) for w in words) / len(words)
    max_gap = max(15, avg_char_w * max_gap_ratio)
    max_y_diff = max(10, avg_h * max_y_ratio)
    max_vert_gap = max(15, avg_h * max_vertical_gap_ratio)

    # Pass 1: group horizontally (same line, left-to-right)
    sorted_words = sorted(words, key=lambda d: (d["y"] + d["h"] / 2, d["x"]))
    phrases = []
    i = 0
    while i < len(sorted_words):
        wd = sorted_words[i]
        group = [wd]
        x_end = wd["x"] + wd["w"]
        y_center = wd["y"] + wd["h"] / 2
        j = i + 1
        while j < len(sorted_words):
            next_wd = sorted_words[j]
            next_y = next_wd["y"] + next_wd["h"] / 2
            gap = next_wd["x"] - x_end
            if abs(next_y - y_center) <= max_y_diff and 0 <= gap <= max_gap:
                group.append(next_wd)
                x_end = next_wd["x"] + next_wd["w"]
                y_center = sum(w["y"] + w["h"] / 2 for w in group) / len(group)
                j += 1
            else:
                break
        phrases.append(_merge_group(group))
        i = j

    # Pass 2: group vertically (stacked, same column — e.g. Oval above Office, PRESI- above DENT)
    # Use x-overlap check instead of just center distance for better matching
    sorted_phrases = sorted(phrases, key=lambda d: (d["x"] + d["w"] / 2, d["y"]))
    final = []
    used = set()
    # Sort by y first for vertical merging
    by_y = sorted(range(len(sorted_phrases)), key=lambda k: sorted_phrases[k]["y"])

    for idx in by_y:
        if idx in used:
            continue
        pd = sorted_phrases[idx]
        group = [pd]
        used.add(idx)

        # Look for phrases below that overlap horizontally
        changed = True
        while changed:
            changed = False
            last = group[-1]
            y_end = last["y"] + last["h"]
            x_lo = min(p["x"] for p in group)
            x_hi = max(p["x"] + p["w"] for p in group)
            x_center = (x_lo + x_hi) / 2
            group_w = x_hi - x_lo

            for jdx in by_y:
                if jdx in used:
                    continue
                cand = sorted_phrases[jdx]
                cand_x_center = cand["x"] + cand["w"] / 2
                vert_gap = cand["y"] - y_end

                # Check vertical proximity and horizontal overlap
                x_overlap = (min(x_hi, cand["x"] + cand["w"]) - max(x_lo, cand["x"]))
                min_w = min(group_w, cand["w"])
                overlap_ok = x_overlap > min_w * 0.3 if min_w > 0 else abs(cand_x_center - x_center) <= max_gap

                if 0 <= vert_gap <= max_vert_gap and overlap_ok:
                    group.append(cand)
                    used.add(jdx)
                    changed = True
                    break  # restart scan with updated group bounds

        # Sort group top-to-bottom and merge with hyphenation handling
        group.sort(key=lambda p: p["y"])
        if len(group) == 1:
            final.append(group[0])
        else:
            texts = [p["text"] for p in group]
            merged_text = _merge_hyphenated(texts)
            x_min = min(p["x"] for p in group)
            y_min = min(p["y"] for p in group)
            x_max = max(p["x"] + p["w"] for p in group)
            y_max = max(p["y"] + p["h"] for p in group)
            final.append({
                "text": merged_text,
                "conf": max(p["conf"] for p in group),
                "x": x_min, "y": y_min,
                "w": x_max - x_min, "h": y_max - y_min,
            })

    return final


def ocr_and_visualize(image_path, output_dir, min_conf=30, render_conf=83):
    """Run OCR, optionally group words, then save annotated image and CSV."""
    orig_cv = cv2.imread(image_path)
    if orig_cv is None:
        print(f"  Could not load: {image_path}")
        return None

    words = extract_words(image_path, min_conf=min_conf)
    words = group_nearby_words(words)

    annotated = orig_cv.copy()
    for wd in words:
        if wd["conf"] < render_conf:
            continue
        x, y, bw, bh = wd["x"], wd["y"], wd["w"], wd["h"]
        color = (0, 220, 0)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(annotated, f"{wd['text']} ({wd['conf']}%)",
                    (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    print(f"\n  Grouped phrases (words close together, same direction):")
    print(f"  {'Phrase':<30} {'Conf':>5}   {'X':>5} {'Y':>5} {'W':>5} {'H':>5}")
    print(f"  {'-'*30} {'-'*5}   {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for wd in sorted(words, key=lambda d: (d["y"], d["x"])):
        print(f"  {wd['text']:<30} {wd['conf']:>5}   {wd['x']:>5} {wd['y']:>5} {wd['w']:>5} {wd['h']:>5}")
    print(f"\n  Total grouped phrases: {len(words)}")

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base}_ocr_regions.png")
    cv2.imwrite(out_path, annotated)
    print(f"  Annotated image saved: {out_path}")

    csv_path = os.path.join(output_dir, f"{base}_ocr_words.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "conf", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(sorted(words, key=lambda d: (d["y"], d["x"])))
    print(f"  Word list saved: {csv_path}")

    return out_path


if __name__ == "__main__":
    plans = [
        ("White House West Wing",
         "dataset/White_House_West_Wing/floor_1/raw.png",
         "dataset/White_House_West_Wing/floor_1"),
        ("Cleveland Union Terminal",
         "dataset/Cleveland_Union_Terminal/floor_1/raw.jpg",
         "dataset/Cleveland_Union_Terminal/floor_1"),
    ]

    annotated_paths = []
    for name, raw, outdir in plans:
        print(f"\n{'='*60}")
        print(f"  OCR: {name}")
        print(f"{'='*60}")
        path = ocr_and_visualize(raw, outdir, min_conf=30, render_conf=83)
        if path:
            annotated_paths.append(path)

    for p in annotated_paths:
        os.system(f'open "{p}"')
