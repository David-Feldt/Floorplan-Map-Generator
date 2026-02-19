#!/usr/bin/env python3
"""View the PGM occupancy map with room polygons and names overlaid.

Usage:
    python view_pgm_rooms.py dataset/White_House_West_Wing/floor_1
    python view_pgm_rooms.py dataset/White_House_West_Wing/floor_1 --output rooms_view.png
"""

import argparse
import os

import cv2
import numpy as np
import yaml


def load_floor_data(floor_dir):
    """Load map.pgm, map.yaml, waypoints.yaml from floor_dir. Returns (pgm_bgr, resolution, waypoints) or (None, None, None)."""
    pgm_path = os.path.join(floor_dir, "map.pgm")
    yaml_path = os.path.join(floor_dir, "map.yaml")
    wp_path = os.path.join(floor_dir, "waypoints.yaml")

    if not os.path.isfile(pgm_path):
        print(f"  Not found: {pgm_path}")
        return None, None, None

    pgm = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if pgm is None:
        print(f"  Could not load: {pgm_path}")
        return None, None, None

    resolution = 0.05
    if os.path.isfile(yaml_path):
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
            resolution = float(cfg.get("resolution", resolution))

    waypoints = []
    if os.path.isfile(wp_path):
        with open(wp_path) as f:
            data = yaml.safe_load(f)
            waypoints = data.get("waypoints") or []
    print(f"  Loaded {len(waypoints)} rooms from {os.path.basename(floor_dir)}")

    # PGM to BGR: white=free, black=walls, 128=doors (brown), 64=stairs (gray-blue)
    bgr = np.zeros((*pgm.shape[:2], 3), dtype=np.uint8)
    bgr[:] = (255, 255, 255)  # free
    bgr[pgm <= 1] = (0, 0, 0)   # wall
    door_mask = (pgm >= 126) & (pgm <= 130)
    bgr[door_mask] = (42, 42, 139)   # brown in BGR
    stair_mask = (pgm >= 62) & (pgm <= 66)
    bgr[stair_mask] = (160, 120, 80)  # blue-gray for stairs
    return bgr, resolution, waypoints


def draw_rooms_view(floor_dir, output_path=None):
    """Draw PGM with room polygons and names; show and optionally save."""
    bgr, resolution, waypoints = load_floor_data(floor_dir)
    if bgr is None:
        return None

    h, w = bgr.shape[:2]

    def meter_to_pixel(x_m, y_m):
        """Convert map coordinates (meters) to image pixel (col, row)."""
        col = int(round(x_m / resolution))
        row = int(round(y_m / resolution))
        return (col, row)

    # Draw polygon outlines on a separate overlay at low opacity so PGM
    # walls remain clearly visible underneath.
    overlay = bgr.copy()
    alpha = 0.3  # polygon border opacity (0=invisible, 1=opaque)

    for i, wp in enumerate(waypoints):
        name = wp.get("name", "?")
        x_m = wp.get("x", 0)
        y_m = wp.get("y", 0)
        polygon = wp.get("polygon") or []

        # Color cycle for visibility (BGR)
        b, g, r = (i * 60) % 200 + 55, (i * 80 + 100) % 200 + 55, 255
        color_bgr = (int(b), int(g), int(r))

        if len(polygon) >= 2:
            pts = np.array([meter_to_pixel(p[0], p[1]) for p in polygon], dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

    # Blend: bgr shows through at (1-alpha), polygons at alpha
    cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0, bgr)

    # Draw labels on top (fully opaque so they stay readable)
    for i, wp in enumerate(waypoints):
        name = wp.get("name", "?")
        x_m = wp.get("x", 0)
        y_m = wp.get("y", 0)
        cx, cy = meter_to_pixel(x_m, y_m)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pad = 4
        cv2.rectangle(bgr, (cx - pad, cy - th - pad), (cx + tw + pad, cy + pad), (0, 0, 0), -1)
        cv2.putText(bgr, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, bgr)
        print(f"  Saved: {output_path}")

    show_window = not output_path
    if show_window:
        try:
            cv2.imshow("PGM with rooms", bgr)
            print("  Close the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            default_path = os.path.join(floor_dir, "pgm_rooms_view.png")
            cv2.imwrite(default_path, bgr)
            print(f"  No display; saved to {default_path}")
    return bgr


def main():
    parser = argparse.ArgumentParser(description="View PGM map with room polygons and names")
    parser.add_argument("floor_dir", help="Floor directory (e.g. dataset/Hotel/floor_1)")
    parser.add_argument("--output", "-o", help="Save view to this image path")
    args = parser.parse_args()

    if not os.path.isdir(args.floor_dir):
        print(f"Not a directory: {args.floor_dir}")
        return

    draw_rooms_view(args.floor_dir, output_path=args.output)


if __name__ == "__main__":
    main()
