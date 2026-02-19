Readme · MD
Copy

# Hotel Floor Plan → ROS Map Dataset

Convert hotel floor plans from [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Floor_plans_of_hotels) into ROS-compatible 2D occupancy grid maps.

## Output Structure

```
dataset/
└── {Hotel_Name}/
    └── floor_{n}/
        ├── map.pgm        # Occupancy grid (white=free, black=wall)
        ├── map.yaml       # ROS map_server config
        └── waypoints.yaml # [Phase 2] Labeled room nav goals
```

## map.yaml
```yaml
image: map.pgm
resolution: 0.05        # m/px — estimated from scale bar or defaulted
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

## waypoints.yaml (Phase 2)
```yaml
waypoints:
  - name: "Room 101"
    x: 12.3
    y: 4.5
    yaw: 0.0    # door-facing direction
    polygon:    # outline of the labeled space (from PGM free-space), in meters; closed if needed
      - [x1, y1]
      - [x2, y2]
      - ...
  - name: "Elevator"
    x: 8.0
    y: 5.0
    yaw: 1.5708
    polygon: [[...], ...]
```
Polygon outlines are derived from the PGM occupancy grid: for each grouped phrase label, the free-space region containing the phrase is found; its contour is the polygon (closed or augmented if not fully closed).

## Pipeline

**Input:** SVG or PNG floor plan images from Wikimedia Commons
**Output:** A `.pgm` occupancy grid + `.yaml` config per floor

The core challenge is converting an architectural drawing into a clean binary map where walls are black and traversable space is white. Some ideas for approaches:
- Classical image processing (thresholding, morphological cleanup) via OpenCV
- ML-based segmentation to better distinguish walls from room labels and furniture
- Manual tracing in a tool like GIMP or Inkscape as a fallback for complex plans
- Resolution can be estimated from any visible scale bar, or defaulted and flagged

A script that batch-processes a folder of images and auto-generates the YAML files would be the ideal end state.

## Dataset Index (`dataset_index.csv`)

| hotel_name | floor | source_url | license | resolution_m_px | scale_estimated |
|---|---|---|---|---|---|
| Hilton_Vienna | 3 | https://commons.wikimedia.org/... | CC BY-SA 4.0 | 0.05 | true |

## Phase 2 — Waypoints

Room waypoints will be extracted via OCR (Tesseract) on the original floor plan image, projected into map frame, and manually corrected. Each point includes `(x, y, yaw)` where yaw faces the room door for delivery orientation.

## Progress

### Pipeline (`convert_floorplan.py`)
- **Image processing**: Otsu threshold with color pre-filter (high-saturation pixels whitened) to handle colored fills
- **OCR text removal**: Tesseract runs at 0/30/45/60/90/270/315/330 degree rotations to catch horizontal, vertical, and diagonal labels. Within each detected phrase box, directional morphological opening identifies wall lines; only small non-line CCs (text characters) are erased — walls passing through text regions are preserved
- **Dot/stipple removal**: Isolated tiny compact blobs removed; all elongated or large structures kept
- **Wall thinning**: Erode-then-dilate for uniform width, combined with skeleton to preserve thin lines
- **Door detection**: Directional morphological close finds wall gaps; gap CCs filtered by size and position; door pixels (value 128) only placed adjacent to walls
- **Stair detection**: Placeholder (value 64) for parallel-line regions
- **Output**: PGM with 0=wall, 128=door, 64=stair, 255=free

### Viewer (`view_pgm_rooms.py`)
- Renders PGM with colored room polygon outlines and name labels from `waypoints.yaml`
- Walls = black, doors = brown, stairs = blue-gray, free = white

### OCR (`ocr_visualize.py`)
- Multi-angle Tesseract OCR (7 rotation passes + 3 preprocessing variants)
- Word grouping into phrases (horizontal + vertical + hyphenated merging)
- Wall-line clearing before OCR so text is read on clean background

### Dataset
| Floor Plan | Rooms | Status |
|---|---|---|
| White House West Wing | 31 | Walls + doors + labels |
| Cleveland Union Terminal | 53 | Walls + doors + labels |

### Known Issues
- Some text remnants survive in dense areas where OCR misses diagonal/stylized labels
- Dot/stipple patterns in decorative areas (e.g. outdoor spaces) partially remain
- Door detection can produce false positives near text gaps

## License

Derived maps inherit the source image license (commonly CC BY-SA). Attribution to the original Wikimedia source is required per entry.
