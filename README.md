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
  - name: "Elevator"
    x: 8.0
    y: 5.0
    yaw: 1.5708
```

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

## License

Derived maps inherit the source image license (commonly CC BY-SA). Attribution to the original Wikimedia source is required per entry.
