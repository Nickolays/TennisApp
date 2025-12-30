# Tennis Analysis - Quick Commands

## Setup

```bash
# Activate virtual environment
cd ~/Projects/TennisAnalysis
source .venv/bin/activate
cd TennisApp
```

---

## Process Video (Complete Pipeline)

### Quick Test (100 frames)
```bash
python process_video.py tests/video3.mp4 --max-frames 100
```

### Full Video
```bash
python process_video.py tests/video3.mp4
```

### Custom Output Directory
```bash
python process_video.py tests/video3.mp4 --output results/my_analysis/
```

### Your Own Video
```bash
python process_video.py /path/to/your/video.mp4
```

---

## What It Does

**Detection (Phase 1)**:
- ✅ Court keypoints (14 points) - green dots
- ✅ Ball position - yellow/red/green circle with glow
- ✅ Players (improved) - blue bounding boxes, catches near + far players

**Temporal Processing (Phase 2)**:
- ✅ Gap filling (interpolation)
- ✅ Trajectory smoothing (Kalman filter)
- ✅ Temporal window extraction (±5 frames)

**Geometry Processing (Phase 3)**:
- ⏸️ Homography estimation (blocked - court model needs retraining)
- ⏸️ Coordinate transformation (pixel → court)
- ⏸️ Velocity estimation

**Event Detection (Phase 4)**:
- ⏸️ Ball bounce detection (requires homography)
- ⏸️ Ball hit detection (requires velocities)
- ✅ Event visualization ready (will work once court model improved)

---

## Output Files

After running, you'll get:

```
results/
├── video3_results.json      # JSON with all detections + events
└── video3_visualized.mp4    # Video with overlays + event colors
```

**Visualization Shows**:
- **Green dots**: Court keypoints (14 points)
- **Yellow circle**: Ball (normal tracking)
- **Red/Orange circle**: Ball bounce event - "BOUNCE!" label
- **Bright green circle**: Ball hit event - "HIT!" label
- **Blue boxes**: Players with labels (improved detection)
- **Info panel**: Frame #, time, detection status

---

## Test Scripts

All test scripts are in the `tests/` directory:

### Test Player Detection Only
```bash
python tests/test_player_detection.py
```

### Test All Phases (1-4)
```bash
python tests/test_phase4_events.py
```

### Test Individual Phases
```bash
python tests/test_phase1_live.py        # Detection only
python tests/test_phase2_temporal.py    # Temporal only
python tests/test_phase3_geometry.py    # Geometry (blocked by models)
```

---

## Example Commands

**Quick evaluation of models**:
```bash
# Process first 100 frames and check visualization
python process_video.py tests/video3.mp4 --max-frames 100

# Watch the output video
vlc results/video3_visualized.mp4
# or
mpv results/video3_visualized.mp4
```

**Full video processing**:
```bash
# Process entire video (493 frames for video3.mp4)
python process_video.py tests/video3.mp4

# Expected time: ~20-30 seconds
```

**Process your own video**:
```bash
# Replace with your video path
python process_video.py ~/Videos/tennis_match.mp4 --max-frames 500

# Check results
ls -lh results/
```

---

## Performance

**Expected Processing Speed** (RTX 3070):
- Court Detection: ~0.6s per 100 frames
- Ball Detection: ~3.0s per 100 frames
- Player Detection: ~1.9s per 100 frames
- Temporal Processing: ~0.01s per 100 frames
- **Total: ~5.5s per 100 frames (18 FPS)**

---

## Troubleshooting

**"Video not found"**:
```bash
# Check path
ls -la tests/video3.mp4

# Use absolute path
python process_video.py /full/path/to/video.mp4
```

**"Model not found"**:
```bash
# Check models exist
ls -la models/

# Should see:
# - court_model_best.pt
# - ball_model_best.pt
# - yolo11n.pt (downloads automatically on first use)
```

**"Out of memory"**:
```bash
# Process fewer frames
python process_video.py video.mp4 --max-frames 100
```

---

## JSON Output Format

```json
{
  "video": {
    "path": "tests/video3.mp4",
    "fps": 29.75,
    "frames": 100
  },
  "detections": {
    "court": 100,
    "ball": 100,
    "players": 100
  },
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "ball": {
        "x": 1326.8,
        "y": 400.4,
        "confidence": 3.72
      },
      "court": {
        "keypoints": [[x1, y1], [x2, y2], ...],
        "confidence": 11.74
      },
      "players": [
        {
          "box": [x1, y1, x2, y2],
          "confidence": 0.73
        }
      ]
    }
  ]
}
```

---

## Quick Start

```bash
# 1. Activate environment
cd ~/Projects/TennisAnalysis && source .venv/bin/activate && cd TennisApp

# 2. Process test video (100 frames)
python process_video.py tests/video3.mp4 --max-frames 100

# 3. Check results
ls -lh results/

# 4. Watch visualization
vlc results/video3_visualized.mp4
```

**That's it!** 🎾
