# Tennis Analytics - Output Format Specification

## 📊 JSON Output Format (v1.0)

Based on colleague's specification + architectural requirements.

---

## Complete Output Schema

```json
{
  "meta": {
    "matchId": "final_us_open_2024",
    "videoPath": "videos/final_us_open_2024.mp4",
    "processedAt": "2025-12-28T10:30:00Z",
    "processingTime": 125.5,
    "framerate": 30,
    "totalFrames": 27000,
    "duration": 900.0,
    "resolution": {"width": 1920, "height": 1080},

    "court": {
      "width": 10.97,
      "height": 23.77,
      "type": "singles",
      "surface": "hard"
    },

    "players": {
      "p1": {
        "id": "1",
        "name": "Alcaraz",
        "color": "#ff0000",
        "side": "near"
      },
      "p2": {
        "id": "2",
        "name": "Djokovic",
        "color": "#0000ff",
        "side": "far"
      }
    },

    "models": {
      "court": {
        "type": "tracknet",
        "version": "v3",
        "path": "models/court_model_best.pt"
      },
      "ball": {
        "type": "tracknet",
        "version": "v3",
        "path": "models/ball_model_best.pt"
      },
      "player": {
        "type": "yolo",
        "version": "v11n",
        "path": "models/yolov11n.pt"
      },
      "hit": {
        "type": "ball_hit_classifier",
        "version": "v1",
        "path": "models/ball_hit_model.pt"
      }
    },

    "config": {
      "profile": "default",
      "parallelMode": true,
      "chunkSize": 500
    }
  },

  "timeline": [
    {
      "t": 0.000,
      "fid": 0,

      "b": {
        "x": 0.501,
        "y": 0.100,
        "s": 120.5,
        "v": true,
        "interp": false,
        "evt": "hit",
        "conf": 0.95
      },

      "p1": {
        "x": 0.450,
        "y": 0.050,
        "v": true,
        "bbox": [0.42, 0.01, 0.48, 0.09],
        "conf": 0.88
      },

      "p2": {
        "x": 0.550,
        "y": 0.950,
        "v": true,
        "bbox": [0.52, 0.91, 0.58, 0.99],
        "conf": 0.91
      },

      "court": {
        "v": true,
        "conf": 0.98,
        "H": [
          [1.2, 0.0, -120],
          [0.0, 1.1, -50],
          [0.0, 0.0, 1.0]
        ]
      }
    },

    {
      "t": 0.033,
      "fid": 1,

      "b": {
        "x": 0.502,
        "y": 0.105,
        "s": 118.0,
        "v": true,
        "interp": false,
        "evt": null,
        "conf": 0.92
      },

      "p1": {
        "x": 0.455,
        "y": 0.052,
        "v": true,
        "bbox": [0.43, 0.02, 0.49, 0.10],
        "conf": 0.89
      },

      "p2": {
        "x": 0.548,
        "y": 0.948,
        "v": true,
        "bbox": [0.52, 0.90, 0.58, 0.98],
        "conf": 0.90
      }
    }
  ],

  "rallies": [
    {
      "id": 0,
      "startFrame": 0,
      "endFrame": 450,
      "startTime": 0.000,
      "endTime": 15.000,
      "duration": 15.000,
      "shotCount": 12,
      "winner": "p1",
      "reason": "ace",

      "events": [
        {
          "type": "hit",
          "frame": 0,
          "time": 0.000,
          "player": "p1",
          "shotType": "serve",
          "speed": 185.5,
          "position": {"x": 0.501, "y": 0.100}
        },
        {
          "type": "bounce",
          "frame": 22,
          "time": 0.733,
          "position": {"x": 0.520, "y": 0.850},
          "inOut": "in"
        },
        {
          "type": "hit",
          "frame": 45,
          "time": 1.500,
          "player": "p2",
          "shotType": "forehand",
          "speed": 98.5,
          "position": {"x": 0.540, "y": 0.920}
        }
      ],

      "statistics": {
        "p1": {
          "shots": 6,
          "avgSpeed": 115.5,
          "maxSpeed": 185.5,
          "winners": 1,
          "errors": 0
        },
        "p2": {
          "shots": 6,
          "avgSpeed": 102.3,
          "maxSpeed": 120.0,
          "winners": 0,
          "errors": 1
        }
      }
    }
  ],

  "statistics": {
    "overall": {
      "totalRallies": 45,
      "totalShots": 540,
      "avgRallyDuration": 8.5,
      "longestRally": 25.5,
      "avgBallSpeed": 105.5,
      "maxBallSpeed": 185.5
    },

    "p1": {
      "name": "Alcaraz",
      "ralliesWon": 24,
      "ralliesLost": 21,
      "totalShots": 275,
      "avgSpeed": 108.5,
      "maxSpeed": 185.5,
      "aces": 8,
      "doubleFaults": 3,
      "winners": 32,
      "unforcedErrors": 18,
      "shotTypes": {
        "serve": 45,
        "forehand": 120,
        "backhand": 85,
        "volley": 18,
        "smash": 7
      }
    },

    "p2": {
      "name": "Djokovic",
      "ralliesWon": 21,
      "ralliesLost": 24,
      "totalShots": 265,
      "avgSpeed": 102.3,
      "maxSpeed": 165.0,
      "aces": 5,
      "doubleFaults": 2,
      "winners": 28,
      "unforcedErrors": 22,
      "shotTypes": {
        "serve": 45,
        "forehand": 110,
        "backhand": 95,
        "volley": 12,
        "smash": 3
      }
    }
  },

  "processing": {
    "pipelineVersion": "2.0",
    "steps": [
      {
        "name": "PreprocessingPipeline",
        "duration": 5.2,
        "framesProcessed": 27000,
        "framesSkipped": 0
      },
      {
        "name": "DetectionPipeline",
        "duration": 85.5,
        "device": "cuda",
        "batchSize": 16
      },
      {
        "name": "TemporalPipeline",
        "duration": 12.3,
        "gapsFilled": 234,
        "smoothingApplied": true
      },
      {
        "name": "GeometryPipeline",
        "duration": 3.5,
        "homographyComputed": 900
      },
      {
        "name": "EventPipeline",
        "duration": 15.8,
        "hitsDetected": 540,
        "bouncesDetected": 520
      },
      {
        "name": "AnalyticsPipeline",
        "duration": 2.5
      },
      {
        "name": "RenderingPipeline",
        "duration": 0.7,
        "outputPath": "results/final_us_open_2024_analyzed.mp4"
      }
    ],
    "totalDuration": 125.5,
    "avgFps": 215.0,
    "errors": [],
    "warnings": [
      {
        "step": "TemporalPipeline",
        "message": "Large gap detected (30 frames) at t=45.2s, skipped interpolation"
      }
    ]
  }
}
```

---

## Field Definitions

### Meta Section

| Field | Type | Description |
|-------|------|-------------|
| `matchId` | string | Unique match identifier |
| `videoPath` | string | Original video file path |
| `processedAt` | ISO8601 | Processing timestamp |
| `processingTime` | float | Total processing time (seconds) |
| `framerate` | int | Video FPS |
| `totalFrames` | int | Total frame count |
| `duration` | float | Video duration (seconds) |
| `resolution` | object | Video resolution {width, height} |
| `court` | object | Court dimensions & type |
| `players` | object | Player metadata |
| `models` | object | Model versions used |
| `config` | object | Pipeline configuration used |

### Timeline Section (Frame-by-Frame)

**Optimized for 10,000+ frames:**

| Field | Type | Description |
|-------|------|-------------|
| `t` | float | Timestamp (seconds, 3 decimals) |
| `fid` | int | Frame ID (optional, can be derived from t * fps) |

#### Ball (`b`)

| Field | Type | Description |
|-------|------|-------------|
| `x` | float | Normalized X coordinate (0-1, court space) |
| `y` | float | Normalized Y coordinate (0-1, court space) |
| `s` | float | Speed (km/h) |
| `v` | bool | Visible (true/false) |
| `interp` | bool | Interpolated (gap-filled) |
| `evt` | string\|null | Event: "hit", "bounce", "in", "out", null |
| `conf` | float | Detection confidence (0-1) |

#### Player (`p1`, `p2`)

| Field | Type | Description |
|-------|------|-------------|
| `x` | float | Normalized center X (0-1, court space) |
| `y` | float | Normalized center Y (0-1, court space) |
| `v` | bool | Visible (detected) |
| `bbox` | array | Bounding box [x1, y1, x2, y2] (normalized) |
| `conf` | float | Detection confidence (0-1) |

#### Court (optional, only on frames with detection)

| Field | Type | Description |
|-------|------|-------------|
| `v` | bool | Court detected |
| `conf` | float | Detection confidence |
| `H` | array | 3x3 Homography matrix (only every 30 frames) |

---

## Coordinate System

**All coordinates in timeline are normalized (0-1) in court space:**

```
         0.0 (baseline, near)
          ┌───────────────┐
          │               │
          │   Singles     │  Court Width: 10.97m
  0.0 ────│   Court       │──── 1.0 (sideline)
          │               │
          │               │
          └───────────────┘
         1.0 (baseline, far)

Court Length: 23.77m
```

**To convert to meters:**
```python
x_meters = x_normalized * 10.97  # Width
y_meters = y_normalized * 23.77  # Length
```

**To convert to pixels:**
```python
x_pixels = x_normalized * video_width
y_pixels = y_normalized * video_height
```

---

## Size Optimization for Large Videos

For a 30-minute match (54,000 frames at 30fps), the JSON could be **~50MB+**.

### Option 1: Compression (Recommended)

```python
import gzip
import json

# Save compressed
with gzip.open("output.json.gz", "wt", encoding="utf-8") as f:
    json.dump(data, f)

# Load compressed
with gzip.open("output.json.gz", "rt", encoding="utf-8") as f:
    data = json.load(f)
```

**Size reduction: ~70-80% (50MB → 10-15MB)**

### Option 2: Split by Rallies

```json
// match_metadata.json (small, ~5KB)
{
  "meta": {...},
  "rallies": [
    {"id": 0, "file": "rally_0.json", "startTime": 0.0, "endTime": 15.0},
    {"id": 1, "file": "rally_1.json", "startTime": 18.5, "endTime": 32.0}
  ],
  "statistics": {...}
}

// rally_0.json (one rally only)
{
  "timeline": [...],  // Only frames for this rally
  "events": [...]
}
```

### Option 3: Binary Format (Advanced)

Use MessagePack or Protocol Buffers for 5-10x size reduction:

```python
import msgpack

# Save
with open("output.msgpack", "wb") as f:
    msgpack.pack(data, f)

# Load
with open("output.msgpack", "rb") as f:
    data = msgpack.unpack(f)
```

**Size: ~3-5MB for 30-min match**

---

## Database Schema Integration

**The JSON output can be imported to PostgreSQL:**

```sql
-- Matches table
INSERT INTO matches (id, video_path, framerate, duration)
VALUES ('final_us_open_2024', 'videos/...', 30, 900.0);

-- Timeline table (for fast queries)
CREATE TABLE timeline (
    match_id TEXT REFERENCES matches(id),
    frame_id INT,
    timestamp FLOAT,
    ball_x FLOAT,
    ball_y FLOAT,
    ball_speed FLOAT,
    ball_visible BOOL,
    ball_event TEXT,
    p1_x FLOAT,
    p1_y FLOAT,
    p1_visible BOOL,
    p2_x FLOAT,
    p2_y FLOAT,
    p2_visible BOOL,
    PRIMARY KEY (match_id, frame_id)
);

-- Import from JSON
INSERT INTO timeline (match_id, frame_id, timestamp, ball_x, ...)
SELECT
    'final_us_open_2024',
    jsonb_array_elements(timeline)->>'fid',
    jsonb_array_elements(timeline)->>'t',
    jsonb_array_elements(timeline)->'b'->>'x',
    ...
FROM matches_json;
```

---

## Output Generation in Pipeline

```python
# app/steps/analytics/json_export.py

class JSONExportStep(PipelineStep):
    """
    Export pipeline results to standardized JSON format
    """

    def process(self, context: ProcessingContext) -> ProcessingContext:
        output = {
            "meta": self._build_meta(context),
            "timeline": self._build_timeline(context),
            "rallies": self._build_rallies(context),
            "statistics": self._build_statistics(context),
            "processing": self._build_processing_info(context),
        }

        # Save JSON
        output_path = f"{context.video_path.stem}_analytics.json"

        if self.config.get('compress', True):
            with gzip.open(f"{output_path}.gz", "wt") as f:
                json.dump(output, f, indent=2)
        else:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)

        context.output_json_path = output_path
        return context

    def _build_timeline(self, context: ProcessingContext) -> list:
        """Build frame-by-frame timeline"""
        timeline = []

        for i, detection in enumerate(context.detections):
            frame_data = {
                "t": round(i / context.fps, 3),
                "fid": i,
            }

            # Ball data
            if detection.has_ball():
                frame_data["b"] = {
                    "x": round(detection.ball_position_court[0] / 10.97, 3),
                    "y": round(detection.ball_position_court[1] / 23.77, 3),
                    "s": round(detection.ball_speed, 1) if detection.ball_speed else None,
                    "v": True,
                    "interp": detection.is_interpolated,
                    "evt": self._get_ball_event(i, context),
                    "conf": round(detection.ball_confidence, 2),
                }
            else:
                frame_data["b"] = {"v": False}

            # Player data
            for p_idx, (box, conf) in enumerate(zip(detection.player_boxes, detection.player_confidences)):
                player_key = f"p{p_idx + 1}"
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2 / context.width
                center_y = (y1 + y2) / 2 / context.height

                frame_data[player_key] = {
                    "x": round(center_x, 3),
                    "y": round(center_y, 3),
                    "v": True,
                    "bbox": [
                        round(x1 / context.width, 3),
                        round(y1 / context.height, 3),
                        round(x2 / context.width, 3),
                        round(y2 / context.height, 3),
                    ],
                    "conf": round(conf, 2),
                }

            timeline.append(frame_data)

        return timeline
```

---

## Frontend Usage (JavaScript Example)

```javascript
// Load and visualize
async function loadMatch(matchId) {
    const response = await fetch(`/api/matches/${matchId}/analytics.json.gz`);
    const blob = await response.blob();

    // Decompress gzip
    const ds = new DecompressionStream('gzip');
    const decompressedStream = blob.stream().pipeThrough(ds);
    const decompressedBlob = await new Response(decompressedStream).blob();
    const text = await decompressedBlob.text();
    const data = JSON.parse(text);

    // Render timeline
    renderCourt(data.meta.court);

    data.timeline.forEach((frame, i) => {
        setTimeout(() => {
            if (frame.b.v) {
                drawBall(frame.b.x, frame.b.y, frame.b.evt);
            }

            drawPlayer(frame.p1, data.meta.players.p1.color);
            drawPlayer(frame.p2, data.meta.players.p2.color);
        }, i * 1000 / data.meta.framerate);
    });
}
```

---

## Recommendations

### ✅ Use This Format If:
- You need frame-by-frame precision
- Frontend visualization required
- API integration planned
- Database import needed

### ⚠️ Consider Alternatives If:
- Only statistics needed (use simplified JSON)
- Real-time streaming (use WebSocket with delta updates)
- Massive scale (use Parquet or database directly)

---

## Summary

This output format:
- ✅ **Compact**: Normalized coords (0-1), 3 decimal precision
- ✅ **Complete**: All detections, events, statistics
- ✅ **Optimized**: Compressible to ~10-15MB for 30-min video
- ✅ **Frontend-Ready**: Easy to parse and visualize
- ✅ **Database-Compatible**: Can import to PostgreSQL
- ✅ **Versioned**: Includes model versions and config

**Next Steps**: Implement `JSONExportStep` in Analytics Pipeline
