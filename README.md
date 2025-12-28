# 🎾 Tennis Analytics Application

**Production-Ready Computer Vision Pipeline for Tennis Match Analysis**

## 🎯 Project Goal
Analyze tennis match videos to produce:
1. **Labeled Video** - Annotated with ball trajectory, court lines, player tracking, event markers
2. **Player Statistics** - Stored in database (shot types, speed, rally analytics, in/out decisions)

---

## 🏗️ Architecture Overview

### **Pipeline Philosophy: Nested Composable Steps**

The system uses a **hierarchical pipeline architecture**:
- **Main Pipeline**: High-level orchestration
- **Sub-Pipelines**: Logical grouping of related steps (Detection, Geometry, Analytics)
- **Parallel Execution**: GPU and CPU pipelines run concurrently

```
┌─────────────────────────────────────────────────────────────────┐
│                     MAIN TENNIS PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] FrameExtractionPipeline (CPU)                              │
│      ├─ VideoLoader                                             │
│      ├─ FrameFilter (motion detection)                          │
│      └─ FrameSampler (skip unused frames)                       │
│                                                                  │
│  [2] DetectionPipeline (GPU - Batched)                          │
│      ├─ CourtDetectionStep (TrackNet, every 30 frames)          │
│      ├─ BallDetectionStep (TrackNet, every frame)               │
│      └─ PlayerDetectionStep (YOLO, every frame)                 │
│                                                                  │
│  [3] TemporalProcessingPipeline (CPU - Parallel with Detection) │
│      ├─ GapFillingStep (interpolate ball NaNs)                  │
│      ├─ TrajectorySmoothing (Kalman filter)                     │
│      └─ TemporalWindowExtractor (±5 frame windows)              │
│                                                                  │
│  [4] GeometryPipeline (CPU)                                     │
│      ├─ HomographyStep (compute & cache every 30 frames)        │
│      └─ CoordinateTransformStep (pixel → court meters)          │
│                                                                  │
│  [5] EventDetectionPipeline (GPU + CPU)                         │
│      ├─ BallHitDetectionStep (11-frame ML model - GPU)          │
│      ├─ BounceDetectionStep (physics-based - CPU)               │
│      ├─ InOutDetectionStep (court boundary - CPU)               │
│      └─ SpeedCalculationStep (velocity analysis - CPU)          │
│                                                                  │
│  [6] AnalyticsPipeline (CPU)                                    │
│      ├─ GameSegmentationStep (rally/idle/prep)                  │
│      ├─ StatisticsAggregation (per-player stats)                │
│      └─ DatabaseExportStep (save to DB)                         │
│                                                                  │
│  [7] RenderingPipeline (CPU)                                    │
│      ├─ OverlayRenderer (draw court, ball, players)             │
│      ├─ MiniMapRenderer (bird's eye view)                       │
│      ├─ EventMarkersRenderer (hits, bounces, in/out)            │
│      └─ VideoWriter (save output video)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### **Parallel Execution Strategy**

```
Timeline (frames 0-500):
─────────────────────────────────────────────────────────────────

Thread 1 (GPU):  [Detection Pipeline]──────────▶
                       ↓ (queue)
Thread 2 (CPU):              [Temporal Pipeline]─────▶
                                    ↓ (queue)
Thread 3 (CPU):                          [Geometry Pipeline]──▶
                                                ↓ (queue)
Thread 4 (GPU):                                      [Event Detection]─▶
                                                           ↓ (queue)
Thread 5 (CPU):                                                 [Analytics]──▶
                                                                      ↓
Thread 6 (CPU):                                                       [Rendering]─▶

Key: Each pipeline processes chunks asynchronously, passing results via queues
```

---

## 📁 Project Structure

```
TennisApp/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── data_models.py           # Pydantic models for all data structures
│   │   ├── pipeline.py              # Base Pipeline, PipelineStep, AsyncPipeline
│   │   ├── context.py               # ProcessingContext (shared state)
│   │   └── executor.py              # ParallelExecutor (async coordination)
│   │
│   ├── steps/                        # Individual pipeline steps
│   │   ├── __init__.py
│   │   ├── base.py                  # PipelineStep base class
│   │   │
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── video_loader.py      # Load video metadata & frames
│   │   │   ├── frame_filter.py      # Motion-based filtering
│   │   │   └── frame_sampler.py     # Skip every N frames
│   │   │
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   ├── court_detection.py   # TrackNet court keypoints
│   │   │   ├── ball_detection.py    # TrackNet ball position
│   │   │   └── player_detection.py  # YOLO player bounding boxes
│   │   │
│   │   ├── temporal/
│   │   │   ├── __init__.py
│   │   │   ├── gap_filling.py       # Interpolate missing ball positions
│   │   │   ├── trajectory_smoothing.py  # Kalman filter
│   │   │   └── window_extractor.py  # Create temporal windows (±N frames)
│   │   │
│   │   ├── geometry/
│   │   │   ├── __init__.py
│   │   │   ├── homography.py        # Compute pixel→court transformation
│   │   │   └── coordinate_transform.py  # Apply homography to all positions
│   │   │
│   │   ├── events/
│   │   │   ├── __init__.py
│   │   │   ├── ball_hit_detection.py    # 11-frame ML model
│   │   │   ├── bounce_detection.py      # Physics-based bounce detection
│   │   │   ├── in_out_detection.py      # Court boundary validation
│   │   │   └── speed_calculation.py     # Ball speed & velocity
│   │   │
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   ├── game_segmentation.py     # Split into rallies/idle
│   │   │   ├── statistics.py            # Aggregate player stats
│   │   │   └── database_export.py       # Save to PostgreSQL/MongoDB
│   │   │
│   │   └── rendering/
│   │       ├── __init__.py
│   │       ├── overlay_renderer.py      # Draw court, ball, players
│   │       ├── minimap_renderer.py      # Bird's eye view
│   │       ├── event_markers.py         # Draw hit/bounce markers
│   │       └── video_writer.py          # Save annotated video
│   │
│   ├── pipelines/                    # Pre-configured sub-pipelines
│   │   ├── __init__.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── detection_pipeline.py
│   │   ├── temporal_pipeline.py
│   │   ├── geometry_pipeline.py
│   │   ├── event_pipeline.py
│   │   ├── analytics_pipeline.py
│   │   └── rendering_pipeline.py
│   │
│   ├── models/                       # Neural network wrappers
│   │   ├── __init__.py
│   │   ├── base_model.py            # BaseDetectionModel interface
│   │   ├── tracknet.py              # TrackNet (court + ball)
│   │   ├── yolo_detector.py         # YOLO (players)
│   │   ├── ball_hit_model.py        # Ball hit classifier (11 frames)
│   │   ├── shot_type_model.py       # FUTURE: shot type classifier
│   │   └── model_registry.py        # Singleton model loader
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video_processor.py       # Main service (simplified orchestrator)
│   │   └── pipeline_builder.py      # Build pipelines from config
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   └── schemas.py               # Pydantic request/response models
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py                # SQLAlchemy/MongoDB models
│   │   └── repository.py            # Database operations
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py           # Video I/O utilities
│       ├── geometry_utils.py        # Coordinate transformations
│       └── validation.py            # Physics validation functions
│
├── configs/                          # YAML configuration files
│   ├── default.yaml                 # Default settings
│   ├── fast.yaml                    # Speed-optimized (lower quality)
│   ├── accurate.yaml                # Accuracy-optimized (slower)
│   └── production.yaml              # Production settings
│
├── models/                           # Model checkpoints
│   ├── court_model_best.pt
│   ├── ball_model_best.pt
│   ├── yolov11n.pt
│   └── ball_hit_model.pt
│
├── tests/
│   ├── unit/                        # Test individual steps
│   ├── integration/                 # Test sub-pipelines
│   └── e2e/                         # Full pipeline tests
│
├── notebooks/                        # Jupyter notebooks for experiments
├── data/                             # Training datasets
├── results/                          # Output videos
├── logs/                             # Application logs
│
├── requirements.txt
├── pyproject.toml                    # Poetry/pip configuration
├── docker-compose.yml                # Docker setup (API + DB + Redis)
├── Dockerfile
└── README.md                         # This file
```

---

## 🔧 Core Design Patterns

### 1. **PipelineStep Interface**

Every step follows the same interface:

```python
from abc import ABC, abstractmethod
from app.core.context import ProcessingContext

class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Design principles:
    - Single responsibility
    - Independently testable
    - Chainable (input/output via context)
    - Stateless (all state in context)
    """

    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process data from context and return updated context.

        Args:
            context: Shared processing state

        Returns:
            Updated context (can be same object or new)
        """
        pass

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """Callable interface for pipeline chaining"""
        return self.process(context)
```

### 2. **ProcessingContext (Shared State)**

A "blackboard" that flows through the pipeline:

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class ProcessingContext:
    """
    Shared state for entire pipeline.
    Steps read from and write to this context.
    """

    # Video metadata
    video_path: str
    fps: float
    total_frames: int
    width: int
    height: int

    # Current processing chunk
    chunk_start: int = 0
    chunk_end: int = 0

    # Frame data (populated by FrameExtractionPipeline)
    frames: List[np.ndarray] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    active_frame_mask: List[bool] = field(default_factory=list)  # Motion filter

    # Detection results (populated by DetectionPipeline)
    detections: List[FrameDetection] = field(default_factory=list)

    # Homography matrices (cached by GeometryPipeline)
    # Key: frame_id, Value: 3x3 homography matrix
    homography_cache: Dict[int, np.ndarray] = field(default_factory=dict)

    # Temporal data (populated by TemporalPipeline)
    ball_trajectories: List[BallTrajectory] = field(default_factory=list)
    temporal_windows: Dict[int, List[FrameDetection]] = field(default_factory=dict)

    # Events (populated by EventPipeline)
    hit_events: List[int] = field(default_factory=list)
    bounce_events: List[int] = field(default_factory=list)
    in_out_events: List[Tuple[int, str]] = field(default_factory=list)  # (frame, 'in'/'out')

    # Analytics (populated by AnalyticsPipeline)
    game_segments: List[GameSegment] = field(default_factory=list)
    player_statistics: Dict[str, Any] = field(default_factory=dict)

    # Rendering (populated by RenderingPipeline)
    annotated_frames: List[np.ndarray] = field(default_factory=list)

    # Performance tracking
    step_timings: Dict[str, float] = field(default_factory=dict)

    def get_homography_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get homography matrix for a frame.
        Uses cached value from nearest computed frame.
        """
        if frame_id in self.homography_cache:
            return self.homography_cache[frame_id]

        # Find closest cached frame
        cached_frames = sorted(self.homography_cache.keys())
        if not cached_frames:
            return None

        closest = min(cached_frames, key=lambda x: abs(x - frame_id))
        return self.homography_cache[closest]

    def clear_frames(self):
        """Clear frame data to free memory after processing chunk"""
        self.frames = []
        self.annotated_frames = []
```

### 3. **Nested Pipeline Pattern**

```python
from typing import List
from app.core.context import ProcessingContext
from app.steps.base import PipelineStep

class Pipeline:
    """
    Composable pipeline of steps.
    Can contain individual steps OR other pipelines (nested).
    """

    def __init__(self, name: str, steps: List[PipelineStep]):
        self.name = name
        self.steps = steps

    def run(self, context: ProcessingContext) -> ProcessingContext:
        """Execute all steps sequentially"""
        print(f"\n{'='*60}")
        print(f"Starting Pipeline: {self.name}")
        print(f"{'='*60}")

        for step in self.steps:
            context = step(context)

        print(f"[{self.name}] ✓ Complete\n")
        return context

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """Make pipeline callable like a step"""
        return self.run(context)


class AsyncPipeline(Pipeline):
    """
    Asynchronous pipeline for parallel execution.
    Processes chunks in background and yields results.
    """

    async def run_async(self, context: ProcessingContext):
        """Execute steps asynchronously with chunk-based processing"""
        # Implementation in app/core/pipeline.py
        pass
```

### 4. **Parallel Execution with Executor**

```python
import asyncio
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ParallelExecutor:
    """
    Coordinates parallel execution of GPU and CPU pipelines.

    Strategy:
    1. GPU pipelines run in dedicated thread (CUDA limitation)
    2. CPU pipelines run in process pool (bypass GIL)
    3. Queues coordinate data flow between pipelines
    """

    def __init__(self, config: dict):
        self.gpu_executor = ThreadPoolExecutor(max_workers=1)  # GPU (single thread)
        self.cpu_executor = ProcessPoolExecutor(max_workers=config['cpu_workers'])

        self.detection_queue = Queue(maxsize=10)  # GPU → CPU
        self.geometry_queue = Queue(maxsize=10)   # CPU → CPU
        self.event_queue = Queue(maxsize=10)      # Mixed → CPU

    def execute_parallel(self, context: ProcessingContext):
        """
        Execute pipeline stages in parallel:

        GPU Thread:  Detection Pipeline
                           ↓
        CPU Process: Temporal + Geometry Pipelines (parallel)
                           ↓
        GPU Thread:  Event Detection (ball hit model)
                           ↓
        CPU Process: Analytics + Rendering
        """

        # Stage 1: Detection (GPU)
        detection_future = self.gpu_executor.submit(
            self.run_detection_pipeline, context
        )

        # Stage 2: Temporal + Geometry (CPU, parallel)
        # Wait for detection results
        context = detection_future.result()

        temporal_future = self.cpu_executor.submit(
            self.run_temporal_pipeline, context
        )
        geometry_future = self.cpu_executor.submit(
            self.run_geometry_pipeline, context
        )

        # Merge results
        context = temporal_future.result()
        context = geometry_future.result()

        # Stage 3: Event Detection (GPU)
        event_future = self.gpu_executor.submit(
            self.run_event_pipeline, context
        )
        context = event_future.result()

        # Stage 4: Analytics + Rendering (CPU)
        analytics_future = self.cpu_executor.submit(
            self.run_analytics_pipeline, context
        )
        rendering_future = self.cpu_executor.submit(
            self.run_rendering_pipeline, context
        )

        context = analytics_future.result()
        context = rendering_future.result()

        return context
```

---

## ⚙️ Configuration System

### **configs/default.yaml**

```yaml
# Default Tennis Analytics Pipeline Configuration

# Video Processing
video:
  chunk_size: 500              # Frames per processing chunk
  output_dir: "results/"
  save_annotated_video: true
  save_analytics_json: true

# Parallel Execution
execution:
  mode: "parallel"             # "sequential" or "parallel"
  cpu_workers: 4               # Number of CPU processes
  gpu_batch_size: 16           # Batch size for GPU inference
  prefetch_chunks: 2           # Number of chunks to prefetch

# [1] Frame Extraction Pipeline
preprocessing:
  skip_frames: false           # Skip every N frames (false = process all)
  frame_skip_interval: 1       # Only used if skip_frames=true

  motion_filter:
    enabled: true
    threshold: 5.0             # Pixel difference threshold
    smoothing_window: 5        # Temporal smoothing (frames)

# [2] Detection Pipeline (GPU)
detection:
  # Court Detection (TrackNet)
  court:
    enabled: true
    model_path: "models/court_model_best.pt"
    model_type: "tracknet"     # Easily swap: "tracknet" | "custom_cnn"
    interval: 30               # Detect every N frames (court doesn't move)
    confidence_threshold: 0.5
    input_size: [640, 360]     # Model input resolution

  # Ball Detection (TrackNet)
  ball:
    enabled: true
    model_path: "models/ball_model_best.pt"
    model_type: "tracknet"     # Easily swap models
    interval: 1                # Every frame
    confidence_threshold: 0.3
    input_size: [640, 360]

  # Player Detection (YOLO)
  player:
    enabled: true
    model_path: "models/yolov11n.pt"
    model_type: "yolo"         # Easily swap: "yolo" | "faster_rcnn" | "detr"
    interval: 1                # Every frame
    confidence_threshold: 0.5
    nms_threshold: 0.4

# [3] Temporal Processing Pipeline (CPU)
temporal:
  # Gap Filling
  gap_filling:
    enabled: true
    max_gap_linear: 5          # Use linear interpolation for gaps < 5 frames
    max_gap_polynomial: 15     # Use polynomial for gaps 5-15 frames
    max_gap_total: 30          # Discard gaps > 30 frames (likely occlusion)

  # Trajectory Smoothing
  smoothing:
    enabled: true
    method: "kalman"           # "kalman" | "savgol" | "gaussian"
    kalman_process_noise: 0.1
    kalman_measurement_noise: 1.0

  # Temporal Windows
  windows:
    ball_hit_window: 5         # ±5 frames = 11 total for hit detection

# [4] Geometry Pipeline (CPU)
geometry:
  homography:
    interval: 30               # Compute every N frames (cache for others)
    court_template: "singles"  # "singles" | "doubles"
    ransac_threshold: 5.0
    min_determinant: 0.01      # Reject ill-conditioned matrices

  coordinate_transform:
    apply_to_ball: true
    apply_to_players: true

# [5] Event Detection Pipeline
events:
  # Ball Hit Detection (ML model)
  ball_hit:
    enabled: true
    model_path: "models/ball_hit_model.pt"
    model_type: "ball_hit_classifier"  # 11-frame input
    confidence_threshold: 0.7
    device: "cuda"             # "cuda" | "cpu"

  # Bounce Detection (physics-based)
  bounce:
    enabled: true
    vertical_acceleration_threshold: 15.0  # m/s²
    direction_change_threshold: 0.3

  # In/Out Detection
  in_out:
    enabled: true
    boundary_margin: 0.05      # 5cm margin (meters)

  # Speed Calculation
  speed:
    enabled: true
    max_ball_speed: 70.0       # m/s (physics validation)
    smoothing_window: 3        # Frames

# [6] Analytics Pipeline (CPU)
analytics:
  game_segmentation:
    enabled: true
    min_rally_frames: 30       # Minimum frames for valid rally
    idle_threshold: 60         # Frames of inactivity = idle segment

  statistics:
    enabled: true
    metrics:
      - "avg_ball_speed"
      - "max_ball_speed"
      - "rally_count"
      - "rally_duration"
      - "shot_count_per_player"
      - "in_out_ratio"

  database_export:
    enabled: true
    database_type: "postgresql"  # "postgresql" | "mongodb" | "sqlite"
    connection_string: "postgresql://user:pass@localhost:5432/tennis_db"
    table_prefix: "match_"

# [7] Rendering Pipeline (CPU)
rendering:
  overlay:
    draw_court_lines: true
    draw_ball_trajectory: true
    draw_player_boxes: true
    draw_event_markers: true   # Hits, bounces, in/out

    colors:
      court_lines: [0, 255, 0]      # Green
      ball_trajectory: [255, 0, 0]  # Red
      player_boxes: [0, 0, 255]     # Blue
      hit_marker: [255, 255, 0]     # Yellow
      bounce_marker: [255, 0, 255]  # Magenta

    line_thickness: 2
    trajectory_length: 30      # Show last N positions

  minimap:
    enabled: true
    position: "bottom_right"   # "top_left" | "top_right" | "bottom_left" | "bottom_right"
    size: [300, 200]           # Width x Height (pixels)
    background_color: [34, 139, 34]  # Court green

  video_output:
    codec: "mp4v"              # "mp4v" | "h264" | "hevc"
    fps: null                  # null = use input video fps
    quality: 95                # 0-100

# Model Registry (for easy model swapping)
model_registry:
  tracknet:
    class: "app.models.tracknet.TrackNetModel"
  yolo:
    class: "app.models.yolo_detector.YOLODetector"
  ball_hit_classifier:
    class: "app.models.ball_hit_model.BallHitClassifier"
  # FUTURE: Add more models here
  # faster_rcnn:
  #   class: "app.models.faster_rcnn.FasterRCNNDetector"

# Logging
logging:
  level: "INFO"              # DEBUG | INFO | WARNING | ERROR
  file: "logs/tennis_pipeline.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### **configs/fast.yaml** (Speed-Optimized)

```yaml
# Inherits from default.yaml, only override differences

video:
  chunk_size: 1000             # Larger chunks

preprocessing:
  skip_frames: true
  frame_skip_interval: 2       # Process every 2nd frame

detection:
  court:
    interval: 60               # Less frequent court detection
    input_size: [320, 180]     # Lower resolution
  ball:
    input_size: [320, 180]
  player:
    model_path: "models/yolov11n.pt"  # Nano model (fastest)

execution:
  gpu_batch_size: 32           # Larger batches

temporal:
  smoothing:
    enabled: false             # Skip smoothing for speed

rendering:
  overlay:
    draw_ball_trajectory: false
  minimap:
    enabled: false
```

### **configs/production.yaml**

```yaml
# Production settings with database, API, monitoring

video:
  chunk_size: 500
  output_dir: "/mnt/storage/tennis_results/"

execution:
  mode: "parallel"
  cpu_workers: 8               # More workers for production server

analytics:
  database_export:
    enabled: true
    database_type: "postgresql"
    connection_string: "${DATABASE_URL}"  # From environment variable
    pool_size: 10
    pool_timeout: 30

api:
  enabled: true
  host: "0.0.0.0"
  port: 8000
  workers: 4
  redis_url: "redis://localhost:6379/0"
  job_timeout: 3600            # 1 hour max per video

monitoring:
  prometheus:
    enabled: true
    port: 9090
  sentry_dsn: "${SENTRY_DSN}"

logging:
  level: "INFO"
  file: "/var/log/tennis_pipeline/app.log"
  rotate: true
  max_bytes: 10485760          # 10MB
  backup_count: 5
```

---

## 🔄 Pipeline Flow Diagrams

### **Sequential Mode** (Simple, easier to debug)

```
Frame 0-499 → [Preprocessing] → [Detection] → [Temporal] → [Geometry] → [Events] → [Analytics] → [Rendering] → Output
Frame 500-999 → [Preprocessing] → [Detection] → ...
```

### **Parallel Mode** (Production, faster)

```
Chunk 1 (frames 0-499):
Thread 1 (GPU):  [Detection]════════════════▶
                       ↓ (queue)
Thread 2 (CPU):              [Temporal]═══════▶
                                    ↓
Thread 3 (CPU):                          [Geometry]═══▶
                                                ↓
Thread 4 (GPU):                                      [Event Det.]══▶
                                                           ↓
Thread 5 (CPU):                                                 [Analytics]══▶
                                                                      ↓
Thread 6 (CPU):                                                            [Render]══▶ Output

Meanwhile, Chunk 2 (frames 500-999) starts on Thread 1...
```

---

## 🚀 Usage Examples

### **1. Command Line (Simple)**

```bash
# Using default config
python -m app.main process video.mp4

# Using custom config
python -m app.main process video.mp4 --config configs/fast.yaml

# Parallel mode
python -m app.main process video.mp4 --config configs/production.yaml --parallel
```

### **2. Python API**

```python
from app.services.pipeline_builder import PipelineBuilder
from app.services.video_processor import VideoProcessor
from app.core.context import ProcessingContext

# Load configuration
config = PipelineBuilder.load_config("configs/default.yaml")

# Build pipeline
pipeline = PipelineBuilder.build_from_config(config)

# Create video processor
processor = VideoProcessor(pipeline, config)

# Process video
result = processor.process_video("match.mp4", output_name="match_analyzed.mp4")

# Access results
print(f"Total rallies: {len(result.game_segments)}")
print(f"Average ball speed: {result.player_statistics['avg_ball_speed']} m/s")
```

### **3. REST API**

```bash
# Start API server
python -m app.api.main

# Submit video for processing
curl -X POST http://localhost:8000/api/v1/upload \
  -F "video=@match.mp4" \
  -F "config=fast"

# Response: {"job_id": "abc123", "status": "queued"}

# Check status
curl http://localhost:8000/api/v1/jobs/abc123
# Response: {"job_id": "abc123", "status": "processing", "progress": 0.45}

# Get results
curl http://localhost:8000/api/v1/results/abc123 -o result.zip
# Downloads: annotated_video.mp4 + analytics.json
```

### **4. Custom Pipeline (Advanced)**

```python
from app.core.pipeline import Pipeline
from app.steps.detection.court_detection import CourtDetectionStep
from app.steps.detection.ball_detection import BallDetectionStep
from app.steps.temporal.gap_filling import GapFillingStep

# Build custom pipeline (only what you need)
custom_pipeline = Pipeline(
    name="BallTrackingOnly",
    steps=[
        BallDetectionStep(config['detection']['ball']),
        GapFillingStep(config['temporal']['gap_filling']),
        # ... add more steps
    ]
)

# Run
context = ProcessingContext(video_path="match.mp4", ...)
result = custom_pipeline.run(context)
```

---

## 🎯 Key Design Principles

### ✅ **1. Universal Model Swapping**

Change neural networks via config (NO code changes):

```yaml
# Use TrackNet for ball detection
detection:
  ball:
    model_type: "tracknet"
    model_path: "models/ball_model_best.pt"

# Swap to custom CNN
detection:
  ball:
    model_type: "custom_cnn"
    model_path: "models/my_ball_detector.onnx"
```

Model registry handles loading:

```python
# app/models/model_registry.py

MODEL_CLASSES = {
    "tracknet": TrackNetModel,
    "yolo": YOLODetector,
    "faster_rcnn": FasterRCNNDetector,
    "custom_cnn": CustomCNNModel,
}

class ModelRegistry:
    @classmethod
    def load_model(cls, model_type: str, model_path: str):
        """Load model by type (defined in config)"""
        model_class = MODEL_CLASSES[model_type]
        return model_class(model_path)
```

### ✅ **2. Simple & Reliable**

- Each step does ONE thing
- Steps are independently testable
- No hidden dependencies
- Clear data flow (context in → context out)

### ✅ **3. Production-Ready Parallel Execution**

```python
# app/services/video_processor.py

class VideoProcessor:
    def process_video(self, video_path: str, parallel: bool = True):
        """
        Process video with optional parallel execution
        """
        if parallel:
            # Use ParallelExecutor for speed
            return self._process_parallel(video_path)
        else:
            # Use sequential execution (easier debugging)
            return self._process_sequential(video_path)

    def _process_parallel(self, video_path: str):
        """
        GPU and CPU pipelines run concurrently on different chunks
        """
        executor = ParallelExecutor(self.config)

        # Split video into chunks
        chunks = self._create_chunks(video_path)

        # Process chunks in parallel
        results = []
        for chunk_context in chunks:
            result = executor.execute_parallel(chunk_context)
            results.append(result)

        # Merge results
        final_result = self._merge_results(results)
        return final_result
```

### ✅ **4. Nested Pipelines**

```python
# app/pipelines/detection_pipeline.py

class DetectionPipeline(Pipeline):
    """
    Sub-pipeline for all detection tasks (GPU-accelerated)
    """
    def __init__(self, config: dict):
        steps = [
            CourtDetectionStep(config['detection']['court']),
            BallDetectionStep(config['detection']['ball']),
            PlayerDetectionStep(config['detection']['player']),
        ]
        super().__init__(name="DetectionPipeline", steps=steps)

# Use in main pipeline
main_pipeline = Pipeline(
    name="MainTennisPipeline",
    steps=[
        PreprocessingPipeline(config),  # Nested!
        DetectionPipeline(config),      # Nested!
        TemporalPipeline(config),       # Nested!
        GeometryPipeline(config),       # Nested!
        EventPipeline(config),          # Nested!
        AnalyticsPipeline(config),      # Nested!
        RenderingPipeline(config),      # Nested!
    ]
)
```

---

## ⚠️ Critical Pitfalls & Solutions

### **1. Temporal Processing Position** ✅ FIXED

**Old (Wrong)**:
```
Detection → Geometry → Temporal
              ↑
          Problem: NaN ball positions break homography
```

**New (Correct)**:
```
Detection → Temporal (gap filling) → Geometry
              ↑
          Fix NaNs first, then homography works on complete data
```

### **2. Homography Caching**

```python
# ❌ BAD: Recompute every frame
for frame in frames:
    H = compute_homography(frame)  # Expensive!

# ✅ GOOD: Compute every 30 frames, cache
for i, frame in enumerate(frames):
    if i % 30 == 0:
        H = compute_homography(frame)
        context.homography_cache[i] = H
    else:
        H = context.get_homography_for_frame(i)  # Use cached
```

### **3. Coordinate System Clarity**

```python
# ❌ BAD: Ambiguous
ball_position: Tuple[float, float]  # Pixels or meters???

# ✅ GOOD: Explicit
ball_position_px: Tuple[float, float]     # Always pixels
ball_position_court: Tuple[float, float]  # Always meters
```

### **4. Gap Filling Validation**

```python
# ❌ BAD: Blind interpolation
ball_positions = np.interp(missing_indices, valid_indices, valid_positions)

# ✅ GOOD: Physics-validated interpolation
def interpolate_with_validation(positions, fps, max_speed=70.0):
    interpolated = np.interp(...)

    # Validate each interpolated point
    for i in range(len(interpolated) - 1):
        dist = np.linalg.norm(interpolated[i+1] - interpolated[i])
        speed = dist * fps  # m/s

        if speed > max_speed:
            # Reject interpolation, mark as NaN
            interpolated[i] = np.nan

    return interpolated
```

### **5. Memory Management**

```python
# ❌ BAD: Load entire video
frames = load_video(video_path)  # OOM for long videos!

# ✅ GOOD: Chunk-based processing
for chunk_start in range(0, total_frames, chunk_size):
    chunk_end = min(chunk_start + chunk_size, total_frames)

    # Load chunk
    frames = load_video_chunk(video_path, chunk_start, chunk_end)

    # Process
    process_chunk(frames)

    # Free memory
    del frames
```

### **6. Parallel GPU Access**

```python
# ❌ BAD: Multiple threads accessing GPU
# (CUDA doesn't support this well)
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(gpu_inference, frames)  # Will crash!

# ✅ GOOD: Single GPU thread with batching
def process_gpu_batch(frames_batch):
    with torch.no_grad():
        results = model(frames_batch)  # Batch inference
    return results

# In executor:
gpu_executor = ThreadPoolExecutor(max_workers=1)  # Single thread!
```

---

## 📊 Database Schema (Player Statistics)

```sql
-- PostgreSQL Schema

-- Matches table
CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    video_path TEXT NOT NULL,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_frames INT,
    duration_seconds FLOAT,
    fps FLOAT
);

-- Players table
CREATE TABLE players (
    id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(id),
    player_index INT,
    name TEXT
);

-- Rallies table
CREATE TABLE rallies (
    id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(id),
    start_frame INT,
    end_frame INT,
    duration_seconds FLOAT,
    winner_player_id INT REFERENCES players(id)
);

-- Shots table
CREATE TABLE shots (
    id SERIAL PRIMARY KEY,
    rally_id INT REFERENCES rallies(id),
    player_id INT REFERENCES players(id),
    frame_id INT,
    timestamp FLOAT,
    shot_type TEXT,  -- 'forehand', 'backhand', 'serve', 'volley'
    ball_speed FLOAT,  -- m/s
    spin_type TEXT,  -- 'flat', 'topspin', 'slice'
    in_out TEXT,  -- 'in', 'out', 'net'
    position_x FLOAT,  -- Court coordinates (meters)
    position_y FLOAT
);

-- Events table
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    rally_id INT REFERENCES rallies(id),
    frame_id INT,
    event_type TEXT,  -- 'hit', 'bounce', 'in', 'out'
    position_x FLOAT,
    position_y FLOAT
);

-- Match statistics (aggregated)
CREATE TABLE match_statistics (
    id SERIAL PRIMARY KEY,
    match_id INT REFERENCES matches(id),
    player_id INT REFERENCES players(id),
    total_shots INT,
    avg_ball_speed FLOAT,
    max_ball_speed FLOAT,
    total_rallies_won INT,
    total_points INT,
    aces INT,
    double_faults INT,
    winners INT,
    unforced_errors INT
);
```

---

## 🧪 Testing Strategy

```python
# tests/unit/test_gap_filling.py
def test_gap_filling_short_gap():
    """Test linear interpolation for gaps < 5 frames"""
    step = GapFillingStep(config)
    context = create_test_context_with_gap(gap_size=3)

    result = step.process(context)

    assert all(det.ball_position_px is not None for det in result.detections)
    assert result.detections[5].is_interpolated == True

# tests/integration/test_detection_temporal_pipeline.py
def test_detection_temporal_integration():
    """Test that detection + temporal pipeline works together"""
    pipeline = Pipeline(steps=[
        DetectionPipeline(config),
        TemporalPipeline(config),
    ])

    context = create_test_context(frames=test_video_frames)
    result = pipeline.run(context)

    # Verify gap filling happened after detection
    assert len(result.detections) == len(test_video_frames)
    assert no_nans_in_trajectory(result)

# tests/e2e/test_full_pipeline.py
def test_full_pipeline_produces_output():
    """End-to-end test: video in, annotated video + stats out"""
    processor = VideoProcessor.from_config("configs/default.yaml")

    result = processor.process_video("tests/data/short_match.mp4")

    assert os.path.exists("results/short_match_analyzed.mp4")
    assert len(result.game_segments) > 0
    assert result.player_statistics['avg_ball_speed'] > 0
```

---

## 🎓 Future Extensions

### **1. Shot Type Classification** (Easy to add)

```yaml
# In configs/default.yaml, add:
events:
  shot_type:
    enabled: true
    model_path: "models/shot_type_model.pt"
    model_type: "shot_type_classifier"
    window_size: 10  # ±10 frames = 21 total
```

```python
# Create app/steps/events/shot_type_classification.py
class ShotTypeClassificationStep(PipelineStep):
    def process(self, context):
        for hit_frame in context.hit_events:
            window = context.temporal_windows[hit_frame]
            shot_type = self.model.predict(window)
            context.detections[hit_frame].shot_type = shot_type
        return context

# Add to EventPipeline
class EventPipeline(Pipeline):
    def __init__(self, config):
        steps = [
            BallHitDetectionStep(...),
            ShotTypeClassificationStep(...),  # ← Just add here!
            BounceDetectionStep(...),
            ...
        ]
```

### **2. Real-Time Streaming Mode**

```python
# app/services/stream_processor.py
class StreamProcessor:
    """Process live video stream (webcam, RTSP)"""

    def process_stream(self, stream_url: str):
        cap = cv2.VideoCapture(stream_url)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process single frame (no future context)
            context = self.create_frame_context(frame)
            result = self.pipeline.run(context)

            # Display immediately
            annotated = self.render(result)
            cv2.imshow("Live Analysis", annotated)
```

### **3. Multi-Camera Fusion**

```python
# Future: Combine views from multiple cameras
class MultiCameraFusionPipeline(Pipeline):
    def process(self, context_list: List[ProcessingContext]):
        # Fuse detections from multiple viewpoints
        # Triangulate 3D ball position
        pass
```

---

## 📚 Documentation

- **API Docs**: Auto-generated with FastAPI at `http://localhost:8000/docs`
- **Code Docs**: Sphinx documentation in `docs/`
- **Architecture**: This README.md
- **Tutorials**: `notebooks/tutorials/`

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model weights
python scripts/download_models.py

# 3. Process a video
python -m app.main process tests/data/sample_match.mp4

# 4. Start API server (optional)
python -m app.api.main

# 5. Run tests
pytest tests/
```

---

## 🎯 Summary

This architecture provides:

✅ **Simple**: Each step does one thing, easy to understand
✅ **Reliable**: Independent components, comprehensive testing
✅ **Fast**: Parallel GPU/CPU execution, optimized batching
✅ **Flexible**: Swap models via config, add features by adding steps
✅ **Production-Ready**: Database export, API, monitoring, Docker support
✅ **Maintainable**: Clear structure, nested pipelines, type-safe

**Key Innovation**: Nested pipelines + parallel execution + shared context = maximum flexibility with minimal complexity.
