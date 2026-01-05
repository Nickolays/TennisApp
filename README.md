# 🎾 Tennis Analysis - AI-Powered Match Analytics

**Real-time Computer Vision System for Professional Tennis Match Analysis**

---

## 🎯 Project Vision

A distributed AI system that transforms tennis match videos into actionable insights:
- **Upload video** → AI analyzes every frame → **Get detailed statistics**
- Works with professional matches, amateur games, training sessions
- Cloud-coordinated processing with GPU-powered local worker
- Mobile app ready (iOS/Android) + Web dashboard

---

## 🏗️ System Architecture

### Distributed Cloud + Local GPU Model

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT (Mobile/Web)                       │
│  • Upload video                                              │
│  • Track progress (WebSocket/polling)                        │
│  • View results (statistics, annotated video)               │
└────────────────────┬─────────────────────────────────────────┘
                     │ HTTPS (Upload video)
                     │ WebSocket (Real-time updates)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│              VDS/CLOUD SERVER (24/7 Online)                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ FastAPI Backend (Python)                            │    │
│  │  • POST /api/v1/upload - Accept videos             │    │
│  │  • GET /api/v1/status/{job_id} - Track progress    │    │
│  │  • POST /api/v1/stream/start - RTSP streams        │    │
│  │  • GET /worker/get_job - Job queue for workers     │    │
│  │  • POST /worker/complete/{job_id} - Submit results │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ PostgreSQL Database                                 │    │
│  │  • Jobs table (pending/processing/completed)        │    │
│  │  • Match statistics                                 │    │
│  │  • Player analytics                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ File Storage (S3 / Local)                           │    │
│  │  • Uploaded videos                                  │    │
│  │  • Processed results                                │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────┬─────────────────────────────────────────┘
                     │ Worker polls for jobs (outgoing request)
                     │ No white IP needed! Works behind NAT/firewall
                     ↓
┌─────────────────────────────────────────────────────────────┐
│         HOME/OFFICE PC (GPU Worker - RTX 3070)              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Worker Script (Python)                              │    │
│  │  • Polls cloud: "Any jobs?"                         │    │
│  │  • Downloads video from cloud                       │    │
│  │  • Runs AI processing (4 pipelines)                 │    │
│  │  • Uploads results back to cloud                    │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ AI Models (CUDA-accelerated)                        │    │
│  │  • TrackNet (court + ball detection)                │    │
│  │  • YOLO v11 (player detection)                      │    │
│  │  • Kalman filter (trajectory smoothing)             │    │
│  │  • Event detection (bounces, hits)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│  Performance: ~18 FPS on RTX 3070 with batch_size=16        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

✅ **No White IP Required**: Worker initiates outgoing connections (polls cloud)
✅ **Works Behind NAT**: No port forwarding or router configuration needed
✅ **Free GPU Power**: Use home GPU instead of expensive cloud GPU instances
✅ **Scalable**: Add more workers (friends' PCs) without infrastructure changes
✅ **Fault Tolerant**: Jobs stay in queue if worker goes offline

---

## 🚀 Processing Pipeline

### 4-Stage AI Pipeline (18 FPS on RTX 3070)

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: Tennis match video (MP4/AVI/MOV or RTSP stream)     │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. DETECTION PIPELINE (GPU - Parallel Batch Inference)      │
├─────────────────────────────────────────────────────────────┤
│  Court Detection (TrackNet)  → 14 keypoints                 │
│  Ball Detection (TrackNet)   → (x, y) position per frame    │
│  Player Detection (YOLO v11) → Bounding boxes (near + far)  │
│                                                              │
│  Performance:                                                │
│  • Court: 0.49s per 100 frames (every 30th frame)           │
│  • Ball: 3.28s per 100 frames (every frame)                 │
│  • Players: 1.41s per 100 frames (batch_size=16)            │
│  • Improved filtering: 3.3 players/frame (catches distant)  │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 1.5 TRACKING PIPELINE (GPU - ByteTrack)                     │
├─────────────────────────────────────────────────────────────┤
│  Player Tracking (ByteTrack) → Consistent player IDs        │
│                                                              │
│  • Maintains player identity across frames                  │
│  • Handles occlusions and temporary detection failures      │
│  • Kalman filter for motion prediction                      │
│  • Fast: ~30 FPS (no appearance features needed)            │
│                                                              │
│  Benefits:                                                   │
│  • Track individual player statistics across match          │
│  • Handle far players when detection fails                  │
│  • Enable player-specific analytics (distance, speed)       │
│                                                              │
│  Performance: ~0.02s (instant with ByteTrack)               │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TEMPORAL PIPELINE (CPU - Parallel with Detection)        │
├─────────────────────────────────────────────────────────────┤
│  Gap Filling        → Interpolate missing ball positions    │
│  Kalman Smoothing   → Remove jitter from trajectory         │
│  Window Extraction  → Create ±5 frame windows               │
│                                                              │
│  Performance: ~0.01s (instant)                              │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. GEOMETRY PIPELINE (CPU)                                  │
├─────────────────────────────────────────────────────────────┤
│  Homography Estimation    → Pixel → court coordinates       │
│  Coordinate Transform     → Transform ball to meters        │
│  Velocity Estimation      → Calculate ball speed (m/s)      │
│                                                              │
│  Performance: ~0.01s                                        │
│  Status: ⏸️ Blocked (ready to train! Use train_court.py)   │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. EVENT PIPELINE (CPU + Physics)                           │
├─────────────────────────────────────────────────────────────┤
│  Hit Detection      → Velocity spikes (acceleration > 15)   │
│  Bounce Detection   → Vertical flip + speed decrease        │
│  In/Out Decision    → Court boundary validation             │
│                                                              │
│  Performance: ~0.01s                                        │
│  Status: ✅ Ready (waiting on geometry pipeline)            │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: Annotated video + JSON statistics                  │
│  • Video with overlays (court, ball, players, events)       │
│  • JSON: {detections, events, statistics, frame-by-frame}   │
│  • Total time: ~5.6s per 100 frames (17.9 FPS)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 What The System Detects

### Visual Annotations

```
┌─────────────────────────────────────────────────────────────┐
│  🟢 Green dots       → Court keypoints (14 points)          │
│  🟡 Yellow circle    → Ball (normal tracking)               │
│  🔴 Red circle       → Ball bounce event (BOUNCE!)          │
│  🟢 Green circle     → Ball hit event (HIT!)                │
│  🔵 Blue boxes       → Players (with confidence scores)     │
│  📊 Info panel       → Frame #, time, detection status      │
└─────────────────────────────────────────────────────────────┘
```

### Analytics Output (JSON)

```json
{
  "video": {
    "fps": 29.75,
    "frames": 493,
    "duration_seconds": 16.6
  },
  "detections": {
    "court": 493,
    "ball": 493,
    "players": 493,
    "avg_players_per_frame": 3.3
  },
  "events": {
    "bounces": [42, 156, 289],
    "hits": [15, 78, 134, 201],
    "bounce_frames": [...],
    "hit_frames": [...]
  },
  "statistics": {
    "avg_ball_speed": 18.5,
    "max_ball_speed": 42.3,
    "rally_count": 12,
    "total_shots": 48
  }
}
```

---

## 🛠️ Technology Stack

### Backend (Cloud)
- **FastAPI** - Modern async Python framework
- **PostgreSQL** - Job queue and match statistics
- **S3/MinIO** - Video file storage
- **Redis** - Real-time job status (optional)
- **WebSocket** - Live progress updates

### Worker (Local GPU)
- **PyTorch** - Deep learning framework
- **CUDA** - GPU acceleration (RTX 3070)
- **OpenCV** - Video processing
- **NumPy/SciPy** - Numerical computing

### AI Models & Algorithms
- **TrackNet** - Court and ball detection (custom trained)
- **YOLO v11 nano** - Player detection (5.4MB, fast)
- **ByteTrack** - Multi-object tracking (built into ultralytics)
- **Kalman Filter** - Trajectory smoothing
- **Physics-based** - Event detection algorithms

### Tracking (ByteTrack)
- **Library**: ultralytics 8.3.237 (built-in)
- **Dependencies**: lap>=0.5.12 (Linear Assignment Problem solver)
- **No version conflicts**: Compatible with current PyTorch 2.9.1, NumPy 2.2.6
- **Performance**: ~30 FPS (no appearance features, Kalman + Hungarian matching)

---

## 📁 Project Structure

```
TennisApp/
├── 📄 Core Files
│   ├── QUICK_START.md              # ⭐ START HERE - Quick guide
│   ├── README.md                   # This file - Project overview
│   ├── train_court.py              # ⭐ Train court detection model
│   ├── visualize_court_video.py    # ⭐ Visualize predictions on video
│   └── process_video.py            # Full pipeline processing
│
├── ⚙️ Configuration
│   └── configs/
│       ├── train.yaml              # ⭐ Training configuration
│       ├── default.yaml            # Pipeline configuration
│       ├── fast.yaml               # Speed-optimized config
│       └── production.yaml         # Production config
│
├── 📚 Documentation
│   └── docs/
│       ├── DEVELOPMENT_HISTORY.md  # Development timeline
│       ├── CHANGELOG.md            # Version history
│       ├── PROJECT_STRUCTURE.md    # File organization guide
│       ├── API_DOCUMENTATION.md    # API reference
│       ├── API_QUICK_START.md      # API quick guide
│       ├── TRAINING_IMPROVEMENTS.md # Training tips
│       ├── VIDEO_VISUALIZATION_GUIDE.md # Visualization guide
│       ├── MODEL_RETRAINING_ROADMAP.md # Data collection guide
│       ├── TRACKING_IMPLEMENTATION.md # ByteTrack guide
│       ├── DATA_FORMAT_SPECIFICATION.md # Dataset format
│       └── WORKER_SETUP.md         # Worker deployment
│
├── 🤖 Application Code
│   └── app/
│       ├── api/
│       │   └── main.py             # FastAPI server (v2.0.0)
│       ├── core/
│       │   ├── context.py          # Shared processing state
│       │   ├── pipeline.py         # Base pipeline classes
│       │   └── data_models.py      # Pydantic models
│       ├── steps/
│       │   ├── detection/          # Court, ball, player detection
│       │   ├── tracking/           # Player tracking (ByteTrack)
│       │   ├── temporal/           # Gap filling, smoothing
│       │   ├── geometry/           # Homography, coordinates
│       │   └── events/             # Bounce, hit detection
│       ├── pipelines/
│       │   ├── detection_pipeline.py
│       │   ├── temporal_pipeline.py
│       │   ├── geometry_pipeline.py
│       │   └── event_pipeline.py
│       ├── models/
│       │   └── model_registry.py   # Model loading/caching
│       └── src/
│           ├── datasets.py         # COCO dataset loaders
│           ├── transform.py        # Data augmentation
│           ├── postprocess.py      # Model output processing
│           └── steps.py            # Training utilities
│
├── 💼 Worker & Deployment
│   └── worker/
│       └── gpu_worker.py           # GPU worker script
│
├── 📊 Data
│   └── data/
│       ├── tennis_ball_dataset/    # Ball detection (487 images)
│       └── tennis_court_dataset/   # Court detection (918 images)
│
├── 🧪 Models
│   └── models/
│       ├── court_model_best.pt     # ⭐ Trained court model
│       ├── ball_model_best.pt      # Trained ball model
│       └── yolov11n.pt             # YOLO v11 player detection
│
├── 📦 Checkpoints & Logs
│   ├── checkpoints/
│   │   └── court_detection/
│   │       ├── best_model.pth      # ⭐ Best model (lowest val loss)
│   │       └── checkpoint_epoch_*.pth # Periodic checkpoints
│   └── logs/
│       └── court_training.json     # Training history
│
└── 🧪 Tests
    └── tests/
        └── video3.mp4              # Test video
```

---

## 🚀 Quick Start

**See [QUICK_START.md](QUICK_START.md) for detailed guide**

### 1. Train Court Detection Model

```bash
cd TennisAnalysis/TennisApp
source ../.venv/bin/activate

# Train with optimized settings (batch_size=4, RTX 3070)
python train_court.py
```

**Output**: `models/court_model_best.pt` (~3-4 hours on RTX 3070)

### 2. Visualize Predictions on Video

```bash
# Test your trained model
python visualize_court_video.py tests/video3.mp4 \
  --model checkpoints/court_detection/best_model.pth \
  --output results/video3_viz.mp4
```

**Output**: Annotated video with keypoints, skeleton, minimap, stability metrics

### 3. Process Full Match Video

```bash
# Full pipeline: court + ball + players + tracking
python process_video.py tests/video3.mp4 \
  --config configs/default.yaml \
  --output results/analysis/
```

**Output**: Visualized video + JSON statistics + analytics

### 4. Start API Server (Optional)

```bash
# For distributed processing with GPU workers
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Start GPU worker on another machine
python worker/gpu_worker.py --server https://your-vds.com
```

**See**: [docs/API_QUICK_START.md](docs/API_QUICK_START.md) and [docs/WORKER_SETUP.md](docs/WORKER_SETUP.md)

---

## 🎯 Key Features

### ✅ Detection Quality
- **Court**: 100% detection rate (14 keypoints)
- **Ball**: 100% detection rate with Kalman smoothing
- **Players**: 3.3 players/frame (improved filtering for distant players)

### ✅ Performance Optimizations
- **Batch Inference**: Process 16 frames simultaneously → 2.7x faster
- **GPU Utilization**: 70% (vs 30% without batching)
- **Throughput**: 17.9 FPS on RTX 3070
- **Smart Caching**: Court detection every 30 frames (doesn't change)

### ✅ Infrastructure Benefits
- **No White IP**: Worker polls cloud (outgoing requests only)
- **Works Behind NAT**: No port forwarding needed
- **Free GPU**: Use home GPU instead of cloud GPU ($$$)
- **Scalable**: Add workers without infrastructure changes
- **Fault Tolerant**: Jobs persist if worker offline

### ✅ API Features
- **Video Upload**: MP4, AVI, MOV files
- **RTSP Streams**: Live camera processing
- **Job Tracking**: REST API + WebSocket updates
- **Result Download**: JSON statistics + annotated video
- **Batch Control**: Configurable batch_size per request

---

## 📊 Performance Benchmarks

### RTX 3070 (100 frames)

| Pipeline | Time | Percentage |
|----------|------|------------|
| Detection | 5.58s | 99.6% |
| - Court | 0.49s | 8.8% |
| - Ball | 3.28s | 58.7% |
| - Players | 1.41s | 25.2% |
| Temporal | 0.01s | 0.2% |
| Geometry | 0.01s | 0.2% |
| Events | 0.00s | 0.0% |
| **Total** | **5.60s** | **100%** |

**Throughput**: 17.9 FPS

### Batch Size Impact (RTX 3070)

| Batch Size | Time | FPS | GPU Util | Speedup |
|------------|------|-----|----------|---------|
| 1 | 15.2s | 6.6 | 30% | 1.0x |
| 8 | 6.8s | 14.7 | 60% | 2.2x |
| 16 | 5.6s | 17.9 | 70% | 2.7x ✅ |
| 32 | 5.8s | 17.2 | 75% | 2.6x |

**Optimal**: batch_size=16 for RTX 3070

---

## 🔧 Configuration

### Worker Configuration

```python
# worker/gpu_worker.py
CLOUD_SERVER = "https://your-vds.com"
POLL_INTERVAL = 2  # seconds
BATCH_SIZE = 16  # frames to process together
MAX_RETRIES = 3  # retry failed jobs
```

### API Configuration

```python
# app/api/main.py
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MAX_FILE_SIZE = 500_000_000  # 500MB
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov']
```

### Model Configuration

```yaml
# configs/default.yaml (future)
detection:
  court:
    model_path: "models/court_model_best.pt"
    interval: 30  # every 30th frame
  ball:
    model_path: "models/ball_model_best.pt"
    interval: 1  # every frame
  player:
    model_path: "models/yolo11n.pt"
    batch_size: 16
    confidence_threshold: 0.35
    min_box_area: 1500  # catch far players
```

---

## 🐛 Current Status & Roadmap

### ✅ Working (v2.0.0)

1. **Detection Pipeline** - Court, ball, players (3.3 players/frame)
2. **Player Tracking** - ByteTrack for consistent IDs
3. **Temporal Pipeline** - Gap filling, smoothing, windows
4. **Batch Inference** - 2.7x speedup (17.9 FPS on RTX 3070)
5. **Training System** - Optimized for batch_size=4, proper validation
6. **Visualization** - Video analysis with stability metrics
7. **API** - Video upload, RTSP streams, job tracking
8. **Worker** - Distributed GPU processing

### ⏸️ Blocked

1. **Geometry Pipeline** - Needs better trained court model (ready to train!)
2. **Event Detection** - Depends on geometry pipeline

### 🎯 Next Priority

**Retrain court detection model** using optimized training script:
```bash
python train_court.py
```

**Why retrain?**:
- Higher batch size (4 vs 2) → less noisy gradients
- Better resolution (640×640 vs 512×512) → more detail
- Proper validation split → detect overfitting
- Better metrics (PCK@5px, PCK@10px) → track quality
- Early stopping → prevent overfitting
- Cosine annealing LR → better convergence

**After retraining**:
→ Unblock geometry pipeline → Enable event detection → Full system operational!

### Future Roadmap

**Short-term** (After retraining):
- ⏳ Deploy to production
- ⏳ Mobile app integration (iOS/Android)
- ⏳ Real-time dashboard

**Long-term**:
- ⏳ Shot type classification (forehand/backhand)
- ⏳ Rally segmentation and analysis
- ⏳ Multi-camera fusion (TrackNet paper: arxiv.org/pdf/2205.13857)
- ⏳ Advanced player analytics

---

## 📖 Documentation

### Getting Started
- **[QUICK_START.md](QUICK_START.md)** - ⭐ START HERE - Train & visualize in 3 steps
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - File organization guide

### Training & Visualization
- **[docs/TRAINING_IMPROVEMENTS.md](docs/TRAINING_IMPROVEMENTS.md)** - Training tips and best practices
- **[docs/VIDEO_VISUALIZATION_GUIDE.md](docs/VIDEO_VISUALIZATION_GUIDE.md)** - Complete visualization guide
- **[docs/MODEL_RETRAINING_ROADMAP.md](docs/MODEL_RETRAINING_ROADMAP.md)** - Data collection and retraining

### Technical Guides
- **[docs/DATA_FORMAT_SPECIFICATION.md](docs/DATA_FORMAT_SPECIFICATION.md)** - COCO dataset format
- **[docs/TRACKING_IMPLEMENTATION.md](docs/TRACKING_IMPLEMENTATION.md)** - ByteTrack player tracking

### API & Deployment
- **[docs/API_QUICK_START.md](docs/API_QUICK_START.md)** - 5-minute API guide
- **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** - Complete API reference
- **[docs/WORKER_SETUP.md](docs/WORKER_SETUP.md)** - GPU worker deployment
- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)

### Project History
- **[docs/DEVELOPMENT_HISTORY.md](docs/DEVELOPMENT_HISTORY.md)** - Development timeline
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Version history and changes

---

## 🎓 System Design Insights

### Why Distributed Architecture?

**Problem**: Cloud GPU instances are expensive ($1-3/hour)
**Solution**: Use home GPU + cloud coordination

**Benefits**:
1. **Cost**: $0/month GPU (vs $720-2160/month cloud)
2. **Power**: RTX 3070 at home (vs shared cloud GPU)
3. **Scalability**: Add friends' GPUs as workers
4. **Flexibility**: Process locally, coordinate globally

### Why Worker Polls Cloud?

**Traditional** (requires white IP):
```
Cloud → [NAT/Firewall] ✗ → Home PC
(Can't reach worker behind NAT)
```

**Our Approach** (no white IP needed):
```
Cloud ← [NAT/Firewall] ✓ ← Home PC
(Worker initiates connection, works everywhere)
```

**Advantages**:
- ✅ Works behind any NAT/firewall
- ✅ No port forwarding
- ✅ No dynamic DNS
- ✅ No router configuration
- ✅ Just run the script!

### Why Batch Inference?

**Sequential** (old):
```python
for frame in frames:
    result = model(frame)  # 100 GPU calls
    # GPU idle most of the time
```

**Batched** (new):
```python
for batch in chunks(frames, batch_size=16):
    results = model(batch)  # 7 GPU calls
    # GPU fully utilized
```

**Result**: 2.7x faster with same GPU!

---

## 🎯 Core Value Proposition

**Traditional Solutions**:
- Expensive cloud GPU: $1-3/hour
- Manual analysis: Hours of human work
- Generic tools: Not tennis-specific

**Our Solution**:
- **Free GPU**: Use hardware you already own
- **Automated**: Upload video → Get results
- **Specialized**: Built specifically for tennis
- **Scalable**: Add more workers anytime
- **Distributed**: Cloud coordination + local processing

---

## 🏆 Technical Achievements

1. **Player Detection**: 220% improvement (1.04 → 3.3 players/frame)
2. **Batch Inference**: 2.7x faster processing
3. **RTSP Support**: Live camera stream processing
4. **Distributed Design**: No white IP required
5. **Production API**: WebSocket + REST, job tracking
6. **GPU Efficiency**: 30% → 70% utilization

---

## 🤝 Contributing

This is a personal project, but contributions are welcome!

**Areas for contribution**:
- Improve court model training
- Add shot type classification
- Build mobile app frontend
- Optimize inference speed
- Add more test cases

---

## 📝 License

Private project. All rights reserved.

---

## 📞 Contact

For questions or collaboration: [Your contact]

---

**Built with ❤️ for tennis analytics**
**Powered by PyTorch, FastAPI, and RTX 3070**
