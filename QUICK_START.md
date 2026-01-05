# 🎾 Tennis Analysis - Quick Start

**Train court detection model and visualize predictions**

---

## Prerequisites

```bash
cd TennisAnalysis/TennisApp
source ../.venv/bin/activate
```

---

## 1. Train Court Detection Model

Train a model to detect 14 court keypoints:

```bash
python train_court.py
```

**What happens**:
- Trains for 50 epochs (~3-4 hours on RTX 3070)
- Saves best model to `models/court_model_best.pt`
- Checkpoints saved to `checkpoints/court_detection/`
- Training logs: `logs/court_training.json`

**Configuration**: Edit [configs/train.yaml](configs/train.yaml) to adjust batch size, epochs, learning rate.

---

## 2. Visualize Predictions on Video

Test your trained model on a video:

```bash
python visualize_court_video.py \
  tests/video3.mp4 \
  --model checkpoints/court_detection/best_model.pth \
  --output results/video3_viz.mp4
```

**What you'll see**:
- 14 court keypoints (green circles)
- Court skeleton (yellow lines)
- Top-down minimap (bottom-right)
- Stability metrics (frame-to-frame movement)

**Fast preview** (10x faster):
```bash
python visualize_court_video.py tests/video3.mp4 \
  --model checkpoints/court_detection/best_model.pth \
  --output results/preview.mp4 \
  --skip-frames 10 --max-frames 100
```

**Export keypoints to JSON**:
```bash
python visualize_court_video.py tests/video3.mp4 \
  --model checkpoints/court_detection/best_model.pth \
  --save-json --json-output results/keypoints.json
```

---

## 3. Full Tennis Analysis Pipeline

Process complete match video (court + ball + players + tracking):

```bash
python process_video.py tests/video3.mp4 \
  --config configs/default.yaml \
  --output results/analysis/
```

**Output**:
- `video3_visualized.mp4` - Annotated video
- `video3_results.json` - Complete analysis data
- `video3_stats.json` - Statistics summary

---

## Configuration Files

**Training**: [configs/train.yaml](configs/train.yaml)
```yaml
training:
  batch_size: 4        # Adjust for your GPU memory
  num_epochs: 50       # Training duration
  initial_lr: 0.001    # Learning rate
```

**Pipeline**: [configs/default.yaml](configs/default.yaml)
```yaml
detection:
  court:
    enabled: true
    model_path: "models/court_model_best.pt"
  ball:
    enabled: true
  player:
    enabled: true

tracking:
  players:
    enabled: true
    tracker: "bytetrack"
```

---

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `batch_size` in [configs/train.yaml](configs/train.yaml) from 4 to 2
- Or reduce `target_size` from `[640, 640]` to `[512, 512]`

**Poor Model Quality** (Mean error > 10px):
- Train longer: Increase `num_epochs` to 100
- See [docs/TRAINING_IMPROVEMENTS.md](docs/TRAINING_IMPROVEMENTS.md)

**Unstable Keypoints** (Stability > 5px):
- Retrain model with more epochs
- Check model quality with `--save-json` option

---

## Documentation

**Quick Reference**:
- **Training**: `python train_court.py`
- **Visualization**: `python visualize_court_video.py <video> --model <model> --output <out>`
- **Full Pipeline**: `python process_video.py <video>`

**Detailed Guides**:
- [docs/VIDEO_VISUALIZATION_GUIDE.md](docs/VIDEO_VISUALIZATION_GUIDE.md) - Complete visualization options
- [docs/TRAINING_IMPROVEMENTS.md](docs/TRAINING_IMPROVEMENTS.md) - Training tips
- [docs/MODEL_RETRAINING_ROADMAP.md](docs/MODEL_RETRAINING_ROADMAP.md) - Data collection guide
- [docs/TRACKING_IMPLEMENTATION.md](docs/TRACKING_IMPLEMENTATION.md) - ByteTrack player tracking
- [README.md](README.md) - Complete project overview

---

## Tips

**Training**:
1. Start with 10 epochs for quick test
2. Monitor validation loss (should decrease smoothly)
3. Target PCK@10px > 95% for good model

**Visualization**:
1. Use `--skip-frames 10` for fast preview
2. Check stability metric (should be < 3px)
3. Export JSON for quantitative analysis

**Pipeline**:
1. Process small clips first to verify settings
2. Use GPU for best performance
3. Check GPU usage with `nvidia-smi`

---

**Ready to start? Run**:
```bash
python train_court.py
```

Good luck! 🚀
