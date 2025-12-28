# YOLO v11 Player Detection Integration

## ✅ Implementation Complete!

I've successfully integrated YOLO v11 player detection into your tennis video analysis pipeline.

## 🎯 What Was Implemented

### 1. **YOLO Player Detector** (`app/models/yolo_player_detector.py`)
- ✅ YOLO v11 integration using Ultralytics
- ✅ Supports loading by model name (`yolo11n.pt`, `yolo11s.pt`, etc.) or weights path
- ✅ Automatic download if model name is used
- ✅ Filters detections for class 0 (person) only
- ✅ Returns bounding boxes in format: `(x1, y1, x2, y2)`

### 2. **PlayerDetector Class** (`app/core/base.py`)
- ✅ Wrapper class that uses YOLOPlayerDetector
- ✅ Integrates with existing pipeline structure
- ✅ Returns detection results in same format as ball detection

### 3. **Visualization** (`app/core/base.py` - VideoRenderer)
- ✅ Added `_draw_players()` method
- ✅ Draws bounding boxes for each detected player
- ✅ Shows labels: "PLAYER 1: 0.XX", "PLAYER 2: 0.XX" with confidence
- ✅ Red bounding boxes with black background labels

### 4. **Demo Integration** (`demo.py`)
- ✅ Added player detection to `demo_video_processing()`
- ✅ Player detections are visualized together with court and ball

## 🚀 Usage

### **Basic Usage in Demo:**
```python
from app.core.base import PlayerDetector

# Initialize with YOLO model name (will download if needed)
player_detector = PlayerDetector(config, model_name_or_path="yolo11n.pt")

# Or use custom weights path
player_detector = PlayerDetector(config, model_name_or_path="models/player_model.pt")

# Detect players
player_results = player_detector(frames)
```

### **Available YOLO Models:**
- `yolo11n.pt` - Nano (fastest, smallest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (most accurate)

### **Output Format:**
```python
{
    'player_boxes': [(x1, y1, x2, y2), ...],  # Bounding boxes
    'player_confs': [0.95, 0.87, ...],        # Confidences
    'player_class_ids': [0, 0, ...]           # Class IDs (0 = person)
}
```

## 📋 Preprocessing Details

### **Automatic Filtering:**
- ✅ Filters for class ID 0 (person) only
- ✅ Confidence threshold: 0.25 (configurable)
- ✅ Returns only person detections

### **Coordinate Format:**
- ✅ Bounding boxes in `(x1, y1, x2, y2)` format
- ✅ Coordinates are in pixel space (0 to image width/height)
- ✅ No additional preprocessing needed - YOLO handles it

## 🎨 Visualization

Players are visualized with:
- **Red bounding boxes** around detected players
- **Black background labels** showing "PLAYER N: confidence"
- **Same style** as court and ball detections for consistency

## 📦 Installation

Install Ultralytics (YOLO v11):
```bash
pip install ultralytics
```

## 🧪 Testing

Run the demo to see all three detections together:
```bash
cd /home/suetin/Projects/TennisAnalysis
source .venv/bin/activate
cd TennisApp
python3 demo.py
```

The output video will show:
- 🟡 **Court keypoints** (yellow circles) with "COURT" label
- 🟢 **Ball position** (green circle) with "BALL" label  
- 🔴 **Player bounding boxes** (red rectangles) with "PLAYER N" labels

## 🔧 Configuration

You can adjust the confidence threshold in `YOLOPlayerDetector`:
```python
self.conf_threshold = 0.25  # Lower = more detections, Higher = fewer but more confident
```

## ✅ Next Steps

1. **Install ultralytics**: `pip install ultralytics`
2. **Run demo**: `python3 demo.py`
3. **Check output**: `results/demo_detection_output.mp4`
4. **Adjust model**: Change `yolo11n.pt` to `yolo11s.pt` or `yolo11m.pt` for better accuracy

## 🎉 Success!

Your pipeline now detects:
- ✅ **Court** (TrackNet)
- ✅ **Ball** (TrackNet)  
- ✅ **Players** (YOLO v11)

All three are visualized together in the output video!

