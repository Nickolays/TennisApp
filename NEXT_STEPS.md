# Next Steps - Tennis Video Analysis Pipeline

## ✅ Completed: Court Detection Training

You now have a working court keypoint detection model with:
- 5-10px mean error
- 95-98% PCK@10px accuracy
- Stable training (no collapse)
- Best model saved at: `checkpoints/court_detection/best_model.pth`

## 🎯 Next Steps Overview

### 1. Test Court Detection on Real Videos
**Goal:** Verify the trained model works on actual tennis match footage

**Steps:**
```bash
# Process a video with court detection
python process_video.py \
  --input path/to/tennis_match.mp4 \
  --output output/court_keypoints.json \
  --model checkpoints/court_detection/best_model.pth \
  --visualize  # Optional: create visualization video
```

**What to check:**
- ✓ Keypoints detected on all frames
- ✓ Stable detection (no jitter)
- ✓ Correct court line alignment
- ✓ Works with different camera angles

**If detection fails:**
- Check if test video is similar to training data
- May need to add more diverse training images
- Consider data augmentation for robustness

---

### 2. Compute Homography Matrix
**Goal:** Transform court coordinates to bird's-eye view (top-down perspective)

**Background:**
- Court keypoints → 2D image coordinates (x, y)
- Need to map to real-world court coordinates
- Homography = 3x3 transformation matrix
- Enables measuring distances, positions, trajectories

**Implementation steps:**

#### A. Define Standard Court Coordinates
Create reference court template:

```python
# Standard tennis court dimensions (in meters or feet)
# See: https://en.wikipedia.org/wiki/Tennis_court
COURT_TEMPLATE = {
    # 14 keypoints in real-world coordinates
    # Format: (x, y) in court coordinate system
    0: (0, 0),           # Top-left corner
    1: (10.97, 0),       # Top service line left
    2: (10.97, 23.77),   # Top service line right
    # ... add all 14 keypoints
}
```

**Resources:**
- Tennis court dimensions: 23.77m × 10.97m (doubles)
- Service box: 6.40m × 4.115m
- Reference: docs/court_dimensions.png (if available)

#### B. Implement Homography Computation

**File to create:** `app/src/homography.py`

```python
import cv2
import numpy as np

def compute_homography(src_points, dst_points):
    """
    Compute homography matrix from detected keypoints to court template.

    Args:
        src_points: (N, 2) detected keypoints in image coordinates
        dst_points: (N, 2) corresponding points in court coordinates

    Returns:
        H: (3, 3) homography matrix
        success: bool indicating if computation succeeded
    """
    # Need at least 4 points for homography
    assert len(src_points) >= 4
    assert len(src_points) == len(dst_points)

    # Compute homography using RANSAC (robust to outliers)
    H, status = cv2.findHomography(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    success = H is not None and np.sum(status) >= 4
    return H, success

def transform_points(points, H):
    """Transform points using homography matrix."""
    points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
    transformed = (H @ points_homogeneous.T).T
    transformed = transformed[:, :2] / transformed[:, 2:]
    return transformed

def create_court_visualization(court_points, player_positions=None):
    """
    Create bird's-eye view court visualization.

    Args:
        court_points: (14, 2) court keypoints in court coordinates
        player_positions: Optional (N, 2) player positions

    Returns:
        court_img: RGB image with court drawn
    """
    # Create blank canvas
    scale = 20  # pixels per meter
    width = int(23.77 * scale)
    height = int(10.97 * scale)
    court_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw court lines
    # ... implement court line drawing ...

    # Draw players if provided
    if player_positions is not None:
        for pos in player_positions:
            x, y = int(pos[0] * scale), int(pos[1] * scale)
            cv2.circle(court_img, (x, y), 5, (0, 255, 0), -1)

    return court_img
```

#### C. Integrate into Processing Pipeline

**Update:** `process_video.py`

```python
from app.src.homography import compute_homography, transform_points

# After detecting court keypoints
court_keypoints = model(frame)  # (14, 2)

# Compute homography
H, success = compute_homography(court_keypoints, COURT_TEMPLATE)

if success:
    # Transform player positions to court coordinates
    player_positions_court = transform_points(player_positions, H)

    # Save or visualize
    results['homography'] = H.tolist()
    results['player_positions_court'] = player_positions_court.tolist()
```

---

### 3. Player Detection & Tracking
**Goal:** Detect and track player positions throughout the match

**Options:**

#### Option A: Use Pre-trained Detector (Recommended)
```bash
# YOLOv8 for person detection
pip install ultralytics

# In code
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Nano model for speed

# Detect players
results = model(frame, classes=[0])  # class 0 = person
player_boxes = results[0].boxes.xyxy  # Bounding boxes
```

#### Option B: Train Custom Detector
- Collect tennis player dataset
- Fine-tune YOLO or similar detector
- More accurate but requires labeled data

**Tracking:**
```python
from ultralytics import YOLO

# YOLO has built-in tracking
model = YOLO('yolov8n.pt')
results = model.track(frame, persist=True)

for box in results[0].boxes:
    track_id = int(box.id)  # Unique player ID
    x1, y1, x2, y2 = box.xyxy[0]
    # Player center position
    player_x = (x1 + x2) / 2
    player_y = y2  # Bottom of box = feet position
```

---

### 4. Ball Detection & Tracking
**Goal:** Detect tennis ball and track its trajectory

**Challenges:**
- Ball is small and fast-moving
- Motion blur
- Occlusions (behind players, net)

**Approach:**

#### Option A: TrackNet (Ball-specific)
Already have TrackNet in codebase!

```python
# app/models/tracknet.py already exists
from app.models.tracknet import TrackNet

# Train ball detection model
# Similar to court detection but output = ball heatmap
ball_model = TrackNet(out_channels=1)  # Single heatmap for ball

# Need ball trajectory dataset:
# - https://github.com/yastrebksv/TennisCourtDetector (has ball annotations)
# - Or label manually using Label Studio
```

#### Option B: YOLOv8 Fine-tuned
```python
# Train custom YOLO for tennis balls
# Need labeled dataset with ball positions
yolo = YOLO('yolov8n.pt')
yolo.train(
    data='tennis_ball.yaml',
    epochs=100,
    imgsz=640
)
```

---

### 5. Complete Analysis Pipeline
**Goal:** Full end-to-end tennis match analysis

**Pipeline overview:**
```
Video → Court Detection → Homography → Player Detection → Ball Detection → Analysis
```

**Implementation:** `analyze_match.py`

```python
def analyze_tennis_match(video_path, output_dir):
    """Complete tennis match analysis pipeline."""

    # 1. Load models
    court_model = load_court_model('checkpoints/court_detection/best_model.pth')
    player_model = YOLO('yolov8n.pt')
    ball_model = load_ball_model('checkpoints/ball_detection/best_model.pth')

    # 2. Process video
    cap = cv2.VideoCapture(video_path)
    results = []

    for frame_idx in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect court keypoints
        court_kps = court_model(frame)
        H, success = compute_homography(court_kps, COURT_TEMPLATE)

        if not success:
            continue

        # Detect players
        players = player_model.track(frame, persist=True)
        player_positions = extract_player_positions(players)
        player_positions_court = transform_points(player_positions, H)

        # Detect ball
        ball_heatmap = ball_model(frame)
        ball_pos = extract_ball_position(ball_heatmap)
        ball_pos_court = transform_points([ball_pos], H)[0] if ball_pos else None

        # Store frame results
        results.append({
            'frame': frame_idx,
            'court_keypoints': court_kps.tolist(),
            'homography': H.tolist(),
            'players': [
                {
                    'id': p['id'],
                    'position_image': p['pos_img'],
                    'position_court': p['pos_court'],
                }
                for p in player_positions_court
            ],
            'ball': {
                'position_image': ball_pos,
                'position_court': ball_pos_court,
            } if ball_pos else None,
        })

    # 3. Analyze rally statistics
    rallies = extract_rallies(results)
    stats = compute_match_statistics(rallies)

    # 4. Save results
    save_analysis(results, stats, output_dir)

    # 5. Create visualizations
    create_trajectory_video(video_path, results, output_dir)
    create_heatmaps(results, output_dir)

    return results, stats
```

---

### 6. Match Statistics & Visualization
**Goal:** Extract meaningful insights from tracking data

**Statistics to compute:**
- Player movement heatmaps
- Ball trajectory analysis
- Rally lengths
- Shot types (forehand/backhand detection)
- Court coverage
- Speed and distance traveled
- Serve placement

**Visualizations:**
- Overlay trajectories on court
- Heat maps of player positions
- Ball trajectory paths
- Interactive dashboard (streamlit)

---

## 📋 Recommended Order

### Phase 1: Court & Homography (1-2 days)
1. ✅ Court detection model trained
2. ⏳ Test on real videos
3. ⏳ Implement homography computation
4. ⏳ Verify transformation accuracy

### Phase 2: Player Tracking (2-3 days)
1. ⏳ Integrate YOLOv8 for player detection
2. ⏳ Implement tracking with IDs
3. ⏳ Transform player positions to court coordinates
4. ⏳ Test on full match videos

### Phase 3: Ball Detection (3-5 days)
1. ⏳ Collect/find ball trajectory dataset
2. ⏳ Train TrackNet for ball detection
3. ⏳ Implement ball tracking
4. ⏳ Handle occlusions and fast motion

### Phase 4: Full Pipeline (2-3 days)
1. ⏳ Integrate all components
2. ⏳ Implement `analyze_match.py`
3. ⏳ Add error handling
4. ⏳ Optimize performance

### Phase 5: Analysis & Visualization (3-5 days)
1. ⏳ Compute match statistics
2. ⏳ Create visualizations
3. ⏳ Build interactive dashboard
4. ⏳ Generate reports

---

## 🛠️ Immediate Next Action

**Start with homography implementation:**

```bash
# 1. Create homography module
touch app/src/homography.py

# 2. Test court detection on video
python process_video.py --input test_video.mp4 --visualize

# 3. Implement homography computation
# Edit app/src/homography.py with the code above

# 4. Create test script
python test_homography.py
```

Would you like me to:
1. **Implement homography computation** (app/src/homography.py)
2. **Update process_video.py** to use homography
3. **Create test script** for homography validation
4. **Something else?**

Let me know which part you want to tackle first!
