import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.data_models import ProcessingConfig
from app.models.detection_models import DetectionPipeline, ModelType


app = FastAPI(title="Tennis CV Backend")

# ============================================================
# Load models once on server start
# ============================================================

print("⚙️ Initializing Detection Pipeline...")

config = ProcessingConfig(
    batch_size=8,
    court_detection_interval=30,
    motion_threshold=5.0
)

pipeline = DetectionPipeline(config)

pipeline.add_model(ModelType.COURT_DETECTION, "models/model_tennis_court_det.pt")
pipeline.add_model(ModelType.BALL_DETECTION, "models/ball_model_best.pt")

print("✅ Models loaded")


# ============================================================
# Helper — read uploaded video into frames
# ============================================================

def read_video_bytes(video_bytes: bytes, max_frames: int = None):
    """Decode video file bytes → list of frames"""
    video_array = np.frombuffer(video_bytes, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(video_array, cv2.IMREAD_COLOR))

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return np.array(frames)


# ============================================================
# FastAPI Endpoint
# ============================================================

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Upload a video → Run both detection models → Return JSON
    """

    # Read file bytes
    video_bytes = await file.read()

    # Decode into numpy frames
    frames = read_video_bytes(video_bytes, max_frames=200)  # limit for speed

    if len(frames) == 0:
        return JSONResponse({"error": "Could not decode video"}, status_code=400)

    # Run detection
    results = pipeline.detect(frames)

    # Build JSON response
    output = {
        "court_detection": [],
        "ball_detection": []
    }

    # COURT DETECTION
    if ModelType.COURT_DETECTION in results:
        for det in results[ModelType.COURT_DETECTION]:
            output["court_detection"].append({
                "frame_id": det.frame_id,
                "confidence": float(det.confidence),
                "coords": det.coords.tolist() if hasattr(det, "coords") else None
            })

    # BALL DETECTION
    if ModelType.BALL_DETECTION in results:
        for det in results[ModelType.BALL_DETECTION]:
            output["ball_detection"].append({
                "frame_id": det.frame_id,
                "confidence": float(det.confidence),
                "x": float(det.x),
                "y": float(det.y)
            })

    return output


# ============================================================
# Run backend
# ============================================================

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

# python -m app.main