"""
Tennis Computer Vision - FastAPI Backend
File: app/api/main.py

Features:
- Video file upload processing
- RTSP stream processing
- Parallel batching for fast inference
- Job tracking with progress
- WebSocket support for real-time updates
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
import uuid
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
import asyncio
from datetime import datetime

# Import process_video pipeline (uses modern architecture)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from process_video import process_video_simple, save_results
from app.core.context import ProcessingContext


# ==================== Configuration ====================

app = FastAPI(
    title="Tennis Computer Vision API",
    description="AI-powered tennis match analysis with video/RTSP stream support",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (In production, use Redis/Database)
processing_jobs: Dict[str, Dict[str, Any]] = {}
stream_sessions: Dict[str, Dict[str, Any]] = {}

# Directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ==================== Pydantic Models ====================

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    result_path: Optional[str] = None
    analytics_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StreamRequest(BaseModel):
    stream_url: str  # RTSP URL or video file path
    max_frames: Optional[int] = None
    batch_size: int = 16


class StreamSessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


# ==================== Background Tasks ====================

def process_video_task(job_id: str, video_path: str, max_frames: Optional[int] = None):
    """Background task to process video file"""
    try:
        # Update status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Initializing pipeline..."

        # Process video using our proven pipeline
        processing_jobs[job_id]["progress"] = 0.3
        processing_jobs[job_id]["message"] = "Running detection pipeline..."

        context = process_video_simple(video_path, max_frames)

        if context is None:
            raise Exception("Processing failed - context is None")

        # Save results
        processing_jobs[job_id]["progress"] = 0.8
        processing_jobs[job_id]["message"] = "Saving results..."

        json_path, video_path_out = save_results(context, RESULTS_DIR)

        # Update job with results
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "Processing completed successfully"
        processing_jobs[job_id]["result_json"] = str(json_path)
        processing_jobs[job_id]["result_video"] = str(video_path_out)

        # Analytics summary
        court_count = sum(1 for d in context.detections if d.court_keypoints is not None)
        ball_count = sum(1 for d in context.detections if d.ball_position_px is not None)
        player_count = sum(1 for d in context.detections if d.has_players())

        processing_jobs[job_id]["analytics_summary"] = {
            "total_frames": len(context.detections),
            "fps": context.fps,
            "duration_seconds": context.duration_seconds,
            "detections": {
                "court": court_count,
                "ball": ball_count,
                "players": player_count
            },
            "events": {
                "bounces": len(context.bounce_events),
                "hits": len(context.hit_events)
            }
        }

    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"
        processing_jobs[job_id]["progress"] = 0.0


async def process_stream_task(session_id: str, stream_url: str, max_frames: Optional[int], batch_size: int):
    """Background task to process RTSP stream or video file as stream"""
    try:
        stream_sessions[session_id]["status"] = "processing"
        stream_sessions[session_id]["message"] = "Connecting to stream..."

        # Open stream (supports RTSP, HTTP, file)
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            raise Exception(f"Failed to open stream: {stream_url}")

        stream_sessions[session_id]["message"] = "Stream connected, processing..."

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        output_path = RESULTS_DIR / f"stream_{session_id[:8]}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Process frames
        frame_count = 0
        frames_buffer = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_buffer.append(frame)
            frame_count += 1

            # Process in batches for efficiency
            if len(frames_buffer) >= batch_size:
                # Here you would run detection on the batch
                # For now, just write frames
                for f in frames_buffer:
                    writer.write(f)

                frames_buffer = []
                stream_sessions[session_id]["frames_processed"] = frame_count

            # Stop if max_frames reached
            if max_frames and frame_count >= max_frames:
                break

            # Update progress
            if max_frames:
                progress = frame_count / max_frames
                stream_sessions[session_id]["progress"] = progress

        # Process remaining frames
        for f in frames_buffer:
            writer.write(f)

        cap.release()
        writer.release()

        # Update session
        stream_sessions[session_id]["status"] = "completed"
        stream_sessions[session_id]["message"] = "Stream processing completed"
        stream_sessions[session_id]["frames_processed"] = frame_count
        stream_sessions[session_id]["result_path"] = str(output_path)

    except Exception as e:
        stream_sessions[session_id]["status"] = "failed"
        stream_sessions[session_id]["error"] = str(e)
        stream_sessions[session_id]["message"] = f"Stream processing failed: {str(e)}"


# ==================== API Routes ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Tennis Computer Vision API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Video file processing",
            "RTSP stream support",
            "Parallel batch inference",
            "Real-time detection (court, ball, players)",
            "Event detection (bounces, hits)"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    import torch

    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_jobs": len([j for j in processing_jobs.values() if j["status"] == "processing"]),
        "active_streams": len([s for s in stream_sessions.values() if s["status"] == "processing"])
    }


@app.post("/api/v1/upload", response_model=JobResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_frames: Optional[int] = None
):
    """
    Upload tennis video file for processing

    - **file**: MP4/AVI/MOV video file
    - **max_frames**: Optional limit on frames to process

    Returns job_id for status tracking
    """
    # Validate file
    if not file.filename.endswith(('.mp4', '.MP4', '.avi', '.mov', '.MOV')):
        raise HTTPException(status_code=400, detail="Only MP4, AVI, MOV files are supported")

    # Generate job ID
    job_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initialize job
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued",
        "created_at": created_at,
        "video_path": str(upload_path)
    }

    # Start background processing
    background_tasks.add_task(process_video_task, job_id, str(upload_path), max_frames)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Video uploaded successfully, processing started",
        created_at=created_at
    )


@app.post("/api/v1/stream/start", response_model=StreamSessionResponse)
async def start_stream_processing(
    background_tasks: BackgroundTasks,
    request: StreamRequest
):
    """
    Start processing RTSP stream or video file as stream

    - **stream_url**: RTSP URL (rtsp://...) or video file path
    - **max_frames**: Optional limit on frames to process
    - **batch_size**: Batch size for parallel inference (default: 16)

    Returns session_id for tracking
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    # Validate URL/path
    if not (request.stream_url.startswith('rtsp://') or
            request.stream_url.startswith('http://') or
            Path(request.stream_url).exists()):
        raise HTTPException(
            status_code=400,
            detail="Invalid stream URL. Must be RTSP URL, HTTP URL, or valid file path"
        )

    # Initialize session
    stream_sessions[session_id] = {
        "session_id": session_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Stream session created",
        "created_at": created_at,
        "stream_url": request.stream_url,
        "frames_processed": 0
    }

    # Start background processing
    background_tasks.add_task(
        process_stream_task,
        session_id,
        request.stream_url,
        request.max_frames,
        request.batch_size
    )

    return StreamSessionResponse(
        session_id=session_id,
        status="pending",
        message="Stream processing started"
    )


@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of video processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message", ""),
        result_path=job.get("result_video"),
        analytics_summary=job.get("analytics_summary"),
        error=job.get("error")
    )


@app.get("/api/v1/stream/status/{session_id}")
async def get_stream_status(session_id: str):
    """Get status of stream processing session"""
    if session_id not in stream_sessions:
        raise HTTPException(status_code=404, detail="Stream session not found")

    return stream_sessions[session_id]


@app.get("/api/v1/results/{job_id}/json")
async def download_json_results(job_id: str):
    """Download JSON results for completed job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    json_path = job.get("result_json")
    if not json_path or not Path(json_path).exists():
        raise HTTPException(status_code=404, detail="Results not found")

    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"results_{job_id[:8]}.json"
    )


@app.get("/api/v1/results/{job_id}/video")
async def download_video_results(job_id: str):
    """Download visualization video for completed job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    video_path = job.get("result_video")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"processed_{job_id[:8]}.mp4"
    )


@app.get("/api/v1/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List all processing jobs"""
    jobs = list(processing_jobs.values())

    # Filter by status if provided
    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Sort by created_at (most recent first)
    jobs = sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)

    # Limit results
    jobs = jobs[:limit]

    return {
        "total": len(jobs),
        "jobs": jobs
    }


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    # Delete uploaded file
    video_path = job.get("video_path")
    if video_path and Path(video_path).exists():
        Path(video_path).unlink()

    # Delete result files
    result_json = job.get("result_json")
    if result_json and Path(result_json).exists():
        Path(result_json).unlink()

    result_video = job.get("result_video")
    if result_video and Path(result_video).exists():
        Path(result_video).unlink()

    # Remove from jobs dict
    del processing_jobs[job_id]

    return {"message": f"Job {job_id} deleted successfully"}


# ==================== Worker Endpoints (GPU Worker Polling) ====================

@app.get("/worker/get_job")
async def worker_get_job():
    """
    Worker polls this endpoint to get pending jobs

    Returns:
        Job data if available, empty dict if no jobs

    Design: Worker initiates connection (no white IP needed)
    """
    # Find first pending job
    for job_id, job in processing_jobs.items():
        if job["status"] == "pending":
            # Mark as claimed by worker
            job["status"] = "claimed"
            job["claimed_at"] = datetime.utcnow().isoformat()

            # Return job info to worker
            return {
                "job_id": job_id,
                "video_url": f"{RESULTS_DIR.parent}/{job['video_path']}",  # Local path for now
                "max_frames": job.get("max_frames"),
                "config": job.get("config", {})
            }

    # No jobs available
    return {}


@app.post("/worker/complete/{job_id}")
async def worker_complete_job(
    job_id: str,
    video: UploadFile = File(...),
    json_data: str = None
):
    """
    Worker submits completed job results

    Args:
        job_id: Job ID
        video: Processed video file
        json_data: JSON results as string
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]

    try:
        # Save video file
        video_path = RESULTS_DIR / f"{Path(job['video_path']).stem}_processed.mp4"
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Save JSON results
        if json_data:
            json_path = RESULTS_DIR / f"{Path(job['video_path']).stem}_results.json"
            with open(json_path, "w") as f:
                f.write(json_data)

            job["result_json"] = str(json_path)

        # Update job status
        job["status"] = "completed"
        job["result_video"] = str(video_path)
        job["completed_at"] = datetime.utcnow().isoformat()
        job["progress"] = 1.0
        job["message"] = "Processing completed by worker"

        # Parse analytics from JSON
        if json_data:
            analytics = json.loads(json_data)
            job["analytics_summary"] = analytics.get("analytics_summary", {})

        return {"message": "Job completed successfully", "job_id": job_id}

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")


@app.post("/worker/fail/{job_id}")
async def worker_fail_job(job_id: str, error: dict):
    """
    Worker reports job failure

    Args:
        job_id: Job ID
        error: Error details {"error": "error message"}
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = processing_jobs[job_id]
    job["status"] = "failed"
    job["error"] = error.get("error", "Unknown error")
    job["failed_at"] = datetime.utcnow().isoformat()
    job["message"] = f"Worker failed: {job['error']}"

    return {"message": "Job failure recorded"}


# ==================== WebSocket for Real-time Updates ====================

@app.websocket("/ws/job/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates"""
    await websocket.accept()

    try:
        while True:
            if job_id not in processing_jobs:
                await websocket.send_json({"error": "Job not found"})
                break

            job = processing_jobs[job_id]
            await websocket.send_json({
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0.0),
                "message": job.get("message", "")
            })

            # Stop if job completed or failed
            if job["status"] in ["completed", "failed"]:
                break

            await asyncio.sleep(1)  # Update every second

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
