"""
Tennis Computer Vision - FastAPI Backend
File: app/api/main.py
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from pathlib import Path
import shutil
from datetime import datetime

from app.core.video_processor import VideoProcessor, StreamProcessor
from app.core.data_models import ProcessingConfig, VideoAnalytics


# ==================== Configuration ====================

app = FastAPI(
    title="Tennis Computer Vision API",
    description="AI-powered tennis match analysis and statistics",
    version="0.1.0"
)

# Global state (In production, use Redis/Database)
processing_jobs: Dict[str, Dict[str, Any]] = {}

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


class ProcessingConfigRequest(BaseModel):
    court_detection_interval: int = 30
    batch_size: int = 16
    motion_threshold: float = 5.0
    save_visualization: bool = True
    draw_court_lines: bool = True
    draw_trajectories: bool = True


# ==================== Background Task ====================

def process_video_task(job_id: str, video_path: str, config: ProcessingConfig):
    """Background task to process video"""
    try:
        # Update status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["progress"] = 0.1
        processing_jobs[job_id]["message"] = "Initializing pipeline..."
        
        # Create processor
        processor = VideoProcessor(config)
        
        # Process video
        processing_jobs[job_id]["progress"] = 0.3
        processing_jobs[job_id]["message"] = "Analyzing video..."
        
        analytics: VideoAnalytics = processor(video_path)
        
        # Generate output path
        video_name = Path(video_path).stem
        output_path = RESULTS_DIR / f"{video_name}_processed_{job_id[:8]}.mp4"
        
        processing_jobs[job_id]["progress"] = 0.9
        processing_jobs[job_id]["message"] = "Finalizing..."
        
        # Update job with results
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["progress"] = 1.0
        processing_jobs[job_id]["message"] = "Processing completed successfully"
        processing_jobs[job_id]["result_path"] = str(output_path)
        processing_jobs[job_id]["analytics_summary"] = {
            "total_frames": analytics.total_frames,
            "fps": analytics.fps,
            "duration_seconds": analytics.duration_seconds,
            "total_segments": len(analytics.game_segments),
            "rally_segments": len(analytics.get_rally_segments()),
            "total_rallies": analytics.total_rallies,
        }
        
    except Exception as e:
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        processing_jobs[job_id]["message"] = f"Processing failed: {str(e)}"


# ==================== API Routes ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Tennis Computer Vision API",
        "status": "running",
        "version": "0.1.0"
    }


@app.post("/api/v1/upload", response_model=JobResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[ProcessingConfigRequest] = None
):
    """
    Upload tennis video for processing
    
    - **file**: MP4 video file
    - **config**: Optional processing configuration
    
    Returns job_id for status tracking
    """
    # Validate file
    if not file.filename.endswith(('.mp4', '.MP4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Only MP4, AVI, MOV files are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create processing config
    if config:
        processing_config = ProcessingConfig(
            court_detection_interval=config.court_detection_interval,
            batch_size=config.batch_size,
            motion_threshold=config.motion_threshold,
            save_visualization=config.save_visualization,
            draw_court_lines=config.draw_court_lines,
            draw_trajectories=config.draw_trajectories,
            results_path=str(RESULTS_DIR)
        )
    else:
        processing_config = ProcessingConfig(results_path=str(RESULTS_DIR))
    
    # Initialize job
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,