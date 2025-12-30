#!/usr/bin/env python3
"""
GPU Worker - Distributed Tennis Video Processing
File: worker/gpu_worker.py

This worker:
1. Polls cloud server for pending jobs (no white IP needed!)
2. Downloads video from cloud
3. Runs AI processing with local GPU
4. Uploads results back to cloud
5. Repeats forever

Usage:
    python worker/gpu_worker.py --server https://your-vds.com
    python worker/gpu_worker.py --server http://localhost:8000  # Local testing
"""
import argparse
import time
import sys
import requests
from pathlib import Path
import json
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from process_video import process_video_simple, save_results


class GPUWorker:
    """
    GPU Worker that polls cloud for jobs and processes videos locally

    Design:
    - Worker initiates ALL connections (outgoing) - no white IP needed
    - Works behind NAT/firewall automatically
    - Fault tolerant: retries on failure
    - Graceful shutdown: finishes current job before exit
    """

    def __init__(self, server_url: str, poll_interval: int = 2, max_retries: int = 3):
        """
        Initialize GPU worker

        Args:
            server_url: Cloud server URL (e.g., "https://your-vds.com")
            poll_interval: How often to check for jobs (seconds)
            max_retries: Max retries for failed operations
        """
        self.server_url = server_url.rstrip('/')
        self.poll_interval = poll_interval
        self.max_retries = max_retries

        # Create work directories
        self.work_dir = Path("worker_temp")
        self.work_dir.mkdir(exist_ok=True)

        print(f"🚀 GPU Worker initialized")
        print(f"   Server: {self.server_url}")
        print(f"   Poll interval: {poll_interval}s")
        print(f"   Work directory: {self.work_dir}")
        print()

    def check_server_health(self) -> bool:
        """Check if cloud server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Server healthy: {data.get('status')}")
                print(f"  GPU available: {data.get('gpu_available')}")
                print(f"  Active jobs: {data.get('active_jobs')}")
                return True
            return False
        except Exception as e:
            print(f"✗ Server unreachable: {e}")
            return False

    def get_job(self) -> dict | None:
        """
        Poll server for pending job

        Returns:
            Job dict with {job_id, video_url, config} or None if no jobs
        """
        try:
            response = requests.get(
                f"{self.server_url}/worker/get_job",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("job_id"):
                    return data

            return None

        except requests.exceptions.RequestException as e:
            print(f"✗ Error getting job: {e}")
            return None

    def download_video(self, job_id: str, video_url: str) -> Path | None:
        """
        Download video from cloud storage

        Args:
            job_id: Job ID
            video_url: URL to download video from

        Returns:
            Path to downloaded video or None on failure
        """
        try:
            print(f"📥 Downloading video for job {job_id[:8]}...")

            # Download video
            response = requests.get(video_url, stream=True, timeout=300)
            response.raise_for_status()

            # Save to temp file
            video_path = self.work_dir / f"{job_id}.mp4"

            with open(video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"✓ Video downloaded: {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")
            return video_path

        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None

    def process_video(self, video_path: Path, max_frames: int | None = None):
        """
        Run AI processing on video

        Args:
            video_path: Path to video file
            max_frames: Optional frame limit

        Returns:
            (json_path, video_path) or (None, None) on failure
        """
        try:
            print(f"🤖 Starting AI processing...")
            print(f"   Video: {video_path}")
            print(f"   Max frames: {max_frames or 'All'}")
            print()

            # Run full pipeline (Detection → Temporal → Geometry → Events)
            start_time = time.time()
            context = process_video_simple(str(video_path), max_frames)

            if context is None:
                print("✗ Processing failed - context is None")
                return None, None

            # Save results
            results_dir = self.work_dir / "results"
            results_dir.mkdir(exist_ok=True)

            json_path, video_out_path = save_results(context, results_dir)

            elapsed = time.time() - start_time
            print(f"\n✓ Processing complete in {elapsed:.1f}s")
            print(f"  Results: {json_path}")
            print(f"  Video: {video_out_path}")

            return json_path, video_out_path

        except Exception as e:
            print(f"✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def upload_results(self, job_id: str, json_path: Path, video_path: Path) -> bool:
        """
        Upload processing results back to cloud

        Args:
            job_id: Job ID
            json_path: Path to JSON results
            video_path: Path to processed video

        Returns:
            True on success, False on failure
        """
        try:
            print(f"📤 Uploading results for job {job_id[:8]}...")

            # Read JSON results
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            # Upload files
            with open(video_path, 'rb') as video_file:
                files = {
                    'video': ('processed.mp4', video_file, 'video/mp4'),
                }

                data = {
                    'json_data': json.dumps(json_data)
                }

                response = requests.post(
                    f"{self.server_url}/worker/complete/{job_id}",
                    files=files,
                    data=data,
                    timeout=600  # 10 minutes for upload
                )

                response.raise_for_status()

            print(f"✓ Results uploaded successfully")
            return True

        except Exception as e:
            print(f"✗ Upload failed: {e}")
            return False

    def cleanup_job(self, job_id: str):
        """Clean up temporary files for job"""
        try:
            # Remove video file
            video_path = self.work_dir / f"{job_id}.mp4"
            if video_path.exists():
                video_path.unlink()

            # Remove results
            results_dir = self.work_dir / "results"
            if results_dir.exists():
                shutil.rmtree(results_dir)
                results_dir.mkdir(exist_ok=True)

            print(f"🧹 Cleaned up job {job_id[:8]}")

        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

    def report_failure(self, job_id: str, error: str) -> bool:
        """Report job failure to cloud"""
        try:
            response = requests.post(
                f"{self.server_url}/worker/fail/{job_id}",
                json={"error": error},
                timeout=10
            )
            return response.status_code == 200
        except:
            return False

    def run(self):
        """
        Main worker loop

        Continuously polls server for jobs and processes them
        """
        print("="*60)
        print("GPU WORKER RUNNING")
        print("="*60)
        print()

        # Initial health check
        if not self.check_server_health():
            print("⚠️  Warning: Server not responding, will keep trying...")

        print()
        print("🔍 Polling for jobs...")
        print("   (Press Ctrl+C to stop)")
        print()

        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            while True:
                try:
                    # Poll for job
                    job = self.get_job()

                    if job:
                        job_id = job['job_id']
                        video_url = job['video_url']
                        max_frames = job.get('max_frames')

                        print()
                        print("="*60)
                        print(f"📋 NEW JOB: {job_id[:8]}")
                        print("="*60)
                        print()

                        # Download video
                        video_path = self.download_video(job_id, video_url)

                        if not video_path:
                            self.report_failure(job_id, "Failed to download video")
                            continue

                        # Process video
                        json_path, video_out_path = self.process_video(video_path, max_frames)

                        if not json_path or not video_out_path:
                            self.report_failure(job_id, "Processing failed")
                            self.cleanup_job(job_id)
                            continue

                        # Upload results
                        success = self.upload_results(job_id, json_path, video_out_path)

                        if not success:
                            self.report_failure(job_id, "Failed to upload results")

                        # Cleanup
                        self.cleanup_job(job_id)

                        print()
                        print("="*60)
                        print(f"✓ JOB COMPLETE: {job_id[:8]}")
                        print("="*60)
                        print()

                        consecutive_failures = 0  # Reset on success

                    else:
                        # No jobs available
                        print(".", end="", flush=True)
                        time.sleep(self.poll_interval)

                except KeyboardInterrupt:
                    raise  # Re-raise to exit gracefully

                except Exception as e:
                    consecutive_failures += 1
                    print(f"\n✗ Error: {e}")

                    if consecutive_failures >= max_consecutive_failures:
                        print(f"\n💀 Too many consecutive failures ({consecutive_failures})")
                        print("   Exiting...")
                        break

                    print(f"   Retrying in {self.poll_interval * 2}s...")
                    time.sleep(self.poll_interval * 2)

        except KeyboardInterrupt:
            print("\n\n🛑 Worker stopped by user")
            print("   Graceful shutdown...")

        finally:
            print("\n👋 GPU Worker shutting down")
            print(f"   Processed jobs successfully")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="GPU Worker for Tennis Video Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect to cloud server
  python worker/gpu_worker.py --server https://your-vds.com

  # Local testing
  python worker/gpu_worker.py --server http://localhost:8000

  # Custom poll interval
  python worker/gpu_worker.py --server https://your-vds.com --poll-interval 5
        """
    )

    parser.add_argument(
        '--server',
        required=True,
        help='Cloud server URL (e.g., https://your-vds.com)'
    )

    parser.add_argument(
        '--poll-interval',
        type=int,
        default=2,
        help='How often to check for jobs (seconds, default: 2)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Max retries for failed operations (default: 3)'
    )

    args = parser.parse_args()

    # Create and run worker
    worker = GPUWorker(
        server_url=args.server,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries
    )

    worker.run()


if __name__ == "__main__":
    main()
