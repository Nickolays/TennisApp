# Configuration Files

This directory contains YAML configuration files for the Tennis Analytics pipeline.

## Available Configurations

### 1. `default.yaml` - Balanced Configuration
- **Use case**: General purpose, good balance between speed and accuracy
- **Features**: All pipelines enabled, standard settings
- **GPU batch size**: 16
- **Chunk size**: 500 frames
- **Best for**: Development, testing, moderate-length videos

```bash
python -m app.main process video.mp4 --config configs/default.yaml
```

### 2. `fast.yaml` - Speed-Optimized
- **Use case**: Quick processing, lower quality acceptable
- **Features**:
  - Skips every 2nd frame
  - Lower model resolutions (320x180)
  - Disabled smoothing and ML-based hit detection
  - No minimap or trajectory rendering
- **GPU batch size**: 32 (larger batches)
- **Chunk size**: 1000 frames
- **Speed improvement**: ~3x faster than default
- **Best for**: Rapid prototyping, long videos (>30 min), real-time preview

```bash
python -m app.main process video.mp4 --config configs/fast.yaml
```

### 3. `production.yaml` - Production Deployment
- **Use case**: Production server with database and API
- **Features**:
  - Database export enabled (PostgreSQL)
  - API server configuration
  - Monitoring (Prometheus, Sentry)
  - Log rotation
  - Higher CPU worker count (8 workers)
  - H.264 video encoding (better compression)
- **Best for**: Deployed servers, batch processing, multi-user systems

```bash
# Set environment variables first
export DATABASE_URL="postgresql://user:pass@localhost:5432/tennis_db"
export SENTRY_DSN="https://..."

python -m app.api.main --config configs/production.yaml
```

---

## Configuration Structure

Each YAML file follows this hierarchical structure:

```yaml
video:              # Video processing settings
execution:          # Parallel execution settings
preprocessing:      # Frame extraction & filtering
detection:          # Neural network detection (court, ball, player)
temporal:           # Gap filling, smoothing, windowing
geometry:           # Homography & coordinate transformation
events:             # Event detection (hit, bounce, in/out, speed)
analytics:          # Game segmentation, statistics, database
rendering:          # Video overlay, minimap, output
model_registry:     # Model class mapping (for easy swapping)
logging:            # Logging configuration
```

---

## Customization Guide

### How to Swap Models

Change any model without code changes:

```yaml
# Use TrackNet for ball detection
detection:
  ball:
    model_type: "tracknet"
    model_path: "models/ball_model_best.pt"

# Swap to your custom model
detection:
  ball:
    model_type: "custom_cnn"  # Must be registered in model_registry
    model_path: "models/my_custom_ball_detector.onnx"
```

### How to Disable Pipeline Steps

Set `enabled: false` for any step:

```yaml
events:
  ball_hit:
    enabled: false  # Skip ML-based hit detection

rendering:
  minimap:
    enabled: false  # Don't render minimap
```

### How to Tune Performance

**For speed** (sacrifice quality):
- Increase `frame_skip_interval`
- Lower `input_size` for models
- Increase detection `interval`
- Disable `smoothing`
- Increase `chunk_size` and `gpu_batch_size`

**For quality** (sacrifice speed):
- Set `frame_skip_interval: 1` (process all frames)
- Higher `input_size` for models
- Lower detection `interval` (more frequent)
- Enable `smoothing` with Kalman filter
- Lower `chunk_size` for better memory management

### How to Create Custom Config

1. Copy `default.yaml`:
   ```bash
   cp configs/default.yaml configs/my_config.yaml
   ```

2. Edit values:
   ```yaml
   # Example: High accuracy config
   detection:
     court:
       interval: 15  # More frequent
       input_size: [1280, 720]  # Higher res
     ball:
       input_size: [1280, 720]

   temporal:
     smoothing:
       enabled: true
       method: "kalman"

   events:
     ball_hit:
       confidence_threshold: 0.8  # Stricter
   ```

3. Use it:
   ```bash
   python -m app.main process video.mp4 --config configs/my_config.yaml
   ```

---

## Environment Variables

Production config uses environment variables for sensitive data:

```bash
# Database
export DATABASE_URL="postgresql://user:password@host:5432/db"

# Monitoring
export SENTRY_DSN="https://your-sentry-dsn"

# Redis (for API job queue)
export REDIS_URL="redis://localhost:6379/0"

# Storage
export STORAGE_PATH="/mnt/storage/tennis_results"
```

Load them before running:

```bash
# Option 1: .env file
cat > .env << EOF
DATABASE_URL=postgresql://user:pass@localhost:5432/tennis_db
SENTRY_DSN=https://...
EOF

# Load and run
export $(cat .env | xargs)
python -m app.api.main --config configs/production.yaml

# Option 2: docker-compose.yml (recommended)
# See docker-compose.yml for environment variable setup
```

---

## Configuration Priority

Configs are loaded in this order (later overrides earlier):

1. Built-in defaults (hardcoded in code)
2. `configs/default.yaml`
3. Specified config file (e.g., `--config configs/fast.yaml`)
4. Environment variables (for production.yaml)
5. Command-line arguments (highest priority)

Example:
```bash
# This overrides chunk_size from fast.yaml
python -m app.main process video.mp4 \
  --config configs/fast.yaml \
  --chunk-size 2000
```

---

## Validation

Configs are validated on load using Pydantic models. Invalid configs will raise clear errors:

```python
# In app/core/config.py
class VideoConfig(BaseModel):
    chunk_size: int = Field(gt=0, le=5000)  # Must be 1-5000
    output_dir: str
    save_annotated_video: bool = True

class PipelineConfig(BaseModel):
    video: VideoConfig
    execution: ExecutionConfig
    # ...
```

Validation errors show exactly what's wrong:

```
ConfigValidationError: Invalid configuration:
  - video.chunk_size: Value must be greater than 0 (got: -100)
  - detection.court.interval: Value must be less than or equal to 1000 (got: 5000)
```

---

## Best Practices

1. **Start with default.yaml** for development
2. **Use fast.yaml** for quick iterations
3. **Create custom configs** for specific use cases (e.g., clay court, indoor, etc.)
4. **Never hardcode paths** - use relative paths or environment variables
5. **Version control configs** - track changes to settings
6. **Document custom configs** - add comments explaining non-obvious choices

---

## Example Workflows

### Development Workflow
```bash
# Quick test on short video
python -m app.main process short.mp4 --config configs/fast.yaml

# Full quality test
python -m app.main process short.mp4 --config configs/default.yaml

# Debug specific pipeline
python -m app.main process short.mp4 --config configs/custom_debug.yaml
```

### Production Workflow
```bash
# Set up environment
export DATABASE_URL="..."
export REDIS_URL="..."

# Start API server
python -m app.api.main --config configs/production.yaml

# Process video via API
curl -X POST http://localhost:8000/api/v1/upload \
  -F "video=@match.mp4" \
  -F "config=production"
```

---

## Troubleshooting

### Config not found
```
FileNotFoundError: Config file not found: configs/xyz.yaml
```
**Fix**: Check file path and spelling

### Invalid YAML syntax
```
yaml.scanner.ScannerError: mapping values are not allowed here
```
**Fix**: Check indentation (use spaces, not tabs)

### Model path error
```
FileNotFoundError: Model not found: models/missing_model.pt
```
**Fix**: Download model or fix path in config

### Database connection error
```
OperationalError: could not connect to server
```
**Fix**: Check `DATABASE_URL` and ensure DB is running

---

## Future Additions

Planned configurations:
- `accurate.yaml` - Maximum quality (no compromises)
- `real_time.yaml` - Live stream processing
- `clay_court.yaml` - Optimized for clay courts
- `indoor.yaml` - Optimized for indoor courts
- `multi_camera.yaml` - Multi-camera fusion settings
