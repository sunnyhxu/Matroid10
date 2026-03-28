"""FastAPI application for Matroid10 pipeline web UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .state_manager import StateManager
from .dedup_store import DedupStore
from .pipeline_runner import PipelineRunner

try:
    from ..search_progress import refresh_progress
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from search_progress import refresh_progress

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIG_PATH = REPO_ROOT / "configs" / "default.toml"
STATIC_DIR = Path(__file__).parent / "static"

# State files
STATE_PATH = ARTIFACTS_DIR / "run_state.json"
RESULTS_PATH = ARTIFACTS_DIR / "accumulated_results.jsonl"
SEEN_IDS_PATH = ARTIFACTS_DIR / "seen_ids.json"

# Initialize components
state_manager = StateManager(STATE_PATH)
dedup_store = DedupStore(RESULTS_PATH, SEEN_IDS_PATH)
pipeline_runner: Optional[PipelineRunner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Matroid10 Pipeline Web UI started")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print(f"  Config: {CONFIG_PATH}")
    yield
    # Shutdown
    print("Matroid10 Pipeline Web UI shutting down")


# FastAPI app
app = FastAPI(
    title="Matroid10 Pipeline",
    description="Web UI for controlling the Matroid10 pipeline",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pipeline_runner() -> PipelineRunner:
    """Get or create the pipeline runner."""
    global pipeline_runner
    if pipeline_runner is None:
        pipeline_runner = PipelineRunner(
            state_manager=state_manager,
            dedup_store=dedup_store,
            config_path=CONFIG_PATH,
            repo_root=REPO_ROOT,
        )
    return pipeline_runner


class StartRequest(BaseModel):
    trial_index_start: Optional[int] = None
    mode: Optional[str] = None  # "representable" or "sparse_paving"


class StatusResponse(BaseModel):
    run_id: Optional[str]
    status: str
    stop_requested: bool
    current_phase: Optional[str]
    next_trial_index_start: int
    phase_status: Dict[str, Any]
    counters: Dict[str, int]
    accumulated_unique_count: int
    last_updated: str


@app.get("/")
async def serve_index():
    """Serve the main UI page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/status")
async def get_status() -> StatusResponse:
    """Get current pipeline status."""
    state = state_manager.load()
    dedup_stats = dedup_store.get_stats()

    return StatusResponse(
        run_id=state.get("run_id"),
        status=state.get("status", "idle"),
        stop_requested=state.get("stop_requested", False),
        current_phase=state.get("current_phase"),
        next_trial_index_start=state.get("next_trial_index_start", 0),
        phase_status=state.get("phase_status", {}),
        counters=state.get("counters", {}),
        accumulated_unique_count=dedup_stats.get("unique_count", 0),
        last_updated=state.get("last_updated", ""),
    )


@app.post("/api/start")
async def start_pipeline(request: StartRequest = StartRequest()):
    """Start the pipeline in background."""
    runner = get_pipeline_runner()

    if runner.is_running():
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    state = state_manager.load()
    if state.get("status") == "running":
        raise HTTPException(status_code=409, detail="Pipeline is already running")

    # Use provided trial_index_start or default from state
    trial_start = request.trial_index_start
    mode = request.mode

    success = runner.start(trial_index_start=trial_start, mode=mode)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start pipeline")

    return JSONResponse(
        content={"message": "Pipeline started", "trial_index_start": trial_start},
        status_code=202,
    )


@app.post("/api/stop")
async def stop_pipeline():
    """Request graceful stop of the pipeline."""
    state = state_manager.load()

    if state.get("status") != "running":
        raise HTTPException(status_code=400, detail="Pipeline is not running")

    state_manager.request_stop()

    return JSONResponse(
        content={"message": "Stop requested. Pipeline will stop after current subprocess completes."},
        status_code=202,
    )


@app.get("/api/config")
async def get_config():
    """Get current pipeline configuration."""
    if not CONFIG_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")

    try:
        import tomllib
        with CONFIG_PATH.open("rb") as f:
            config = tomllib.load(f)
        return JSONResponse(content=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load config: {e}")


@app.get("/api/accumulated")
async def get_accumulated_stats():
    """Get accumulated results statistics."""
    stats = dedup_store.get_stats()
    return JSONResponse(content=stats)


@app.post("/api/search-progress")
async def get_search_progress():
    """Refresh search progress, update README, and return results."""
    try:
        progress = refresh_progress(update_readme=True)
        return JSONResponse(content={"progress": progress})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files (must be after API routes)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
