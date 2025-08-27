import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from celery.result import AsyncResult
from .tasks import run_analysis_task

app = FastAPI(title="Culvert Analysis API")

OUTPUTS_BASE_DIR = Path('/home/appuser/app/outputs')

class AnalysisRequest(BaseModel):
    county: str
    state: str

class JobResponse(BaseModel):
    job_id: str
    status_url: str

@app.post("/analysis", response_model=JobResponse, status_code=202)
async def start_analysis(req: AnalysisRequest, http_request: Request):
    """
    Starts a new culvert analysis job in the background.
    """
    task = run_analysis_task.delay(req.county, req.state)
    status_url = http_request.url_for('get_status', job_id=task.id)
    return {"job_id": task.id, "status_url": status_url}

@app.get("/analysis/{job_id}", name="get_status")
async def get_status(job_id: str):
    """
    Retrieves the status and results of an analysis job.
    """
    task_result = AsyncResult(job_id)

    if task_result.state == 'PENDING':
        response = {'state': task_result.state, 'status': 'Pending...'}
    elif task_result.state == 'PROGRESS':
        response = {'state': task_result.state, 'status': task_result.info.get('status', '')}
    elif task_result.state == 'SUCCESS':
        response = {'state': task_result.state, 'result': task_result.result}
    else: # FAILURE
        response = {
            'state': task_result.state,
            'status': str(task_result.info), # This is the exception info
        }
    return JSONResponse(content=response)


@app.get("/analysis/{job_id}/map")
async def get_map(job_id: str):
    """
    Serves the interactive map HTML file for a completed job.
    """
    map_path = OUTPUTS_BASE_DIR / job_id / "culvert_analysis_map.html"
    if not map_path.is_file():
        raise HTTPException(status_code=404, detail="Map not found or job is not complete.")
    return FileResponse(map_path)

@app.get("/analysis/{job_id}/data/{filename}")
async def get_data_file(job_id: str, filename: str):
    """
    Serves data artifact files (e.g., culverts.geojson) for a completed job.
    """
    # Basic path traversal protection
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = OUTPUTS_BASE_DIR / job_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found or job is not complete.")
    return FileResponse(file_path)
