#!/usr/bin/env python3
"""
FastAPI endpoints for compiler optimization tools (FOGA, HBRF, XGBoost, and Compare)
Provides RESTful API for running optimizations and retrieving results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
import subprocess
import os
import uuid
import json
import time
import shutil
from datetime import datetime
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Compiler Optimization API",
    description="API for FOGA, HBRF, XGBoost optimizers and comparison tools",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory job storage (use Redis or database in production)
jobs: Dict[str, Dict] = {}


# Pydantic models
class OptimizationJob(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    optimizer: str
    source_file: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict] = None
    output: Optional[str] = None
    error: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    optimizer: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: Optional[str] = None


# Helper functions
def create_job(optimizer: str, source_file: str) -> str:
    """Create a new optimization job"""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "optimizer": optimizer,
        "source_file": source_file,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None
    }
    return job_id


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status and additional fields"""
    if job_id in jobs:
        jobs[job_id]["status"] = status
        if status == "running" and "started_at" not in jobs[job_id]:
            jobs[job_id]["started_at"] = datetime.now().isoformat()
        elif status in ["completed", "failed"]:
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        for key, value in kwargs.items():
            jobs[job_id][key] = value


def run_optimizer_task(job_id: str, optimizer: str, source_path: str, test_input_path: Optional[str] = None):
    """Background task to run optimizer"""
    try:
        update_job_status(job_id, "running")
        
        # Build command
        cmd = ["python3", "-u", f"{optimizer}.py", source_path]
        if test_input_path:
            cmd.append(test_input_path)
        
        # Prepare output file
        output_file = RESULTS_DIR / f"{job_id}_output.txt"
        
        # Run optimizer
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        output_lines = []
        
        # Open file for writing in real-time
        with open(output_file, 'w') as f:
            for line in process.stdout:
                line_content = line.rstrip()
                output_lines.append(line_content)
                f.write(line)
                f.flush()  # Ensure it's written immediately
        
        process.wait(timeout=7200)  # 2 hour timeout
        output = "\n".join(output_lines)
        
        # Parse results based on optimizer
        result = parse_optimizer_output(optimizer, output)
        
        # Save detailed results
        result_file = RESULTS_DIR / f"{job_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump({
                "job_id": job_id,
                "optimizer": optimizer,
                "output": output,
                "result": result
            }, f, indent=2)
        
        update_job_status(job_id, "completed", result=result, output=output)
        
    except subprocess.TimeoutExpired:
        update_job_status(
            job_id, 
            "failed", 
            error="Optimization timed out after 2 hours"
        )
    except Exception as e:
        update_job_status(
            job_id, 
            "failed", 
            error=str(e)
        )


def parse_optimizer_output(optimizer: str, output: str) -> Dict:
    """Parse optimizer output to extract key results"""
    result = {
        "best_time": None,
        "total_time": None,
        "evaluations": None,
        "enabled_flags": []
    }
    
    try:
        if optimizer in ["foga", "hbrf_optimizer", "xgboost_optimizer"]:
            # Try to load JSON results file
            json_files = {
                "hbrf_optimizer": "hbrf_results.json",
                "xgboost_optimizer": "xgboost_results.json"
            }
            
            if optimizer in json_files and os.path.exists(json_files[optimizer]):
                with open(json_files[optimizer], 'r') as f:
                    data = json.load(f)
                    result["best_time"] = data.get("best_time")
                    result["total_time"] = data.get("total_optimization_time") or data.get("total_time")
                    result["evaluations"] = data.get("total_evaluations")
                    result["enabled_flags"] = data.get("enabled_flags", [])
            
            # Parse from output as fallback
            for line in output.split('\n'):
                if 'Best Execution Time:' in line:
                    try:
                        result["best_time"] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif 'Total Optimization Time:' in line or 'Total time:' in line:
                    try:
                        result["total_time"] = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif 'Total Evaluations:' in line or 'Evaluations:' in line:
                    try:
                        result["evaluations"] = int(line.split(':')[1].strip())
                    except:
                        pass
        
        elif optimizer == "compare_optimizers":
            # Load comparison results
            if os.path.exists("comparison_results.json"):
                with open("comparison_results.json", 'r') as f:
                    result = json.load(f)
    
    except Exception as e:
        result["parse_error"] = str(e)
    
    return result


# API Endpoints

@app.get("/benchmarks")
async def list_benchmarks():
    """List available benchmark files"""
    benchmarks_dir = Path("benchmarks")
    if not benchmarks_dir.exists():
        return {"benchmarks": []}
    
    files = [f.name for f in benchmarks_dir.glob("*") if f.is_file()]
    return {"benchmarks": files}


@app.get("/benchmarks/{filename}")
async def get_benchmark_file(filename: str):
    """Get a specific benchmark file"""
    file_path = Path("benchmarks") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return FileResponse(file_path)


@app.get("/")
async def read_index():
    return FileResponse('static/index.html')


@app.post("/optimize/foga", response_model=JobResponse)
async def optimize_foga(
    background_tasks: BackgroundTasks,
    source_file: UploadFile = File(...),
    test_input_file: Optional[UploadFile] = File(None)
):
    """
    Run FOGA (Flag Optimization with Genetic Algorithm) optimizer
    
    - **source_file**: C/C++ source code file
    - **test_input_file**: Optional test input for the program
    """
    try:
        # Save source file
        source_path = UPLOAD_DIR / f"{uuid.uuid4()}_{source_file.filename}"
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
        
        # Save test input if provided
        test_input_path = None
        if test_input_file:
            test_input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{test_input_file.filename}"
            with open(test_input_path, "wb") as f:
                shutil.copyfileobj(test_input_file.file, f)
        
        # Create job
        job_id = create_job("foga", str(source_path))
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimizer_task,
            job_id,
            "foga",
            str(source_path),
            str(test_input_path) if test_input_path else None
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="FOGA optimization job created successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/hbrf", response_model=JobResponse)
async def optimize_hbrf(
    background_tasks: BackgroundTasks,
    source_file: UploadFile = File(...),
    test_input_file: Optional[UploadFile] = File(None)
):
    """
    Run HBRF (Hybrid Bayesian-Random Forest) optimizer
    
    - **source_file**: C/C++ source code file
    - **test_input_file**: Optional test input for the program
    """
    try:
        # Save source file
        source_path = UPLOAD_DIR / f"{uuid.uuid4()}_{source_file.filename}"
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
        
        # Save test input if provided
        test_input_path = None
        if test_input_file:
            test_input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{test_input_file.filename}"
            with open(test_input_path, "wb") as f:
                shutil.copyfileobj(test_input_file.file, f)
        
        # Create job
        job_id = create_job("hbrf_optimizer", str(source_path))
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimizer_task,
            job_id,
            "hbrf_optimizer",
            str(source_path),
            str(test_input_path) if test_input_path else None
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="HBRF optimization job created successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/xgboost", response_model=JobResponse)
async def optimize_xgboost(
    background_tasks: BackgroundTasks,
    source_file: UploadFile = File(...),
    test_input_file: Optional[UploadFile] = File(None)
):
    """
    Run XGBoost optimizer
    
    - **source_file**: C/C++ source code file
    - **test_input_file**: Optional test input for the program
    """
    try:
        # Save source file
        source_path = UPLOAD_DIR / f"{uuid.uuid4()}_{source_file.filename}"
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
        
        # Save test input if provided
        test_input_path = None
        if test_input_file:
            test_input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{test_input_file.filename}"
            with open(test_input_path, "wb") as f:
                shutil.copyfileobj(test_input_file.file, f)
        
        # Create job
        job_id = create_job("xgboost_optimizer", str(source_path))
        
        # Start optimization in background
        background_tasks.add_task(
            run_optimizer_task,
            job_id,
            "xgboost_optimizer",
            str(source_path),
            str(test_input_path) if test_input_path else None
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="XGBoost optimization job created successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/compare", response_model=JobResponse)
async def optimize_compare(
    background_tasks: BackgroundTasks,
    source_file: UploadFile = File(...),
    test_input_file: Optional[UploadFile] = File(None)
):
    """
    Run comparison of all optimizers (FOGA, HBRF, XGBoost)
    
    - **source_file**: C/C++ source code file
    - **test_input_file**: Optional test input for the program
    """
    try:
        # Save source file
        source_path = UPLOAD_DIR / f"{uuid.uuid4()}_{source_file.filename}"
        with open(source_path, "wb") as f:
            shutil.copyfileobj(source_file.file, f)
        
        # Save test input if provided
        test_input_path = None
        if test_input_file:
            test_input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{test_input_file.filename}"
            with open(test_input_path, "wb") as f:
                shutil.copyfileobj(test_input_file.file, f)
        
        # Create job
        job_id = create_job("compare_optimizers", str(source_path))
        
        # Start comparison in background
        background_tasks.add_task(
            run_optimizer_task,
            job_id,
            "compare_optimizers",
            str(source_path),
            str(test_input_path) if test_input_path else None
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Comparison job created successfully"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=OptimizationJob)
async def get_job_status(job_id: str):
    """Get status of an optimization job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str):
    """Get detailed result of a completed job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "pending":
        raise HTTPException(status_code=400, detail="Job is still pending")
    elif job["status"] == "running":
        raise HTTPException(status_code=400, detail="Job is still running")
    elif job["status"] == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "job_id": job_id,
                "status": "failed",
                "error": job.get("error", "Unknown error")
            }
        )
    
    # Load detailed results
    result_file = RESULTS_DIR / f"{job_id}_result.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    
    return job


@app.get("/jobs")
async def list_jobs(status: Optional[str] = None, optimizer: Optional[str] = None):
    """
    List all jobs, optionally filtered by status or optimizer
    
    - **status**: Filter by status (pending, running, completed, failed)
    - **optimizer**: Filter by optimizer type
    """
    filtered_jobs = list(jobs.values())
    
    if status:
        filtered_jobs = [j for j in filtered_jobs if j["status"] == status]
    
    if optimizer:
        filtered_jobs = [j for j in filtered_jobs if j["optimizer"] == optimizer]
    
    return {
        "total": len(filtered_jobs),
        "jobs": filtered_jobs
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Delete associated files
    try:
        source_path = Path(job["source_file"])
        if source_path.exists():
            source_path.unlink()
        
        result_file = RESULTS_DIR / f"{job_id}_result.json"
        if result_file.exists():
            result_file.unlink()
    except Exception as e:
        pass
    
    # Remove from jobs dict
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs/{job_id}/download")
async def download_optimized_binary(job_id: str):
    """Download the optimized binary if available"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not completed")
    
    # Look for optimized binary based on optimizer type
    binary_names = {
        "foga": "optimized_binary",
        "hbrf_optimizer": "optimized_hbrf",
        "xgboost_optimizer": "optimized_xgboost"
    }
    
    binary_name = binary_names.get(job["optimizer"])
    if binary_name and os.path.exists(binary_name):
        return FileResponse(
            binary_name,
            media_type="application/octet-stream",
            filename=binary_name
        )
    
    raise HTTPException(status_code=404, detail="Optimized binary not found")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "jobs_count": len(jobs),
        "active_jobs": len([j for j in jobs.values() if j["status"] in ["pending", "running"]])
    }


@app.get("/jobs/{job_id}/stream")
async def stream_job_output(job_id: str):
    """Stream the output of a running job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        output_file = RESULTS_DIR / f"{job_id}_output.txt"
        
        # Wait for file to be created
        retries = 0
        while not output_file.exists():
            if retries > 50: # 5 seconds
                yield "data: Error: Output file not created\n\n"
                return
            time.sleep(0.1)
            retries += 1
            
        # Tail the file
        with open(output_file, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line}"
                else:
                    # Check if job is done
                    job = jobs[job_id]
                    if job["status"] in ["completed", "failed"]:
                        # Read any remaining lines
                        rest = f.read()
                        if rest:
                            yield f"data: {rest}"
                        yield "event: close\ndata: closed\n\n"
                        break
                    time.sleep(0.1)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
