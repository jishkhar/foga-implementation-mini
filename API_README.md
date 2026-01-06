# Compiler Optimization API Documentation

A FastAPI-based REST API for running compiler optimization tools including AutoFlag (Flag Optimization with Genetic Algorithm), HBRF (Hybrid Bayesian-Random Forest), XGBoost optimizer, and comparison tools.

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

3. Access the API:
- Base URL: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📋 API Endpoints

### Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Compiler Optimization API",
  "version": "1.0.0",
  "endpoints": {
    "autoflag": "/optimize/autoflag",
    "hbrf": "/optimize/hbrf",
    "xgboost": "/optimize/xgboost",
    "compare": "/optimize/compare",
    "status": "/jobs/{job_id}",
    "result": "/jobs/{job_id}/result",
    "list": "/jobs"
  }
}
```

---

### Run AutoFlag Optimization
```http
POST /optimize/autoflag
```

Run Flag Optimization with Genetic Algorithm.

**Parameters:**
- `source_file` (file, required): C/C++ source code file
- `test_input_file` (file, optional): Test input file for the program

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/optimize/autoflag" \
  -F "source_file=@matrix_multiply.c" \
  -F "test_input_file=@input.txt"
```

**Example using Python requests:**
```python
import requests

files = {
    'source_file': open('matrix_multiply.c', 'rb'),
    'test_input_file': open('input.txt', 'rb')  # optional
}

response = requests.post('http://localhost:8000/optimize/autoflag', files=files)
job = response.json()
print(f"Job ID: {job['job_id']}")
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "AutoFlag optimization job created successfully"
}
```

---

### Run HBRF Optimization
```http
POST /optimize/hbrf
```

Run Hybrid Bayesian-Random Forest optimizer.

**Parameters:**
- `source_file` (file, required): C/C++ source code file
- `test_input_file` (file, optional): Test input file for the program

**Example:**
```bash
curl -X POST "http://localhost:8000/optimize/hbrf" \
  -F "source_file=@fibonacci.c"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "pending",
  "message": "HBRF optimization job created successfully"
}
```

---

### Run XGBoost Optimization
```http
POST /optimize/xgboost
```

Run XGBoost-based optimizer.

**Parameters:**
- `source_file` (file, required): C/C++ source code file
- `test_input_file` (file, optional): Test input file for the program

**Example:**
```bash
curl -X POST "http://localhost:8000/optimize/xgboost" \
  -F "source_file=@program.c"
```

---

### Run Full Comparison
```http
POST /optimize/compare
```

Run all three optimizers (AutoFlag, HBRF, XGBoost) and generate comprehensive comparison.

**Parameters:**
- `source_file` (file, required): C/C++ source code file
- `test_input_file` (file, optional): Test input file for the program

**Example:**
```bash
curl -X POST "http://localhost:8000/optimize/compare" \
  -F "source_file=@matrix_multiply.c"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "pending",
  "message": "Comparison job created successfully"
}
```

---

### Check Job Status
```http
GET /jobs/{job_id}
```

Get the current status of an optimization job.

**Example:**
```bash
curl "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "optimizer": "foga",
  "source_file": "uploads/abc123_matrix_multiply.c",
  "created_at": "2025-12-03T10:30:00.000000",
  "started_at": "2025-12-03T10:30:01.000000",
  "completed_at": null,
  "result": null,
  "error": null
}
```

**Status values:**
- `pending`: Job is queued
- `running`: Job is currently executing
- `completed`: Job finished successfully
- `failed`: Job encountered an error

---

### Get Job Result
```http
GET /jobs/{job_id}/result
```

Get detailed results of a completed job.

**Example:**
```bash
curl "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000/result"
```

**Response (AutoFlag/HBRF/XGBoost):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "optimizer": "autoflag",
  "output": "... full console output ...",
  "result": {
    "best_time": 0.123456,
    "total_time": 45.67,
    "evaluations": 2770,
    "enabled_flags": [
      "-faggressive-loop-optimizations",
      "-finline-functions",
      "-ftree-vectorize"
    ]
  }
}
```

**Response (Comparison):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440002",
  "optimizer": "compare_optimizers",
  "result": {
    "timestamp": "2025-12-03T10:35:00.000000",
    "source_file": "matrix_multiply.c",
    "baseline": {
      "-O1": 1.234,
      "-O2": 0.567,
      "-O3": 0.456
    },
    "AutoFlag": {
      "best_time": 0.412,
      "total_time": 123.45,
      "evaluations": 2770
    },
    "HBRF": {
      "best_time": 0.398,
      "total_time": 98.76,
      "evaluations": 160
    },
    "XGBOOST": {
      "best_time": 0.405,
      "total_time": 87.32,
      "evaluations": 150
    },
    "winner": "HBRF",
    "improvements": {
      "AutoFlag_vs_O3": -9.65,
      "HBRF_vs_O3": -12.72,
      "XGBOOST_vs_O3": -11.18
    }
  }
}
```

---

### List All Jobs
```http
GET /jobs?status={status}&optimizer={optimizer}
```

List all jobs with optional filtering.

**Query Parameters:**
- `status` (optional): Filter by status (`pending`, `running`, `completed`, `failed`)
- `optimizer` (optional): Filter by optimizer type (`autoflag`, `hbrf_optimizer`, `xgboost_optimizer`, `compare_optimizers`)

**Examples:**
```bash
# Get all jobs
curl "http://localhost:8000/jobs"

# Get running jobs
curl "http://localhost:8000/jobs?status=running"

# Get completed AutoFlag jobs
curl "http://localhost:8000/jobs?optimizer=autoflag&status=completed"
```

**Response:**
```json
{
  "total": 5,
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "optimizer": "autoflag",
      "created_at": "2025-12-03T10:30:00.000000",
      "started_at": "2025-12-03T10:30:01.000000",
      "completed_at": "2025-12-03T10:32:15.000000"
    }
  ]
}
```

---

### Delete Job
```http
DELETE /jobs/{job_id}
```

Delete a job and its associated files.

**Example:**
```bash
curl -X DELETE "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000"
```

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

---

### Download Optimized Binary
```http
GET /jobs/{job_id}/download
```

Download the optimized binary produced by the job.

**Example:**
```bash
curl -O "http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000/download"
```

---

### Health Check
```http
GET /health
```

Check API health and get statistics.

**Example:**
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-03T10:40:00.000000",
  "jobs_count": 10,
  "active_jobs": 2
}
```

---

## 🔄 Complete Workflow Example

### Python Example

```python
import requests
import time

API_BASE = "http://localhost:8000"

# 1. Submit optimization job
with open('matrix_multiply.c', 'rb') as f:
    files = {'source_file': f}
    response = requests.post(f"{API_BASE}/optimize/autoflag", files=files)
    job_id = response.json()['job_id']
    print(f"Job submitted: {job_id}")

# 2. Poll for completion
while True:
    response = requests.get(f"{API_BASE}/jobs/{job_id}")
    job = response.json()
    status = job['status']
    print(f"Status: {status}")
    
    if status == 'completed':
        break
    elif status == 'failed':
        print(f"Error: {job['error']}")
        exit(1)
    
    time.sleep(5)  # Wait 5 seconds before checking again

# 3. Get results
response = requests.get(f"{API_BASE}/jobs/{job_id}/result")
result = response.json()
print(f"Best execution time: {result['result']['best_time']} seconds")
print(f"Enabled flags: {len(result['result']['enabled_flags'])}")

# 4. Download optimized binary (optional)
response = requests.get(f"{API_BASE}/jobs/{job_id}/download")
with open('optimized_binary', 'wb') as f:
    f.write(response.content)
print("Optimized binary downloaded")
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

const API_BASE = 'http://localhost:8000';

async function optimizeCode() {
  // 1. Submit job
  const form = new FormData();
  form.append('source_file', fs.createReadStream('matrix_multiply.c'));
  
  const submitResponse = await axios.post(
    `${API_BASE}/optimize/autoflag`,
    form,
    { headers: form.getHeaders() }
  );
  
  const jobId = submitResponse.data.job_id;
  console.log(`Job submitted: ${jobId}`);
  
  // 2. Poll for completion
  while (true) {
    const statusResponse = await axios.get(`${API_BASE}/jobs/${jobId}`);
    const status = statusResponse.data.status;
    console.log(`Status: ${status}`);
    
    if (status === 'completed') break;
    if (status === 'failed') {
      console.error(`Error: ${statusResponse.data.error}`);
      return;
    }
    
    await new Promise(resolve => setTimeout(resolve, 5000));
  }
  
  // 3. Get results
  const resultResponse = await axios.get(`${API_BASE}/jobs/${jobId}/result`);
  console.log(`Best time: ${resultResponse.data.result.best_time}s`);
}

optimizeCode();
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```env
# Server configuration
HOST=0.0.0.0
PORT=8000

# Timeout settings (seconds)
COMPILATION_TIMEOUT=30
EXECUTION_TIMEOUT=10
JOB_TIMEOUT=7200

# Storage paths
UPLOAD_DIR=uploads
RESULTS_DIR=results
```

---

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Running in Production

```bash
# Using gunicorn with uvicorn workers
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install GCC/G++
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t compiler-optimizer-api .
docker run -p 8000:8000 compiler-optimizer-api
```

---

## 📊 Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad request (invalid parameters) |
| 404  | Job not found |
| 500  | Internal server error |

---

## ⚠️ Important Notes

1. **Job Storage**: Jobs are stored in memory. In production, use Redis or a database.
2. **File Cleanup**: Uploaded files and results should be cleaned up periodically.
3. **Timeouts**: Long-running optimizations may timeout (default: 2 hours).
4. **Concurrent Jobs**: Multiple jobs can run simultaneously in the background.
5. **Security**: Add authentication for production use.

---

## 📖 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [AutoFlag Paper](https://ieeexplore.ieee.org/document/example)
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

---

## 🤝 Support

For issues or questions, please check:
- API interactive docs: `/docs`
- Health endpoint: `/health`
- Job status endpoint: `/jobs/{job_id}`
