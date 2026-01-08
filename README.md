# Compiler Optimization API

A comprehensive compiler optimization platform that provides access to multiple state-of-the-art optimization algorithms through a RESTful API and modern web interface. Compare and benchmark different optimization strategies including AutoFlag (Genetic Algorithm), HBRF (Hybrid Bayesian-Random Forest), and XGBoost-based optimizers.

## 🌟 Features

- **Multiple Optimization Algorithms**
  - **AutoFlag**: Flag optimization using genetic algorithms for intelligent compiler flag selection
  - **HBRF**: Hybrid Bayesian-Random Forest optimizer combining Bayesian optimization with greedy refinement
  - **XGBoost**: Machine learning-based optimizer using gradient boosting for flag prediction
  - **Comparison Tool**: Run all optimizers simultaneously and generate comprehensive performance reports

- **RESTful API**: FastAPI-based backend with automatic interactive documentation
- **Web Interface**: Modern, responsive UI for easy benchmark submission and result visualization
- **Real-time Monitoring**: Stream optimization progress and view live results
- **Benchmark Library**: Pre-configured C benchmarks for testing and comparison

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
  - [Web Interface](#web-interface)
  - [API Usage](#api-usage)
  - [CLI Usage](#cli-usage)
- [Optimizer Comparison](#-optimizer-comparison)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Configuration](#-configuration)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- GCC/G++ compiler
- pip or uv package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mini_project_2025
   ```

2. **Create and activate virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or using `uv`:
   ```bash
   uv pip install -r requirements.txt
   ```

## ⚡ Quick Start

### Start the API Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the server
python3 api.py
```

The server will start at `http://localhost:8000`

### Access the Web Interface

Open your browser and navigate to:
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Run Your First Optimization

1. Navigate to http://localhost:8000
2. Select a benchmark file (e.g., `matrix_multiply.c`)
3. Choose an optimizer (AutoFlag, HBRF, or XGBoost)
4. Click "Run Optimization"
5. Monitor progress and view results

## 📁 Project Structure

```
mini_project_2025/
├── api.py                      # FastAPI server and main entry point
├── scripts/                    # Optimizer implementations
│   ├── autoflag.py            # AutoFlag genetic algorithm optimizer
│   ├── hbrf_optimizer.py      # HBRF hybrid optimizer
│   ├── xgboost_optimizer.py   # XGBoost-based optimizer
│   └── compare_optimizers.py  # Comparison tool for all optimizers
├── static/                     # Web frontend
│   ├── index.html             # Main UI
│   ├── script.js              # Frontend logic
│   └── style.css              # Styling
├── benchmarks/                 # Sample C benchmarks
│   ├── matrix_multiply.c      # Matrix multiplication benchmark
│   └── numeralIntegration.c   # Numerical integration benchmark
├── uploads/                    # Uploaded source files (generated at runtime)
├── results/                    # Optimization results (generated at runtime)
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
└── README.md                  # This file
```

## 💻 Usage

### Web Interface

The web interface provides the easiest way to run optimizations:

1. **Access the UI** at http://localhost:8000
2. **Upload or Select** a C/C++ source file
3. **Choose Optimizer**:
   - AutoFlag: Best for thorough exploration
   - HBRF: Balanced speed and quality
   - XGBoost: Fast learning-based approach
   - Compare: Run all three and get comprehensive comparison
4. **Monitor Progress**: Real-time status updates
5. **View Results**: Best execution time, enabled flags, and optimization statistics

### API Usage

#### Submit an Optimization Job

```bash
curl -X POST "http://localhost:8000/optimize/autoflag" \
  -F "source_file=@benchmarks/matrix_multiply.c"
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "AutoFlag optimization job created successfully"
}
```

#### Check Job Status

```bash
curl "http://localhost:8000/jobs/{job_id}"
```

#### Get Results

```bash
curl "http://localhost:8000/jobs/{job_id}/result"
```

See [API_README.md](API_README.md) for complete API documentation.

### CLI Usage

Run optimizers directly from the command line:

```bash
# AutoFlag
python3 scripts/autoflag.py benchmarks/matrix_multiply.c

# HBRF
python3 scripts/hbrf_optimizer.py benchmarks/matrix_multiply.c

# XGBoost
python3 scripts/xgboost_optimizer.py benchmarks/matrix_multiply.c

# Compare all optimizers
python3 scripts/compare_optimizers.py benchmarks/matrix_multiply.c
```

## 📊 Optimizer Comparison

| Optimizer | Approach | Speed | Quality | Best For |
|-----------|----------|-------|---------|----------|
| **AutoFlag** | Genetic Algorithm | ⭐⭐ | ⭐⭐⭐⭐⭐ | Thorough exploration, maximum performance |
| **HBRF** | Hybrid Bayesian-RF | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Balanced optimization, good results fast |
| **XGBoost** | Gradient Boosting | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick predictions, learning from patterns |

### When to Use Each Optimizer

- **AutoFlag**: When you need the absolute best performance and can afford longer optimization times
- **HBRF**: For production use where you need good results in reasonable time
- **XGBoost**: When you need quick results or are optimizing similar codebases repeatedly
- **Compare**: When benchmarking or unsure which optimizer suits your use case

## 📖 API Documentation

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web interface |
| `/optimize/autoflag` | POST | Run AutoFlag optimizer |
| `/optimize/hbrf` | POST | Run HBRF optimizer |
| `/optimize/xgboost` | POST | Run XGBoost optimizer |
| `/optimize/compare` | POST | Run all optimizers and compare |
| `/jobs/{job_id}` | GET | Get job status |
| `/jobs/{job_id}/result` | GET | Get job results |
| `/jobs/{job_id}/stream` | GET | Stream job output (SSE) |
| `/jobs` | GET | List all jobs |
| `/health` | GET | Health check |
| `/benchmarks` | GET | List available benchmarks |

For detailed API documentation with examples, see [API_README.md](API_README.md) or visit http://localhost:8000/docs

## 🛠️ Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run a test optimization
curl -X POST "http://localhost:8000/optimize/autoflag" \
  -F "source_file=@benchmarks/matrix_multiply.c"
```

### Adding Custom Benchmarks

Add your C/C++ files to the `benchmarks/` directory:

```bash
cp your_code.c benchmarks/
```

They will automatically appear in the web interface and be available via the API.

## ⚙️ Configuration

### Default Settings

- **Upload Directory**: `uploads/` (created automatically)
- **Results Directory**: `results/` (created automatically)
- **Server Host**: `0.0.0.0` (all interfaces)
- **Server Port**: `8000`
- **Job Timeout**: 2 hours

### Customization

Modify `api.py` to change:
- Optimization parameters
- Timeout values
- Storage locations
- CORS settings

## 🔧 Tech Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Optimizers**: NumPy, scikit-optimize, scikit-learn, XGBoost
- **Visualization**: Matplotlib
- **Server**: Uvicorn (ASGI)

## 📝 Notes

- **Job Storage**: Jobs are stored in memory. For production, integrate Redis or a database.
- **File Cleanup**: Uploaded files and results accumulate in `uploads/` and `results/`. Clean periodically or implement automatic cleanup.
- **Concurrent Jobs**: Multiple optimizations can run simultaneously in the background.
- **Security**: Add authentication/authorization for production deployment.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional optimization algorithms
- Performance visualizations
- Job persistence (database integration)
- Authentication and user management
- Docker containerization
- Batch optimization support

## 📄 License

[Specify your license here]

## 🔗 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GCC Optimization Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
- [Full API Documentation](API_README.md)

## 📧 Support

For questions or issues:
- Check the interactive API docs at `/docs`
- Review examples in [API_README.md](API_README.md)
- Test with the health endpoint: `/health`

---

**Made with ❤️ for compiler optimization research and education**
