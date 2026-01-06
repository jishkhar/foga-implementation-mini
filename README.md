# mini_project_2025



Walkthrough - Benchmark API Frontend
I have created a web frontend for the Compiler Optimization API. This allows users to easily select benchmarks and run optimizations from a browser.

Changes
Backend (
api.py
)
Added StaticFiles support to serve the frontend.
Added GET /benchmarks to list available benchmark files.
Added GET /benchmarks/{filename} to serve individual benchmark files.
Updated GET / to serve the 
index.html
 file.
Frontend (static/)
index.html: The main interface with a benchmark selector and optimizer options.
style.css: Modern styling for the interface.
script.js: Logic to fetch benchmarks, submit jobs, and poll for results.
Verification Results
API Endpoints
GET /: Successfully serves 
index.html
.
GET /benchmarks: Returns the list of files in 
benchmarks/
.
GET /benchmarks/{filename}: Successfully downloads the benchmark file.
POST /optimize/*: Endpoints are ready to accept requests from the frontend.
Frontend UI
Displays the list of benchmarks.
Allows selecting an optimizer.
Submits the job and displays the status (Pending -> Running -> Completed).
Shows the results (Best Time, Total Time, Evaluations).
How to Run
Start the API server:

.venv/bin/python3 api.py
(Note: I created a virtual environment .venv and installed dependencies there).

Open the frontend: Navigate to http://localhost:8000 in your web browser.

Run a benchmark:
-falign-jumps -fcompare-elim -fdelayed-branch -fdse -ffloat-store -fipa-pure-const -fjump-tables -fmove-loop-invariants -fschedule-insns -ftree-copy-prop -ftree-dce -ftree-ter -ftree-vrp
Select a file (e.g., 
matrix_multiply.c
).
Choose an optimizer (e.g., AutoFlag).
Click "Run Optimization".
