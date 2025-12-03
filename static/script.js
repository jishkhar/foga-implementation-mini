document.addEventListener('DOMContentLoaded', () => {
    const benchmarkSelect = document.getElementById('benchmarkSelect');
    const form = document.getElementById('optimizationForm');
    const runBtn = document.getElementById('runBtn');
    const resultsPanel = document.getElementById('resultsPanel');
    const statusBadge = document.getElementById('statusBadge');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const terminalOutput = document.getElementById('terminalOutput');
    const outputContent = document.getElementById('outputContent');
    const errorState = document.getElementById('errorState');
    const errorMessage = errorState.querySelector('.error-message');

    // Fetch benchmarks on load
    fetch('/benchmarks')
        .then(response => response.json())
        .then(data => {
            benchmarkSelect.innerHTML = '<option value="" disabled selected>Select a benchmark</option>';
            data.benchmarks.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                benchmarkSelect.appendChild(option);
            });
        })
        .catch(err => {
            console.error('Failed to load benchmarks:', err);
            benchmarkSelect.innerHTML = '<option value="" disabled>Error loading benchmarks</option>';
        });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const benchmark = benchmarkSelect.value;
        const optimizer = document.querySelector('input[name="optimizer"]:checked').value;

        if (!benchmark || !optimizer) return;

        // Reset UI
        runBtn.disabled = true;
        resultsPanel.style.display = 'block';
        loadingState.style.display = 'none';
        resultsContent.innerHTML = '';
        terminalOutput.style.display = 'block';
        outputContent.textContent = 'Initializing optimization...\n';
        errorState.style.display = 'none';
        updateStatus('running');


        try {
            // 1. Get the file content first (we need to upload it)
            // Since we can't easily read local files from browser JS without user selection,
            // and the API expects an upload, we'll need a way to tell the API to use a local file
            // OR we fetch the file content first if the API supported it.

            // WAIT: The API expects an UploadFile. 
            // But the files are already on the server in `benchmarks/`.
            // The API design assumes the user uploads a file.
            // I should modify the API to accept a file path OR an upload.
            // OR, I can fetch the file content from the server (if I add an endpoint) and then re-upload it.

            // Let's try to fetch the file content first.
            // I'll add a helper endpoint to get benchmark content or just modify the API to accept a 'benchmark_name'

            // Actually, the easiest way without changing the API too much is to:
            // 1. Fetch the file content from the server (I need an endpoint for this).
            // 2. Create a Blob and upload it.

            // Let's assume I can't easily change the API to accept paths right now (though I should).
            // I'll fetch the file content via a new endpoint I'll add, or just try to read it if I can.

            // Wait, I am the developer. I can change the API.
            // I will modify the frontend to just send the benchmark name, 
            // BUT the current API endpoints (`/optimize/foga` etc) expect `source_file: UploadFile`.

            // Workaround: Fetch the file from the server using a new endpoint, then upload it back.
            // I'll add `GET /benchmarks/{filename}` to api.py first.

            const fileResponse = await fetch(`/benchmarks/${benchmark}`);
            if (!fileResponse.ok) throw new Error('Failed to fetch benchmark file');
            const blob = await fileResponse.blob();
            const file = new File([blob], benchmark, { type: 'text/plain' });

            const formData = new FormData();
            formData.append('source_file', file);

            // 2. Start optimization
            let endpoint = '';
            if (optimizer === 'compare') {
                endpoint = '/optimize/compare';
            } else {
                endpoint = `/optimize/${optimizer}`;
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Optimization failed to start');
            }

            const job = await response.json();

            // Start streaming output
            startStreaming(job.job_id);

            // Also poll for status to know when it's fully done (for results JSON)
            pollJobStatus(job.job_id);

        } catch (err) {
            showError(err.message);
            runBtn.disabled = false;
        }
    });

    async function pollJobStatus(jobId) {
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/jobs/${jobId}`);
                const job = await response.json();

                if (job.status === 'completed') {
                    clearInterval(pollInterval);
                    updateStatus('completed');
                    // loadingState.style.display = 'none'; // Already hidden
                    showResults(job);
                    runBtn.disabled = false;
                } else if (job.status === 'failed') {
                    clearInterval(pollInterval);
                    updateStatus('failed');
                    // loadingState.style.display = 'none'; // Already hidden
                    showError(job.error || 'Optimization failed');
                    runBtn.disabled = false;
                }
            } catch (err) {
                clearInterval(pollInterval);
                showError('Failed to poll job status');
                runBtn.disabled = false;
            }
        }, 2000);
    }

    function startStreaming(jobId) {
        const eventSource = new EventSource(`/jobs/${jobId}/stream`);

        eventSource.onmessage = (event) => {
            // Append new data to the terminal output
            // outputContent.textContent += event.data + '\n'; // This might double newlines if data already has them
            // The data from server is line-based.
            outputContent.textContent += event.data;

            // Auto-scroll to bottom
            const terminalContainer = document.querySelector('.terminal-output');
            if (terminalContainer) {
                terminalContainer.scrollTop = terminalContainer.scrollHeight;
            }
        };

        eventSource.addEventListener('close', () => {
            eventSource.close();
        });

        eventSource.onerror = (err) => {
            console.error('EventSource failed:', err);
            eventSource.close();
        };
    }

    function updateStatus(status) {
        statusBadge.className = `badge badge-${status}`;
        statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    function showError(msg) {
        errorState.style.display = 'block';
        errorMessage.textContent = msg;
        loadingState.style.display = 'none';
    }

    function showResults(job) {
        let html = '';

        if (job.result) {
            html += `
                <div class="result-item">
                    <div class="result-label">Best Execution Time</div>
                    <div class="result-value">${job.result.best_time ? job.result.best_time.toFixed(6) + ' s' : 'N/A'}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Total Optimization Time</div>
                    <div class="result-value">${job.result.total_time ? job.result.total_time.toFixed(2) + ' s' : 'N/A'}</div>
                </div>
                <div class="result-item">
                    <div class="result-label">Evaluations</div>
                    <div class="result-value">${job.result.evaluations || 'N/A'}</div>
                </div>
            `;

            if (job.result.enabled_flags && job.result.enabled_flags.length > 0) {
                html += `
                    <div class="result-item">
                        <div class="result-label">Enabled Flags</div>
                        <pre>${job.result.enabled_flags.join('\n')}</pre>
                    </div>
                `;
            }
        }

        if (job.optimizer === 'compare_optimizers' && job.result) {
            // Handle comparison results specifically if structure differs
            html = '<pre>' + JSON.stringify(job.result, null, 2) + '</pre>';
        }

        resultsContent.innerHTML = html;

        if (job.output) {
            terminalOutput.style.display = 'block';
            // We don't overwrite here because we might have streamed it.
            // But to be safe, we can ensure it's fully consistent.
            outputContent.textContent = job.output;
        }
    }
});
