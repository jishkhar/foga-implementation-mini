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

        // UI Reset
        runBtn.disabled = true;
        runBtn.textContent = 'Processing...';
        resultsPanel.style.display = 'block';
        loadingState.style.display = 'block';
        resultsContent.innerHTML = '';
        terminalOutput.style.display = 'none'; // Hide terminal initially until data comes
        outputContent.textContent = '';
        errorState.style.display = 'none';
        updateStatus('running');

        // Scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });

        try {
            // Fetch file content
            const fileResponse = await fetch(`/benchmarks/${benchmark}`);
            if (!fileResponse.ok) throw new Error('Failed to fetch benchmark file');
            const blob = await fileResponse.blob();
            const file = new File([blob], benchmark, { type: 'text/plain' });

            const formData = new FormData();
            formData.append('source_file', file);

            // Determine endpoint
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

            // Show terminal once job starts
            terminalOutput.style.display = 'block';
            startStreaming(job.job_id);
            pollJobStatus(job.job_id);

        } catch (err) {
            showError(err.message);
            resetBtn();
        }
    });

    function resetBtn() {
        runBtn.disabled = false;
        runBtn.textContent = '[ EXECUTE OPTIMIZATION ]';
    }

    async function pollJobStatus(jobId) {
        let failures = 0;
        const maxFailures = 5;

        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/jobs/${jobId}`);

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const job = await response.json();
                failures = 0; // Reset failures on success

                if (job.status === 'completed') {
                    clearInterval(pollInterval);
                    updateStatus('completed');
                    loadingState.style.display = 'none';
                    showResults(job);
                    resetBtn();
                } else if (job.status === 'failed') {
                    clearInterval(pollInterval);
                    updateStatus('failed');
                    loadingState.style.display = 'none';
                    showError(job.error || 'Optimization failed');
                    resetBtn();
                }
            } catch (err) {
                console.warn('Polling attempt failed:', err);
                failures++;

                if (failures >= maxFailures) {
                    clearInterval(pollInterval);
                    showError('Failed to poll job status after multiple attempts');
                    resetBtn();
                }
            }
        }, 2000);
    }

    function startStreaming(jobId) {
        const eventSource = new EventSource(`/jobs/${jobId}/stream`);

        eventSource.onmessage = (event) => {
            outputContent.textContent += event.data + '\n';
            // Auto-scroll
            const pre = document.querySelector('.terminal-output pre');
            if (pre) pre.scrollTop = pre.scrollHeight;
        };

        eventSource.addEventListener('close', () => {
            eventSource.close();
        });

        eventSource.onerror = (err) => {
            eventSource.close();
        };
    }

    function updateStatus(status) {
        statusBadge.className = `badge badge-${status}`;
        statusBadge.textContent = status.toUpperCase();
    }

    function showError(msg) {
        errorState.style.display = 'block';
        errorMessage.textContent = `ERROR: ${msg}`;
        loadingState.style.display = 'none';
    }

    function showResults(job) {
        let html = '';

        if (job.result) {
            // Format generic results
            if (job.result.best_time !== undefined && job.result.best_time !== null) {
                html += `
                    <div class="result-item">
                        <div class="result-label">BEST EXECUTION TIME</div>
                        <div class="result-value">${job.result.best_time.toFixed(6)} s</div>
                    </div>
                `;
            }
            if (job.result.total_time !== undefined && job.result.total_time !== null) {
                html += `
                    <div class="result-item">
                        <div class="result-label">OPTIMIZATION TIME</div>
                        <div class="result-value">${job.result.total_time.toFixed(2)} s</div>
                    </div>
                `;
            }
            if (job.result.evaluations !== undefined && job.result.evaluations !== null) {
                html += `
                    <div class="result-item">
                        <div class="result-label">TOTAL EVALUATIONS</div>
                        <div class="result-value">${job.result.evaluations}</div>
                    </div>
                `;
            }

            if (job.result.enabled_flags && job.result.enabled_flags.length > 0) {
                html += `
                    <div class="result-item" style="display:block">
                        <div class="result-label" style="margin-bottom:0.5rem; display: flex; justify-content: space-between; align-items: center;">
                            ENABLED FLAGS
                            <button onclick="copyFlags(this)" class="copy-btn" style="background: transparent; border: none; color: #888; cursor: pointer; padding: 4px; transition: all 0.2s;" title="Copy Flags">
                                <i data-lucide="copy" style="width: 16px; height: 16px;"></i>
                            </button>
                        </div>
                        <div class="result-value" id="flagsContent" style="font-size:0.8rem; color: #fff;">${job.result.enabled_flags.join(' ')}</div>
                    </div>
                `;
            }
        }

        // Handle comparison results
        if (job.optimizer === 'compare_optimizers' && job.result) {
            const r = job.result;
            const o3_time = r.baseline ? r.baseline['-O3'] : Infinity;

            // Collect all times
            const methods = [];

            // Add baseline methods
            if (r.baseline) {
                for (const [method, time] of Object.entries(r.baseline)) {
                    methods.push({ name: method, time: time });
                }
            }

            // Add optimizers
            if (r.FOGA) methods.push({ name: 'FOGA', time: r.FOGA.best_time });
            if (r.HBRF) methods.push({ name: 'HBRF', time: r.HBRF.best_time });
            if (r.XGBOOST) methods.push({ name: 'XGBOOST', time: r.XGBOOST.best_time });

            // Sort by time (ascending)
            methods.sort((a, b) => {
                const tA = a.time === null ? Infinity : a.time;
                const tB = b.time === null ? Infinity : b.time;
                return tA - tB;
            });

            // Build table HTML
            let tableRows = '';
            methods.forEach((item, index) => {
                const timeStr = (item.time === null || item.time === undefined || item.time === Infinity) ? 'Failed' : item.time.toFixed(6);

                let speedupStr = 'N/A';
                if (item.time !== null && item.time !== Infinity && o3_time !== Infinity && o3_time !== 0) {
                    const speedup = ((o3_time - item.time) / o3_time) * 100;
                    speedupStr = (speedup >= 0 ? '+' : '') + speedup.toFixed(2) + '%';
                }
                if (item.name === '-O3') speedupStr = '+0.00%';

                tableRows += `
                    <tr>
                        <td>${item.name}</td>
                        <td>${timeStr}</td>
                        <td>${speedupStr}</td>
                        <td>#${index + 1}</td>
                    </tr>
                `;
            });

            html += `
                <div class="result-item" style="display:block; width:100%;">
                    <div class="result-label" style="margin-bottom:1rem; border-bottom: 1px solid #444; padding-bottom: 0.5rem;">EXECUTION TIME COMPARISON</div>
                    <table style="width:100%; border-collapse: collapse; font-family: monospace; font-size: 0.9rem;">
                        <thead>
                            <tr style="border-bottom: 1px solid #444; text-align: left;">
                                <th style="padding: 8px;">Method</th>
                                <th style="padding: 8px;">Time (s)</th>
                                <th style="padding: 8px;">Speedup vs -O3</th>
                                <th style="padding: 8px;">Rank</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${tableRows}
                        </tbody>
                    </table>
                </div>
            `;
        }

        resultsContent.innerHTML = html;

        // Initialize Lucide icons
        if (window.lucide) {
            lucide.createIcons();
        }

        if (job.output) {
            outputContent.textContent = job.output;
        }

        // Render Charts
        renderCharts(job);
    }

    // Chart.js Global Defaults
    Chart.defaults.color = '#cccccc';
    Chart.defaults.font.family = '"JetBrains Mono", monospace';
    Chart.defaults.borderColor = '#444444';

    let currentChart = null;

    function renderCharts(job) {
        const chartsContainer = document.getElementById('chartsContainer');
        const canvas = document.getElementById('optimizationChart');
        const ctx = canvas.getContext('2d');

        if (currentChart) {
            currentChart.destroy();
            currentChart = null;
        }

        if (!job.result) return;

        chartsContainer.style.display = 'none';

        // Helper to create gradient
        function createGradient(ctx, colorStart, colorEnd) {
            const gradient = ctx.createLinearGradient(0, 0, 0, 400);
            gradient.addColorStop(0, colorStart);
            gradient.addColorStop(1, colorEnd);
            return gradient;
        }

        // Common Chart Options
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ecf0f1',
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#ecf0f1',
                    bodyColor: '#ecf0f1',
                    borderColor: '#444',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(6) + 's';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(68, 68, 68, 0.3)',
                        borderColor: '#444'
                    },
                    ticks: { color: '#bdc3c7' }
                },
                y: {
                    grid: {
                        color: 'rgba(68, 68, 68, 0.3)',
                        borderColor: '#444'
                    },
                    ticks: { color: '#bdc3c7' },
                    beginAtZero: false
                }
            }
        };

        // FOGA Chart
        if (job.optimizer === 'foga' && job.result.history && job.result.history.length > 0) {
            chartsContainer.style.display = 'block';
            // Resize container for better visibility
            canvas.style.height = '400px';

            const labels = job.result.history.map(h => `Gen ${h.iteration}`);
            const bestData = job.result.history.map(h => h.best);
            const avgData = job.result.history.map(h => h.avg);

            const bestGradient = createGradient(ctx, 'rgba(46, 204, 113, 0.5)', 'rgba(46, 204, 113, 0.0)');
            const avgGradient = createGradient(ctx, 'rgba(52, 152, 219, 0.5)', 'rgba(52, 152, 219, 0.0)');

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Best Fitness',
                            data: bestData,
                            borderColor: '#2ecc71',
                            backgroundColor: bestGradient,
                            borderWidth: 2,
                            pointBackgroundColor: '#2ecc71',
                            pointRadius: 4,
                            pointHoverRadius: 6,
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Avg Fitness',
                            data: avgData,
                            borderColor: '#3498db',
                            backgroundColor: avgGradient,
                            borderWidth: 2,
                            pointBackgroundColor: '#3498db',
                            pointRadius: 3,
                            fill: true,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    ...commonOptions,
                    plugins: {
                        ...commonOptions.plugins,
                        title: {
                            display: true,
                            text: 'FOGA: Genetic Algorithm Progress',
                            color: '#ecf0f1',
                            font: { size: 16, weight: 'bold' }
                        }
                    }
                }
            });
        }
        // HBRF & XGBoost Charts
        else if ((job.optimizer === 'hbrf_optimizer' || job.optimizer === 'xgboost_optimizer') && job.result.history && job.result.history.length > 0) {
            chartsContainer.style.display = 'block';
            canvas.style.height = '400px';

            const labels = job.result.history.map(h => h.iteration);
            const bestData = job.result.history.map(h => h.best);

            const color = job.optimizer === 'hbrf_optimizer' ? '#9b59b6' : '#e67e22'; // Purple for HBRF, Orange for XGB
            const gradient = createGradient(ctx, job.optimizer === 'hbrf_optimizer' ? 'rgba(155, 89, 182, 0.5)' : 'rgba(230, 126, 34, 0.5)', 'rgba(0,0,0,0)');

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Best Execution Time',
                            data: bestData,
                            borderColor: color,
                            backgroundColor: gradient,
                            borderWidth: 2,
                            pointBackgroundColor: color,
                            pointRadius: 2,
                            pointHoverRadius: 5,
                            fill: true,
                            tension: 0.1,
                            stepped: true
                        }
                    ]
                },
                options: {
                    ...commonOptions,
                    plugins: {
                        ...commonOptions.plugins,
                        title: {
                            display: true,
                            text: `${job.optimizer === 'hbrf_optimizer' ? 'HBRF' : 'XGBoost'} Optimization Trajectory`,
                            color: '#ecf0f1',
                            font: { size: 16, weight: 'bold' }
                        }
                    }
                }
            });
        }
        // Compare Chart
        else if (job.optimizer === 'compare_optimizers' && job.result) {
            chartsContainer.style.display = 'block';
            canvas.style.height = '400px';

            const r = job.result;
            const labels = [];
            const data = [];
            const backgroundColors = [];
            const borderColors = [];

            // Define colors for known methods
            const methodColors = {
                '-O1': '#95a5a6',
                '-O2': '#7f8c8d',
                '-O3': '#34495e',
                'FOGA': '#2ecc71',
                'HBRF': '#9b59b6',
                'XGBOOST': '#e67e22'
            };

            if (r.baseline) {
                for (const [method, time] of Object.entries(r.baseline)) {
                    if (time !== null && time !== Infinity) {
                        labels.push(method);
                        data.push(time);
                        const c = methodColors[method] || '#95a5a6';
                        backgroundColors.push(c);
                        borderColors.push(c);
                    }
                }
            }

            if (r.FOGA && r.FOGA.best_time !== Infinity) {
                labels.push('FOGA');
                data.push(r.FOGA.best_time);
                backgroundColors.push(methodColors['FOGA']);
                borderColors.push(methodColors['FOGA']);
            }
            if (r.HBRF && r.HBRF.best_time !== Infinity) {
                labels.push('HBRF');
                data.push(r.HBRF.best_time);
                backgroundColors.push(methodColors['HBRF']);
                borderColors.push(methodColors['HBRF']);
            }
            if (r.XGBOOST && r.XGBOOST.best_time !== Infinity) {
                labels.push('XGBOOST');
                data.push(r.XGBOOST.best_time);
                backgroundColors.push(methodColors['XGBOOST']);
                borderColors.push(methodColors['XGBOOST']);
            }

            currentChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Execution Time (s)',
                        data: data,
                        backgroundColor: backgroundColors,
                        borderColor: borderColors,
                        borderWidth: 1,
                        borderRadius: 4,
                        barPercentage: 0.6
                    }]
                },
                options: {
                    ...commonOptions,
                    indexAxis: 'y', // Horizontal bar chart for better label readability
                    plugins: {
                        ...commonOptions.plugins,
                        title: {
                            display: true,
                            text: 'Optimizer Performance Comparison',
                            color: '#ecf0f1',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            ...commonOptions.scales.x,
                            title: {
                                display: true,
                                text: 'Execution Time (seconds)',
                                color: '#95a5a6'
                            }
                        },
                        y: {
                            ...commonOptions.scales.y,
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    }

    // Expose copy function to global scope
    window.copyFlags = function (btn) {
        const flagsContent = document.getElementById('flagsContent');
        if (!flagsContent) {
            console.error('Element with id "flagsContent" not found');
            return;
        }

        const text = flagsContent.textContent;
        console.log('Attempting to copy:', text);

        // Visual feedback helper
        const showSuccess = () => {
            btn.innerHTML = '<i data-lucide="check" style="width: 16px; height: 16px;"></i>';
            btn.style.color = '#2ecc71';
            if (window.lucide) lucide.createIcons();

            setTimeout(() => {
                btn.innerHTML = '<i data-lucide="copy" style="width: 16px; height: 16px;"></i>';
                btn.style.color = '#888';
                if (window.lucide) lucide.createIcons();
            }, 2000);
        };

        const showError = (err) => {
            console.error('Copy failed:', err);
            btn.innerHTML = '<i data-lucide="x" style="width: 16px; height: 16px;"></i>';
            btn.style.color = '#e74c3c';
            if (window.lucide) lucide.createIcons();
        };

        // Try Clipboard API first
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text)
                .then(showSuccess)
                .catch(err => {
                    console.warn('Clipboard API failed, trying fallback...', err);
                    fallbackCopy(text);
                });
        } else {
            console.log('Clipboard API unavailable, using fallback');
            fallbackCopy(text);
        }

        function fallbackCopy(textToCopy) {
            try {
                const textArea = document.createElement("textarea");
                textArea.value = textToCopy;

                // Ensure it's not visible but part of the DOM
                textArea.style.position = "fixed";
                textArea.style.left = "-9999px";
                textArea.style.top = "0";
                document.body.appendChild(textArea);

                textArea.focus();
                textArea.select();

                const successful = document.execCommand('copy');
                document.body.removeChild(textArea);

                if (successful) {
                    showSuccess();
                } else {
                    throw new Error('execCommand returned false');
                }
            } catch (err) {
                showError(err);
            }
        }
    };
});