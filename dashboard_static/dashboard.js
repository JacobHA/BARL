// Dashboard state
let selectedMetrics = new Set(['eval/avg_reward']);
let allMetrics = [];
let metricsData = {};
let isPaused = false;
let xAxisMode = 'time';  // 'time' or 'steps'

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    startDataPolling();
    connectEventStream();
});

function initializeDashboard() {
    log('Dashboard initialized', 'success');
    fetchMetrics();
    updateAgentState();
    createPlot();
}

function setupEventListeners() {
    // Pause button
    document.getElementById('pauseBtn').addEventListener('click', function() {
        isPaused = !isPaused;
        const btn = this;
        if (isPaused) {
            sendCommand('pause', {});
            btn.textContent = '▶️ Resume';
            btn.classList.remove('btn-warning');
            btn.classList.add('btn-success');
        } else {
            sendCommand('resume', {});
            btn.textContent = '⏸️ Pause';
            btn.classList.remove('btn-success');
            btn.classList.add('btn-warning');
        }
    });

    // Evaluate button
    document.getElementById('evaluateBtn').addEventListener('click', function() {
        log('Starting evaluation...', 'info');
        sendCommand('evaluate', {n_episodes: 10});
    });

    // Save button
    document.getElementById('saveBtn').addEventListener('click', function() {
        log('Saving model...', 'info');
        sendCommand('save', {});
    });
    
    // Toggle axis button
    document.getElementById('toggleAxisBtn').addEventListener('click', async function() {
        const response = await fetch('/api/axis/toggle', {method: 'POST'});
        const result = await response.json();
        xAxisMode = result.x_axis_mode;
        
        const btn = this;
        if (xAxisMode === 'time') {
            btn.textContent = '🕐 Show Steps';
        } else {
            btn.textContent = '👣 Show Time';
        }
        
        updatePlot();
        log(`X-axis switched to ${xAxisMode}`, 'info');
    });

    // Metric search
    document.getElementById('metricSearch').addEventListener('input', function(e) {
        filterMetrics(e.target.value);
    });
}

function setLearningRate() {
    const lr = document.getElementById('lrInput').value;
    if (lr) {
        sendCommand('set_learning_rate', {value: parseFloat(lr)});
    }
}

function setEpsilon() {
    const eps = document.getElementById('epsInput').value;
    if (eps) {
        sendCommand('set_epsilon', {value: parseFloat(eps)});
    }
}

async function sendCommand(command, params) {
    try {
        const response = await fetch('/api/agent/command', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({command, params})
        });
        const result = await response.json();
        
        if (result.success) {
            log(result.result.message || 'Command executed', 'success');
        } else {
            log(result.error || 'Command failed', 'error');
        }
    } catch (error) {
        log('Error sending command: ' + error, 'error');
    }
}

async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        
        allMetrics = data.metrics.sort();
        metricsData = data.data;
        xAxisMode = data.x_axis_mode || 'time';
        
        updateMetricList();
        updatePlot();
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }
}

async function updateAgentState() {
    try {
        const response = await fetch('/api/agent/state');
        const state = await response.json();
        
        document.getElementById('stat-steps').textContent = state.learn_steps || 0;
        document.getElementById('stat-episodes').textContent = state.num_episodes || 0;
        document.getElementById('stat-lr').textContent = state.learning_rate ? state.learning_rate.toFixed(6) : '-';
        document.getElementById('stat-epsilon').textContent = state.epsilon ? state.epsilon.toFixed(4) : '-';
        
        // Update input fields
        if (state.learning_rate) {
            document.getElementById('lrInput').value = state.learning_rate;
        }
        if (state.epsilon !== null) {
            document.getElementById('epsInput').value = state.epsilon;
        }
    } catch (error) {
        console.error('Error updating agent state:', error);
    }
}

function updateMetricList() {
    const container = document.getElementById('metricList');
    container.innerHTML = '';
    
    allMetrics.forEach(metric => {
        const item = document.createElement('div');
        item.className = 'metric-item';
        if (selectedMetrics.has(metric)) {
            item.classList.add('active');
        }
        item.textContent = metric;
        item.onclick = () => toggleMetric(metric);
        container.appendChild(item);
    });
}

function toggleMetric(metric) {
    if (selectedMetrics.has(metric)) {
        selectedMetrics.delete(metric);
    } else {
        selectedMetrics.add(metric);
    }
    updateMetricList();
    updatePlot();
}

function filterMetrics(query) {
    const items = document.querySelectorAll('.metric-item');
    const lowerQuery = query.toLowerCase();
    
    items.forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(lowerQuery) ? 'block' : 'none';
    });
}

function createPlot() {
    const xAxisTitle = xAxisMode === 'time' ? 'Training Time (seconds)' : 'Environment Steps';
    const layout = {
        title: 'Training Metrics',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0.3)',
        font: {color: '#eee'},
        xaxis: {
            title: xAxisTitle,
            gridcolor: 'rgba(255,255,255,0.1)',
            color: '#aaa'
        },
        yaxis: {
            title: 'Value',
            gridcolor: 'rgba(255,255,255,0.1)',
            color: '#aaa'
        },
        legend: {
            bgcolor: 'rgba(0,0,0,0.5)',
            bordercolor: 'rgba(0,217,255,0.3)',
            borderwidth: 1
        },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('mainPlot', [], layout, {responsive: true});
}

function updatePlot() {
    const traces = [];
    const colors = ['#00d9ff', '#00ff88', '#ffa500', '#ff4444', '#ff00ff', '#ffff00'];
    let colorIndex = 0;
    
    const xAxisTitle = xAxisMode === 'time' ? 'Training Time (seconds)' : 'Environment Steps';
    const xAxisLabel = xAxisMode === 'time' ? 'Time' : 'Step';
    
    selectedMetrics.forEach(metric => {
        if (metricsData[metric]) {
            const xData = xAxisMode === 'time' ? metricsData[metric].times : metricsData[metric].steps;
            
            if (xData && xData.length > 0) {
                traces.push({
                    x: xData,
                    y: metricsData[metric].values,
                    name: metric,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {color: colors[colorIndex % colors.length], width: 2},
                    marker: {size: 4},
                    hovertemplate: `<b>%{fullData.name}</b><br>${xAxisLabel}: %{x:.2f}<br>Value: %{y:.4f}<extra></extra>`
                });
                colorIndex++;
            }
        }
    });
    
    if (traces.length === 0) {
        traces.push({
            x: [0],
            y: [0],
            type: 'scatter',
            mode: 'markers',
            marker: {size: 0},
            showlegend: false
        });
    }
    
    Plotly.react('mainPlot', traces, {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0.3)',
        font: {color: '#eee'},
        xaxis: {
            title: xAxisTitle,
            gridcolor: 'rgba(255,255,255,0.1)',
            color: '#aaa'
        },
        yaxis: {
            title: 'Value',
            gridcolor: 'rgba(255,255,255,0.1)',
            color: '#aaa'
        },
        legend: {
            bgcolor: 'rgba(0,0,0,0.5)',
            bordercolor: 'rgba(0,217,255,0.3)',
            borderwidth: 1
        },
        hovermode: 'x unified'
    });
}

function startDataPolling() {
    setInterval(() => {
        fetchMetrics();
        updateAgentState();
    }, 2000);
}

function connectEventStream() {
    const eventSource = new EventSource('/api/stream');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        // Update stats in real-time
        if (data.agent_state) {
            if (data.agent_state.epsilon !== null) {
                document.getElementById('stat-epsilon').textContent = data.agent_state.epsilon.toFixed(4);
            }
        }
    };
    
    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
    };
}

function log(message, type = 'info') {
    const logOutput = document.getElementById('logOutput');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    logOutput.insertBefore(entry, logOutput.firstChild);
    
    // Keep only last 50 entries
    while (logOutput.children.length > 50) {
        logOutput.removeChild(logOutput.lastChild);
    }
}