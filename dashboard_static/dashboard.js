// Dashboard state
let selectedMetrics = new Set(['eval/avg_reward']);
let allMetrics = [];
let metricsData = {};
let isPaused = false;
let xAxisMode = 'steps';  // 'time' or 'steps'

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
    }, 2000);
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