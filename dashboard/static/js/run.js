let selectedMetrics = new Set(['eval/avg_reward']);
let metricsData = {};
let xAxisMode = 'time';

const runId = window.__RUN_ID__;

async function fetchMetrics() {
  const res = await fetch(`/api/runs/${runId}/metrics`);
  const data = await res.json();
  metricsData = data.data || {};
  xAxisMode = data.x_axis_mode || 'time';
  renderMetricList(data.metrics || []);
  updatePlot();
  updateAxisLabel();
}

async function fetchNotes() {
  const res = await fetch(`/api/runs/${runId}/notes`);
  const data = await res.json();
  document.getElementById('notesArea').value = data.notes || '';
}

async function saveNotes() {
  const notes = document.getElementById('notesArea').value;
  await fetch(`/api/runs/${runId}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes })
  });
}

async function fetchHparams() {
  const res = await fetch(`/api/runs/${runId}/hparams`);
  const data = await res.json();
  const hparams = data.hparams || {};
  const container = document.getElementById('hparamsList');
  
  if (Object.keys(hparams).length === 0) {
    container.innerHTML = '<div class="muted">No hyperparameters found</div>';
    return;
  }
  
  container.innerHTML = Object.entries(hparams)
    .map(([key, value]) => `<div style="margin-bottom: 4px;"><span style="color: #7ed3b2;">${key}:</span> <span style="color: #e6e8ee;">${value}</span></div>`)
    .join('');
  
  // Update summary text when opened
  const details = document.getElementById('hparamsDetails');
  details.addEventListener('toggle', () => {
    const summary = details.querySelector('summary span');
    if (details.open) {
      summary.textContent = '▼ Hide hyperparameters';
    } else {
      summary.textContent = '▶ Show hyperparameters';
    }
  });
}

function renderMetricList(metrics) {
  const container = document.getElementById('metricList');
  container.innerHTML = '';

  metrics.forEach(metric => {
    const item = document.createElement('div');
    item.className = 'metric-item';
    if (selectedMetrics.has(metric)) item.classList.add('active');
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
  renderMetricList(Object.keys(metricsData));
  updatePlot();
}

function filterMetrics(query) {
  const items = document.querySelectorAll('.metric-item');
  const q = query.toLowerCase();
  items.forEach(item => {
    item.style.display = item.textContent.toLowerCase().includes(q) ? 'flex' : 'none';
  });
}

function updateAxisLabel() {
  const label = document.getElementById('axisLabel');
  if (xAxisMode === 'time') {
    label.textContent = 'X-Axis: Training Time';
  } else if (xAxisMode === 'steps') {
    label.textContent = 'X-Axis: Environment Steps';
  } else {
    label.textContent = 'X-Axis: Episodes';
  }
}

function buildEpisodeMap() {
  const series = metricsData['train/num. episodes'];
  if (!series || !series.steps || !series.values) return [];
  const pairs = series.steps.map((step, i) => [step, series.values[i]]);
  return pairs.sort((a, b) => a[0] - b[0]);
}

function mapStepsToEpisodes(steps) {
  const map = buildEpisodeMap();
  if (!map.length) return steps;
  let idx = 0;
  let current = map[0][1];
  return steps.map(step => {
    while (idx < map.length && step >= map[idx][0]) {
      current = map[idx][1];
      idx += 1;
    }
    return current;
  });
}

async function toggleAxis() {
  const res = await fetch('/api/axis/toggle', { method: 'POST' });
  const data = await res.json();
  xAxisMode = data.x_axis_mode || 'time';
  updatePlot();
  updateAxisLabel();
}

function createPlot() {
  Plotly.newPlot('mainPlot', [], {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: 'Training Time' },
    yaxis: { title: 'Value' },
    margin: { t: 30, l: 50, r: 20, b: 40 }
  }, { responsive: true });
}

function updatePlot() {
  const colors = ['#5b8cff', '#7ed3b2', '#f5c16c', '#f28b82', '#c792ea', '#80cbc4'];
  const traces = [];
  let colorIndex = 0;

  selectedMetrics.forEach(metric => {
    const series = metricsData[metric];
    if (!series) return;
    let xData = xAxisMode === 'time' ? series.times : series.steps;
    if (xAxisMode === 'episodes') {
      xData = mapStepsToEpisodes(series.steps || []);
    }
    const yData = series.values;
    if (!xData || !xData.length) return;

    traces.push({
      x: xData,
      y: yData,
      type: 'scatter',
      mode: 'lines',
      name: metric,
      line: { width: 2, color: colors[colorIndex % colors.length] }
    });
    colorIndex++;
  });

  const xTitle = xAxisMode === 'time'
    ? 'Training Time (s)'
    : (xAxisMode === 'steps' ? 'Environment Steps' : 'Episodes');

  Plotly.react('mainPlot', traces, {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { title: 'Value' },
    margin: { t: 30, l: 50, r: 20, b: 40 },
    legend: { orientation: 'h', y: 1.1 }
  }, { responsive: true });
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function bindEvents() {
  document.getElementById('metricSearch').addEventListener('input', e => {
    filterMetrics(e.target.value);
  });
  document.getElementById('saveNotesBtn').addEventListener('click', saveNotes);
  document.getElementById('toggleAxisBtn').addEventListener('click', toggleAxis);
}

createPlot();
bindEvents();
fetchMetrics();
fetchNotes();
fetchHparams();
setInterval(fetchMetrics, 2000);
