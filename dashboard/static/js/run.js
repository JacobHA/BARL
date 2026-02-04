let selectedMetrics = new Set(['eval/avg_reward']);
let selectedBufferPlots = new Set();
let selectedNetworkLayer = null;
let selectedNetworkMetrics = new Set(['weight_norm', 'grad_norm']);
let metricsData = {};
let lastMetricsListKey = '';
let lastBufferStats = null;
let xAxisMode = 'time';
let lastDataUpdateTime = Date.now();
let metricsInterval = null;
let videosInterval = null;
let statusInterval = null;
let isRunActive = false; // Start as false, will be set by status check
let trackUpdates = true; // When false, enable zoom preservation
let lastVideoKey = '';
let networkLayers = [];
let networkMetricsData = {};
let categorizedMetrics = { training: [], buffer: [], network: [] };
let networkViewMode = 'parameter';
let videosCollapsed = false;

const runId = window.__RUN_ID__;
const bufferPlotOptions = [
  { id: 'buffer_size', label: 'Buffer Size' },
  { id: 'terminated_fraction', label: 'Terminated Fraction' },
  { id: 'reward_histogram', label: 'Reward Histogram' }
];

async function checkRunStatus() {
  try {
    const res = await fetch(`/api/runs/${runId}/status`);
    const data = await res.json();
    const wasActive = isRunActive;
    isRunActive = data.status === 'running';
    
    // If status changed, adjust polling
    if (wasActive !== isRunActive) {
      adjustPolling();
    }
  } catch (e) {
    console.error('Failed to check run status:', e);
  }
}

function adjustPolling() {
  // Clear existing intervals
  if (metricsInterval) {
    clearInterval(metricsInterval);
    metricsInterval = null;
  }
  if (videosInterval) {
    clearInterval(videosInterval);
    videosInterval = null;
  }
  
  // Only start polling if run is active
  if (isRunActive) {
    console.log('Run is active. Starting polling.');
    metricsInterval = setInterval(fetchMetrics, 2000);
    videosInterval = setInterval(fetchVideos, 10000);
  } else {
    console.log('Run is not active. Polling stopped.');
  }
}

async function fetchMetrics() {
  const res = await fetch(`/api/runs/${runId}/metrics`);
  const data = await res.json();
  const incomingData = data.data || {};
  const oldDataStr = JSON.stringify(metricsData);
  const newDataStr = JSON.stringify(incomingData);
  
  // Check if data changed
  if (oldDataStr !== newDataStr) {
    lastDataUpdateTime = Date.now();
  }
  
  // Only update visuals if we actually have data to show
  if (Object.keys(incomingData).length > 0) {
    metricsData = incomingData;
    xAxisMode = data.x_axis_mode || 'time';

    // Use categorized metrics from API (Training, Buffer, Network)
    const categorized = data.categorized_metrics || {
      training: [],
      buffer: [],
      network: []
    };
    categorizedMetrics = categorized;

    // Build network layers/metrics from categorized.network
    const networkMetrics = {};
    networkMetricsData = {};
    for (const metric of categorized.network || []) {
      const parts = metric.split('/');
      if (parts.length < 3) continue;
      const metricType = parts[parts.length - 1];
      const fullLayer = parts.slice(0, -1).join('/');

      if (!networkMetrics[fullLayer]) {
        networkMetrics[fullLayer] = new Set();
      }
      networkMetrics[fullLayer].add(metricType);

      if (!networkMetricsData[fullLayer]) {
        networkMetricsData[fullLayer] = {};
      }
      networkMetricsData[fullLayer][metricType] = incomingData[metric];
    }
    networkLayers = Object.keys(networkMetrics).sort();

    // Render metrics list with Training stats only
    const metricsKey = JSON.stringify(categorized.training || []);
    if (metricsKey !== lastMetricsListKey) {
      renderMetricList(categorized.training || []);
      lastMetricsListKey = metricsKey;
    }

    renderNetworkLayerList(networkMetrics);
    updatePlot();
    updateAxisLabel();
  }
  // Also fetch and plot buffer stats
  if (selectedBufferPlots.size > 0) {
    fetchBufferStats();
  }
  if (selectedNetworkLayer) {
    updateNetworkPlot();
  }
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

async function fetchVideos() {
  try {
    const res = await fetch(`/api/runs/${runId}/videos`);
    const data = await res.json();
    const videos = data.videos || [];
    const key = videos.map(v => v.name + v.mtime).join('|');
    if (key !== lastVideoKey) {
      renderVideos(videos);
      lastVideoKey = key;
    }
  } catch (error) {
    console.error('Error fetching videos:', error);
  }
}

function renderVideos(videos) {
  const container = document.getElementById('videoList');
  const emptyState = document.getElementById('videoEmptyState');
  if (!container || !emptyState) return;

  container.innerHTML = '';
  if (!videos.length) {
    emptyState.style.display = 'block';
    return;
  }
  emptyState.style.display = 'none';

  videos.forEach(video => {
    const card = document.createElement('div');
    card.className = 'video-card';

    const title = document.createElement('div');
    title.className = 'video-title';
    title.textContent = video.name;

    const videoEl = document.createElement('video');
    videoEl.controls = true;
    videoEl.preload = 'metadata';
    videoEl.src = video.url;

    const meta = document.createElement('div');
    meta.className = 'video-meta';
    const size = document.createElement('span');
    size.textContent = formatBytes(video.size || 0);
    const time = document.createElement('span');
    if (video.mtime) {
      const date = new Date(video.mtime * 1000);
      time.textContent = date.toLocaleString();
    }
    meta.appendChild(size);
    if (video.reward !== null && video.reward !== undefined) {
      const reward = document.createElement('span');
      reward.textContent = `Reward: ${Number(video.reward).toFixed(2)}`;
      meta.appendChild(reward);
    }
    meta.appendChild(time);

    const actions = document.createElement('div');
    actions.className = 'video-actions';
    const link = document.createElement('a');
    link.href = video.url;
    link.textContent = 'Open';
    link.className = 'btn btn-small';
    link.target = '_blank';
    actions.appendChild(link);

    card.appendChild(title);
    card.appendChild(videoEl);
    card.appendChild(meta);
    card.appendChild(actions);
    container.appendChild(card);
  });
}

async function fetchBufferStats() {
  const res = await fetch(`/api/runs/${runId}/buffer_stats`);
  const data = await res.json();
  const hasData = Object.keys(data || {}).length > 0;
  if (!hasData && lastBufferStats) {
    plotBufferStats(lastBufferStats);
    return;
  }
  if (hasData) {
    lastBufferStats = data;
    plotBufferStats(data);
  }
}

function renderBufferPlotList() {
  const container = document.getElementById('bufferPlotList');
  container.innerHTML = '';

  bufferPlotOptions.forEach(option => {
    const item = document.createElement('div');
    item.className = 'metric-item';
    if (selectedBufferPlots.has(option.id)) item.classList.add('active');
    item.textContent = option.label;
    item.onclick = () => toggleBufferPlot(option.id);
    container.appendChild(item);
  });
}

function toggleBufferPlot(plotId) {
  if (selectedBufferPlots.has(plotId)) {
    selectedBufferPlots.delete(plotId);
  } else {
    selectedBufferPlots.add(plotId);
  }
  renderBufferPlotList();
  updateBufferPanelVisibility();
  if (selectedBufferPlots.size > 0) {
    fetchBufferStats();
  }
}

function updateBufferPanelVisibility() {
  const panel = document.getElementById('bufferPanel');
  if (!panel) return;
  panel.style.display = selectedBufferPlots.size > 0 ? 'block' : 'none';

  if (selectedBufferPlots.size === 0) {
    const bufferPlot = document.getElementById('bufferPlot');
    if (bufferPlot) {
      bufferPlot.innerHTML = '';
    }
    const histPlot = document.getElementById('rewardHistPlot');
    if (histPlot && histPlot.parentElement) {
      histPlot.parentElement.remove();
    }
  }
}

function plotBufferStats(data) {
  if (selectedBufferPlots.size === 0) {
    updateBufferPanelVisibility();
    return;
  }

  const nStored = data.n_stored || {};
  const terminatedFrac = data.terminated_fraction || {};
  const rewardHist = data.reward_histogram || {};
  
  const traces = [];
  
  if (selectedBufferPlots.has('buffer_size') && nStored.steps && nStored.steps.length > 0) {
    const xData = xAxisMode === 'time' ? nStored.times : nStored.steps;
    traces.push({
      x: xData,
      y: nStored.values,
      type: 'scatter',
      mode: 'lines',
      name: 'Buffer Size',
      line: { width: 2, color: '#5b8cff' },
      yaxis: 'y'
    });
  }
  
  if (selectedBufferPlots.has('terminated_fraction') && terminatedFrac.steps && terminatedFrac.steps.length > 0) {
    const xData = xAxisMode === 'time' ? terminatedFrac.times : terminatedFrac.steps;
    traces.push({
      x: xData,
      y: terminatedFrac.values,
      type: 'scatter',
      mode: 'lines',
      name: 'Terminated Fraction',
      line: { width: 2, color: '#f5c16c' },
      yaxis: 'y2'
    });
  }
  
  const wantsHistogram = selectedBufferPlots.has('reward_histogram');
  if (traces.length === 0 && (!wantsHistogram || Object.keys(rewardHist).length === 0)) {
    // Avoid clearing the plot to prevent flicker if data briefly goes empty
    return;
  }
  
  const xTitle = xAxisMode === 'time' ? 'Training Time (s)' : 'Environment Steps';
  
  // Get current axis ranges if they exist (preserve zoom)
  const bufferPlotDiv = document.getElementById('bufferPlot');
  const preserveRange = !trackUpdates && bufferPlotDiv.layout && bufferPlotDiv.layout.xaxis && bufferPlotDiv.layout.xaxis.range;
  
  const layout = {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { 
      title: 'Buffer Size',
      titlefont: { color: '#5b8cff' },
      tickfont: { color: '#5b8cff' }
    },
    yaxis2: {
      title: 'Terminated Fraction',
      titlefont: { color: '#f5c16c' },
      tickfont: { color: '#f5c16c' },
      overlaying: 'y',
      side: 'right'
    },
    margin: { t: 20, l: 50, r: 50, b: 40 },
    legend: { orientation: 'h', y: 1.15 }
  };
  
  // Preserve zoom/pan if user has interacted and tracking is off
  if (preserveRange) {
    layout.xaxis.range = bufferPlotDiv.layout.xaxis.range;
    layout.yaxis.range = bufferPlotDiv.layout.yaxis.range;
    if (bufferPlotDiv.layout.yaxis2 && bufferPlotDiv.layout.yaxis2.range) {
      layout.yaxis2.range = bufferPlotDiv.layout.yaxis2.range;
    } else {
      layout.yaxis2.range = [0, 1];
    }
    layout.xaxis.autorange = false;
    layout.yaxis.autorange = false;
    layout.yaxis2.autorange = false;
  } else {
    layout.yaxis2.range = [0, 1];
  }
  
  Plotly.react('bufferPlot', traces, layout, { responsive: true });
  
  // Plot reward histogram if available
  if (wantsHistogram && Object.keys(rewardHist).length > 0) {
    console.log('Reward histogram data:', rewardHist);
    
    // Keep original keys and sort by numeric value
    const sortedEntries = Object.entries(rewardHist).sort((a, b) => Number(a[0]) - Number(b[0]));
    const rewards = sortedEntries.map(e => Number(e[0]));
    const counts = sortedEntries.map(e => e[1]);
    
    console.log('Processed rewards:', rewards);
    console.log('Processed counts:', counts);
    
    let barWidth = 0.5;
    if (rewards.length > 1) {
      const diffs = rewards.slice(1).map((v, i) => v - rewards[i]).filter(d => d > 0);
      if (diffs.length > 0) {
        barWidth = Math.min(...diffs) * 0.9;
      }
    }

    const histTrace = [{
      x: rewards,
      y: counts,
      type: 'bar',
      name: 'Reward Histogram',
      marker: { color: '#7ed3b2' },
      width: barWidth
    }];
    

    console.log('Plotting histogram with trace:', histTrace);
    
    // Create element first if needed
    let histPlotDiv = document.getElementById('rewardHistPlot');
    if (!histPlotDiv) {
      const panel = document.getElementById('bufferPlot').closest('.panel');
      const histSection = document.createElement('div');
      histSection.innerHTML = '<div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #2d3748;"><h3 style="margin-bottom: 10px; font-size: 14px; color: #e6e8ee;">Reward Distribution (from Buffer)</h3><div id="rewardHistPlot" class="plot" style="height: 250px;"></div></div>';
      panel.appendChild(histSection);
      histPlotDiv = document.getElementById('rewardHistPlot');
    }
    
    // Get current axis ranges if they exist (preserve zoom)
    const preserveRange = !trackUpdates && histPlotDiv && histPlotDiv.layout && histPlotDiv.layout.xaxis && histPlotDiv.layout.xaxis.range;
    
    const layout = {
      paper_bgcolor: '#151821',
      plot_bgcolor: '#151821',
      font: { color: '#e6e8ee' },
      xaxis: { 
        title: 'Reward Value',
        tickformat: '.1f'
      },
      yaxis: { 
        title: 'Count',
        type: 'log'
      },
      margin: { t: 10, l: 60, r: 20, b: 40 },
      bargap: 0.2
    };
    
    // Preserve zoom/pan if user has interacted and tracking is off
    if (preserveRange) {
      layout.xaxis.range = histPlotDiv.layout.xaxis.range;
      layout.yaxis.range = histPlotDiv.layout.yaxis.range;
      layout.xaxis.autorange = false;
      layout.yaxis.autorange = false;
    }
    
    Plotly.react('rewardHistPlot', histTrace, layout, { responsive: true });
    
    console.log('Histogram plot rendered');
  } else {
    console.log('No reward histogram data available');
  }
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
  renderMetricList(categorizedMetrics.training || []);
  updatePlot();
}

function filterMetrics(query) {
  const items = document.querySelectorAll('.metric-item');
  const q = query.toLowerCase();
  items.forEach(item => {
    item.style.display = item.textContent.toLowerCase().includes(q) ? 'flex' : 'none';
  });
}

function renderNetworkLayerList(networkMetrics) {
  const container = document.getElementById('networkLayerList');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (networkLayers.length === 0) {
    container.innerHTML = '<div class="muted" style="padding: 8px;">No network stats available</div>';
    return;
  }
  
  networkLayers.forEach(layerName => {
    const item = document.createElement('div');
    item.className = 'network-layer-item';
    if (selectedNetworkLayer === layerName) {
      item.classList.add('active');
    }
    
    // Extract readable layer name
    const parts = layerName.split('/');
    const displayName = parts[parts.length - 1] || layerName;
    const prefix = parts[0];
    
    const nameDiv = document.createElement('div');
    nameDiv.className = 'layer-name';
    nameDiv.textContent = displayName;
    
    const prefixDiv = document.createElement('div');
    prefixDiv.className = 'layer-prefix';
    prefixDiv.textContent = prefix;
    
    item.appendChild(nameDiv);
    item.appendChild(prefixDiv);
    item.onclick = () => selectNetworkLayer(layerName);
    container.appendChild(item);
  });
}

function selectNetworkLayer(layerName) {
  selectedNetworkLayer = layerName;
  renderNetworkLayerList({});
  updateNetworkPlot();
  updateNetworkPanelVisibility();
}

function updateNetworkPanelVisibility() {
  const panel = document.getElementById('networkPanel');
  const otherPanel = document.getElementById('networkOtherPanel');
  if (!panel) return;
  panel.style.display = selectedNetworkLayer ? 'block' : 'none';
  if (otherPanel) {
    otherPanel.style.display = (selectedNetworkLayer && networkViewMode === 'network') ? 'block' : 'none';
  }
}

function updateNetworkPlot() {
  if (!selectedNetworkLayer || !networkMetricsData[selectedNetworkLayer]) {
    return;
  }
  
  const layerData = networkMetricsData[selectedNetworkLayer];
  const metricTypes = Object.keys(layerData);
  
  // Group metrics by type (weight vs grad)
  const weightMetrics = metricTypes.filter(m => 
    m.startsWith('weight_') || m === 'stable_rank' || m === 'top_eigenvalue'
  );
  const gradMetrics = metricTypes.filter(m => m.startsWith('grad_'));
  const otherMetrics = metricTypes.filter(m =>
    !m.startsWith('weight_') &&
    !m.startsWith('grad_') &&
    m !== 'stable_rank' &&
    m !== 'top_eigenvalue' &&
    m !== 'weight_spectral_norm'
  );
  
  const colors = ['#5b8cff', '#7ed3b2', '#f5c16c', '#f28b82', '#c792ea', '#80cbc4'];
  const traces = [];
  let colorIdx = 0;
  
  // Plot weight metrics
  weightMetrics.forEach(metric => {
    const data = layerData[metric];
    if (!data || !data.steps || data.steps.length === 0) return;
    
    const xData = xAxisMode === 'time' ? data.times : data.steps;
    traces.push({
      x: xData,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: metric,
      line: { width: 2, color: colors[colorIdx % colors.length] },
      yaxis: 'y'
    });
    colorIdx++;
  });
  
  // Build gradient traces
  const gradTraces = [];
  gradMetrics.forEach(metric => {
    const data = layerData[metric];
    if (!data || !data.steps || data.steps.length === 0) return;
    
    const xData = xAxisMode === 'time' ? data.times : data.steps;
    gradTraces.push({
      x: xData,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: metric,
      line: { width: 2, color: colors[colorIdx % colors.length] }
    });
    colorIdx++;
  });

  // Build other network stats traces
  const otherTraces = [];
  otherMetrics.forEach(metric => {
    const data = layerData[metric];
    if (!data || !data.steps || data.steps.length === 0) return;

    const xData = xAxisMode === 'time' ? data.times : data.steps;
    otherTraces.push({
      x: xData,
      y: data.values,
      type: 'scatter',
      mode: 'lines',
      name: metric,
      line: { width: 2, color: colors[colorIdx % colors.length] }
    });
    colorIdx++;
  });
  
  if (traces.length === 0 && gradTraces.length === 0 && otherTraces.length === 0) return;
  
  const xTitle = xAxisMode === 'time' ? 'Training Time (s)' : 'Environment Steps';
  
  const weightsDiv = document.getElementById('networkWeightsPlot');
  const gradsDiv = document.getElementById('networkGradientsPlot');
  const otherDiv = document.getElementById('networkOtherPlot');
  const otherPanel = document.getElementById('networkOtherPanel');
  const otherLabel = document.getElementById('networkOtherLabel');
  const preserveWeights = !trackUpdates && weightsDiv.layout && weightsDiv.layout.xaxis && weightsDiv.layout.xaxis.range;
  const preserveGrads = !trackUpdates && gradsDiv.layout && gradsDiv.layout.xaxis && gradsDiv.layout.xaxis.range;
  const preserveOther = !trackUpdates && otherDiv && otherDiv.layout && otherDiv.layout.xaxis && otherDiv.layout.xaxis.range;
  
  const weightsLayout = {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { title: 'Weight Stats' },
    margin: { t: 20, l: 60, r: 20, b: 40 },
    legend: { orientation: 'h', y: 1.15 }
  };
  const gradsLayout = {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { title: 'Gradient Stats' },
    margin: { t: 20, l: 60, r: 20, b: 40 },
    legend: { orientation: 'h', y: 1.15 }
  };
  const otherLayout = {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { title: 'Network Stats' },
    margin: { t: 20, l: 60, r: 20, b: 40 },
    legend: { orientation: 'h', y: 1.15 }
  };
  
  if (preserveWeights) {
    weightsLayout.xaxis.range = weightsDiv.layout.xaxis.range;
    weightsLayout.yaxis.range = weightsDiv.layout.yaxis.range;
    weightsLayout.xaxis.autorange = false;
    weightsLayout.yaxis.autorange = false;
  }
  if (preserveGrads) {
    gradsLayout.xaxis.range = gradsDiv.layout.xaxis.range;
    gradsLayout.yaxis.range = gradsDiv.layout.yaxis.range;
    gradsLayout.xaxis.autorange = false;
    gradsLayout.yaxis.autorange = false;
  }

  if (preserveOther) {
    otherLayout.xaxis.range = otherDiv.layout.xaxis.range;
    otherLayout.yaxis.range = otherDiv.layout.yaxis.range;
    otherLayout.xaxis.autorange = false;
    otherLayout.yaxis.autorange = false;
  }
  
  const label = document.getElementById('networkLayerLabel');
  if (label) {
    label.textContent = `Layer: ${selectedNetworkLayer}`;
  }
  
  Plotly.react('networkWeightsPlot', traces, weightsLayout, { responsive: true });
  Plotly.react('networkGradientsPlot', gradTraces, gradsLayout, { responsive: true });

  if (otherDiv && otherPanel && networkViewMode === 'network') {
    if (otherTraces.length > 0) {
      otherPanel.style.display = 'block';
      if (otherLabel) otherLabel.textContent = selectedNetworkLayer;
      Plotly.react('networkOtherPlot', otherTraces, otherLayout, { responsive: true });
    } else {
      otherPanel.style.display = 'none';
    }
  }
}

const networkViewSelect = document.getElementById('networkViewSelect');
if (networkViewSelect) {
  networkViewSelect.addEventListener('change', (e) => {
    networkViewMode = e.target.value === 'network' ? 'network' : 'parameter';
    updateNetworkPanelVisibility();
    updateNetworkPlot();
  });
}

function filterNetworkLayers(query) {
  const items = document.querySelectorAll('.network-layer-item');
  const q = query.toLowerCase();
  items.forEach(item => {
    const text = item.textContent.toLowerCase();
    item.style.display = text.includes(q) ? 'flex' : 'none';
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

  // Get current axis ranges if they exist (preserve zoom)
  const mainPlotDiv = document.getElementById('mainPlot');
  const preserveRange = !trackUpdates && mainPlotDiv.layout && mainPlotDiv.layout.xaxis && mainPlotDiv.layout.xaxis.range;
  
  const layout = {
    paper_bgcolor: '#151821',
    plot_bgcolor: '#151821',
    font: { color: '#e6e8ee' },
    xaxis: { title: xTitle },
    yaxis: { title: 'Value' },
    margin: { t: 30, l: 50, r: 20, b: 40 },
    legend: { orientation: 'h', y: 1.1 }
  };
  
  // Preserve zoom/pan if user has interacted
  if (preserveRange) {
    layout.xaxis.range = mainPlotDiv.layout.xaxis.range;
    layout.yaxis.range = mainPlotDiv.layout.yaxis.range;
    layout.xaxis.autorange = false;
    layout.yaxis.autorange = false;
  }

  Plotly.react('mainPlot', traces, layout, { responsive: true });
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
  const refreshVideosBtn = document.getElementById('refreshVideosBtn');
  if (refreshVideosBtn) {
    refreshVideosBtn.addEventListener('click', fetchVideos);
  }
  const toggleVideosBtn = document.getElementById('toggleVideosBtn');
  if (toggleVideosBtn) {
    toggleVideosBtn.addEventListener('click', toggleVideosPanel);
  }
  fetchMemoryBreakdown();
  const networkSearch = document.getElementById('networkSearch');
  if (networkSearch) {
    networkSearch.addEventListener('input', e => {
      filterNetworkLayers(e.target.value);
    });
  }
  const btn = document.getElementById('trackUpdatesBtn');
  btn.textContent = trackUpdates ? 'Track Updates: ON' : 'Track Updates: OFF';
  btn.style.backgroundColor = trackUpdates ? '#7ed3b2' : '';
  btn.addEventListener('click', () => {
    trackUpdates = !trackUpdates;
    btn.textContent = trackUpdates ? 'Track Updates: ON' : 'Track Updates: OFF';
    btn.style.backgroundColor = trackUpdates ? '#7ed3b2' : '';
  });
}

function toggleVideosPanel() {
  videosCollapsed = !videosCollapsed;
  const list = document.getElementById('videoList');
  const empty = document.getElementById('videoEmptyState');
  const btn = document.getElementById('toggleVideosBtn');
  if (!list || !btn) return;
  list.style.display = videosCollapsed ? 'none' : 'grid';
  if (empty) {
    empty.style.display = videosCollapsed ? 'none' : empty.style.display;
  }
  btn.textContent = videosCollapsed ? 'Expand' : 'Collapse';
  if (!videosCollapsed) {
    fetchVideos();
  }
}

async function fetchMemoryBreakdown() {
  try {
    const res = await fetch(`/api/runs/${runId}/storage`);
    const data = await res.json();
    renderMemoryBreakdown(data);
  } catch (error) {
    console.error('Error fetching memory breakdown:', error);
  }
}

function renderMemoryBreakdown(data) {
  const totalEl = document.getElementById('memoryTotal');
  const breakdownEl = document.getElementById('memoryBreakdown');
  if (!totalEl || !breakdownEl) return;

  const total = data.total || 0;
  totalEl.textContent = `Total: ${formatBytes(total)}`;

  const breakdown = data.breakdown || {};
  const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
  breakdownEl.innerHTML = entries
    .map(([key, value]) => {
      const pct = total > 0 ? ((value / total) * 100).toFixed(1) : '0.0';
      return `
        <div class="memory-row">
          <span class="memory-label">${key}</span>
          <span class="memory-value">${formatBytes(value)} (${pct}%)</span>
        </div>
      `;
    })
    .join('');
}


createPlot();
bindEvents();
renderBufferPlotList();
updateBufferPanelVisibility();
// Initial load with status check
checkRunStatus().then(() => {
  fetchMetrics();
  fetchNotes();
  fetchHparams();
  fetchVideos();
  adjustPolling();
});

// Check status every 5 seconds to detect when run starts/stops
statusInterval = setInterval(checkRunStatus, 5000);
