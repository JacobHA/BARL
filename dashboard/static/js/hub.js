let allRuns = [];
let displayCount = 8;
let pollInterval = null;

const tbBrowseBtn = document.getElementById('tbBrowseBtn');
const tbDirPicker = document.getElementById('tbDirPicker');
const tbPanel = document.getElementById('tbPanel');
const tbFileList = document.getElementById('tbFileList');
const tbSummary = document.getElementById('tbSummary');
const tbUploadBtn = document.getElementById('tbUploadBtn');

let selectedFiles = [];

async function fetchRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  allRuns = data.runs || [];
  console.log('Fetched runs:', allRuns.length, 'first run status:', allRuns[0]?.status);
  renderRuns();
  adjustPollingRate();
}

function adjustPollingRate() {
  // Check if any runs are active
  const hasActiveRuns = allRuns.some(run => run.status === 'running');
  console.log('Adjust polling - has active runs:', hasActiveRuns);
  
  // Clear existing interval
  if (pollInterval) {
    clearInterval(pollInterval);
    pollInterval = null;
  }
  
  // Only poll if there are active runs
  if (hasActiveRuns) {
    console.log('Starting polling (5s interval)');
    pollInterval = setInterval(fetchRuns, 5000);
  } else {
    console.log('No active runs - polling stopped');
  }
}

function renderRuns() {
  const container = document.getElementById('runs');
  const loadMoreContainer = document.getElementById('loadMoreContainer');
  container.innerHTML = '';

  if (!allRuns.length) {
    container.innerHTML = '<div class="muted">No runs found in logs/</div>';
    loadMoreContainer.style.display = 'none';
    return;
  }

  const runsToShow = allRuns.slice(0, displayCount);
  
  runsToShow.forEach(run => {
    const card = document.createElement('a');
    card.className = 'run-card';
    card.href = `/run/${run.id}`;

    const title = document.createElement('div');
    title.className = 'run-title';
    const displayName = run.algo_name && run.env_str 
      ? `${run.algo_name} (${run.env_str}): ${run.id}` 
      : run.algo_name 
        ? `${run.algo_name}: ${run.id}` 
        : run.id;
    
    // Add status indicator
    const statusIndicator = run.status === 'running' ? ' 🟢' : '';
    title.textContent = displayName + statusIndicator;

    card.appendChild(title);
    container.appendChild(card);
  });
  
  // Show/hide load more button
  if (allRuns.length > displayCount) {
    loadMoreContainer.style.display = 'block';
  } else {
    loadMoreContainer.style.display = 'none';
  }
}

function filterRuns(query) {
  const q = query.toLowerCase();
  const cards = document.querySelectorAll('.run-card');
  cards.forEach(card => {
    const title = card.querySelector('.run-title')?.textContent?.toLowerCase() || '';
    card.style.display = title.includes(q) ? 'flex' : 'none';
  });
}

function formatTime(epochSeconds) {
  try {
    const d = new Date(epochSeconds * 1000);
    return d.toLocaleString();
  } catch (e) {
    return 'unknown';
  }
}

document.getElementById('runSearch').addEventListener('input', e => {
  filterRuns(e.target.value);
});

document.getElementById('loadMoreBtn').addEventListener('click', () => {
  displayCount = allRuns.length;
  renderRuns();
});

// Initial fetch and start adaptive polling
fetchRuns();

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'unknown size';
  const units = ['B', 'KB', 'MB', 'GB'];
  let idx = 0;
  let value = bytes;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(value < 10 && idx > 0 ? 1 : 0)} ${units[idx]}`;
}

function resolveRunIdFromPath(relativePath) {
  if (!relativePath || !allRuns.length) return null;
  const normalized = relativePath.replace(/\\/g, '/');
  const segments = normalized.split('/').filter(Boolean);
  const runIds = new Set(allRuns.map(run => run.id));

  // Direct match on any segment
  for (const segment of segments) {
    if (runIds.has(segment)) {
      return segment;
    }
  }

  // Fallback: match run id by segment substring (handles extra prefixes like "logs/")
  for (const runId of runIds) {
    if (segments.some(segment => segment.includes(runId))) {
      return runId;
    }
  }

  return null;
}

function renderTbFiles(files) {
  if (!tbPanel || !tbFileList || !tbSummary) return;

  tbPanel.style.display = 'block';
  tbFileList.innerHTML = '';

  if (!files.length) {
    tbSummary.textContent = 'No TensorBoard files found';
    tbFileList.innerHTML = '<div class="muted">Try selecting a logs directory that contains events.out.tfevents files.</div>';
    if (tbUploadBtn) tbUploadBtn.disabled = true;
    return;
  }

  tbSummary.textContent = `${files.length} TensorBoard file${files.length === 1 ? '' : 's'} found`;
  if (tbUploadBtn) tbUploadBtn.disabled = !selectedFiles.length;
  files.forEach(file => {
    const item = document.createElement('div');
    item.className = 'file-item';

    const path = document.createElement('div');
    path.className = 'file-path';
    path.textContent = file.webkitRelativePath || file.name;

    const meta = document.createElement('div');
    meta.className = 'file-meta';
    meta.textContent = formatBytes(file.size);

    const actions = document.createElement('div');
    actions.className = 'file-actions';

    const runId = resolveRunIdFromPath(file.webkitRelativePath || file.name);
    if (runId) {
      const openLink = document.createElement('a');
      openLink.className = 'btn btn-small';
      openLink.href = `/run/${runId}`;
      openLink.textContent = 'Open as run';
      actions.appendChild(openLink);
    } else {
      const hint = document.createElement('span');
      hint.className = 'muted';
      hint.textContent = 'Not in logs/';
      actions.appendChild(hint);
    }

    item.appendChild(path);
    item.appendChild(meta);
    item.appendChild(actions);
    tbFileList.appendChild(item);
  });
}

if (tbBrowseBtn && tbDirPicker) {
  tbBrowseBtn.addEventListener('click', () => {
    tbDirPicker.click();
  });

  tbDirPicker.addEventListener('change', async () => {
    const files = Array.from(tbDirPicker.files || []);
    selectedFiles = files;
    if (!allRuns.length) {
      await fetchRuns();
    }
    const tbFiles = files.filter(file => /tfevents|events\.out\.tfevents/i.test(file.name));
    renderTbFiles(tbFiles);
  });
}

if (tbUploadBtn) {
  tbUploadBtn.addEventListener('click', async () => {
    if (!selectedFiles.length) return;
    tbUploadBtn.disabled = true;
    tbUploadBtn.textContent = 'Importing...';

    try {
      const formData = new FormData();
      selectedFiles.forEach(file => {
        const relPath = file.webkitRelativePath || file.name;
        formData.append('files', file, relPath);
      });

      console.log('Uploading files:', selectedFiles.map(f => f.webkitRelativePath || f.name));
      const res = await fetch('/api/upload_logs', { method: 'POST', body: formData });
      const data = await res.json();
      console.log('Upload response:', data);

      if (!res.ok || !data.success) {
        console.error('Upload failed:', data.error || 'unknown error');
        tbUploadBtn.textContent = 'Import failed';
        return;
      }

      await fetchRuns();
      renderRuns();

      if (data.run_ids && data.run_ids.length === 1) {
        window.location.href = `/run/${data.run_ids[0]}`;
        return;
      }

      tbUploadBtn.textContent = 'Imported';
    } catch (err) {
      console.error('Upload error:', err);
      tbUploadBtn.textContent = 'Import failed';
    } finally {
      setTimeout(() => {
        tbUploadBtn.textContent = 'Import to dashboard';
        tbUploadBtn.disabled = !selectedFiles.length;
      }, 1500);
    }
  });
}
