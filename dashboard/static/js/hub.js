let allRuns = [];
let displayCount = 8;

async function fetchRuns() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  allRuns = data.runs || [];
  renderRuns();
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

  const currentRun = window.__CURRENT_RUN__ || '';
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
    title.textContent = run.id === currentRun ? `${displayName} (current)` : displayName;

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

fetchRuns();
setInterval(fetchRuns, 5000);
