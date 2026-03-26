// Matroid10 Pipeline Web UI

const POLL_INTERVAL = 2000; // 2 seconds

// DOM elements
const elements = {
    statusBadge: document.getElementById('status-badge'),
    startBtn: document.getElementById('start-btn'),
    stopBtn: document.getElementById('stop-btn'),
    runId: document.getElementById('run-id'),
    nextTrial: document.getElementById('next-trial'),
    lastUpdated: document.getElementById('last-updated'),
    phase1Status: document.getElementById('phase1-status'),
    phase1Progress: document.getElementById('phase1-progress'),
    phase1Detail: document.getElementById('phase1-detail'),
    phase1Container: document.getElementById('phase1-container'),
    phase2Status: document.getElementById('phase2-status'),
    phase2Progress: document.getElementById('phase2-progress'),
    phase2Detail: document.getElementById('phase2-detail'),
    phase2Container: document.getElementById('phase2-container'),
    phase3Status: document.getElementById('phase3-status'),
    phase3Progress: document.getElementById('phase3-progress'),
    phase3Detail: document.getElementById('phase3-detail'),
    phase3Container: document.getElementById('phase3-container'),
    totalProcessed: document.getElementById('total-processed'),
    uniqueFound: document.getElementById('unique-found'),
    feasible: document.getElementById('feasible'),
    infeasible: document.getElementById('infeasible'),
    accumulatedUnique: document.getElementById('accumulated-unique'),
};

// State
let pollingInterval = null;

// API calls
async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error('Failed to fetch status:', error);
        return null;
    }
}

async function startPipeline() {
    try {
        elements.startBtn.disabled = true;
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        if (!response.ok) {
            const data = await response.json();
            alert(`Failed to start: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Failed to start pipeline:', error);
        alert('Failed to start pipeline');
    } finally {
        // Re-enable will happen on next status poll
    }
}

async function stopPipeline() {
    try {
        elements.stopBtn.disabled = true;
        const response = await fetch('/api/stop', { method: 'POST' });
        if (!response.ok) {
            const data = await response.json();
            alert(`Failed to stop: ${data.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Failed to stop pipeline:', error);
        alert('Failed to stop pipeline');
    }
}

// UI update functions
function updateStatusBadge(status) {
    elements.statusBadge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    elements.statusBadge.className = 'status-badge ' + status.toLowerCase();
}

function updateButtons(status, stopRequested) {
    const isRunning = status === 'running';
    elements.startBtn.disabled = isRunning;
    elements.stopBtn.disabled = !isRunning || stopRequested;

    if (stopRequested && isRunning) {
        elements.stopBtn.textContent = 'Stopping...';
    } else {
        elements.stopBtn.textContent = 'Stop Pipeline';
    }
}

function updatePhase(phaseNum, phaseData) {
    const statusEl = elements[`phase${phaseNum}Status`];
    const progressEl = elements[`phase${phaseNum}Progress`];
    const detailEl = elements[`phase${phaseNum}Detail`];
    const containerEl = elements[`phase${phaseNum}Container`];

    const status = phaseData?.status || 'pending';
    const progress = phaseData?.progress || {};

    // Update status badge
    statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    statusEl.className = 'phase-status ' + status;

    // Update container class
    containerEl.className = 'phase ' + status;

    // Update progress bar and details based on phase
    if (status === 'completed') {
        progressEl.style.width = '100%';
    } else if (status === 'running') {
        if (phaseNum === 1 && progress.total_targets) {
            const pct = (progress.current_target / progress.total_targets) * 100;
            progressEl.style.width = pct + '%';
            detailEl.textContent = `Target ${progress.current_target}/${progress.total_targets}: ${progress.category || ''}`;
        } else {
            progressEl.style.width = '50%'; // Indeterminate
            detailEl.textContent = 'Processing...';
        }
    } else {
        progressEl.style.width = '0%';
        detailEl.textContent = '';
    }

    // Show stats for completed phases
    if (status === 'completed' && progress) {
        if (phaseNum === 1 && progress.output_records !== undefined) {
            detailEl.textContent = `Output: ${progress.output_records} records (${progress.duplicates_removed || 0} duplicates removed)`;
        } else if (phaseNum === 2 && progress.output_count !== undefined) {
            detailEl.textContent = `Output: ${progress.output_count} h-vectors`;
        } else if (phaseNum === 3) {
            const feasible = progress.feasible || 0;
            const infeasible = progress.infeasible || 0;
            detailEl.textContent = `Feasible: ${feasible}, Infeasible: ${infeasible}`;
        }
    }
}

function updateCounters(counters) {
    elements.totalProcessed.textContent = formatNumber(counters.total_processed || 0);
    elements.uniqueFound.textContent = formatNumber(counters.unique_found || 0);
    elements.feasible.textContent = formatNumber(counters.feasible || 0);
    elements.infeasible.textContent = formatNumber(counters.infeasible || 0);
}

function formatNumber(num) {
    return num.toLocaleString();
}

function formatTime(isoString) {
    if (!isoString) return '-';
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString();
    } catch {
        return isoString;
    }
}

function updateUI(data) {
    if (!data) return;

    // Status
    updateStatusBadge(data.status);
    updateButtons(data.status, data.stop_requested);

    // Info
    elements.runId.textContent = data.run_id || '-';
    elements.nextTrial.textContent = data.next_trial_index_start || 0;
    elements.lastUpdated.textContent = formatTime(data.last_updated);

    // Phases
    const phaseStatus = data.phase_status || {};
    updatePhase(1, phaseStatus.phase1);
    updatePhase(2, phaseStatus.phase2);
    updatePhase(3, phaseStatus.phase3);

    // Counters
    updateCounters(data.counters || {});

    // Accumulated
    elements.accumulatedUnique.textContent = formatNumber(data.accumulated_unique_count || 0);
}

// Polling
function startPolling() {
    if (pollingInterval) return;
    poll(); // Immediate first poll
    pollingInterval = setInterval(poll, POLL_INTERVAL);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function poll() {
    const data = await fetchStatus();
    updateUI(data);
}

// Event listeners
elements.startBtn.addEventListener('click', startPipeline);
elements.stopBtn.addEventListener('click', stopPipeline);

// Start polling on page load
document.addEventListener('DOMContentLoaded', startPolling);

// Handle visibility change (pause polling when tab is hidden)
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopPolling();
    } else {
        startPolling();
    }
});
