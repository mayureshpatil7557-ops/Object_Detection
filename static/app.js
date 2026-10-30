/* VisionAI — app.js (Flask frontend) */
'use strict';

const $ = id => document.getElementById(id);

// DOM
const webcamBtn     = $('webcamBtn');
const uploadLabel   = $('uploadLabel');
const videoUpload   = $('videoUpload');
const stopBtn       = $('stopBtn');
const confSlider    = $('confSlider');
const confValue     = $('confValue');
const toggleTrackBtn= $('toggleTrackBtn');
const statusBadge   = $('statusBadge');
const statusDot     = $('statusDot');
const statusText    = $('statusText');
const streamImg     = $('streamImg');
const idleScreen    = $('idleScreen');
const idleTitle     = $('idleTitle');
const idleSub       = $('idleSub');
const idleFeatures  = $('idleFeatures');
const modelBarWrap  = $('modelBarWrap');
const scanOverlay   = $('scanOverlay');
const metFPS        = $('metFPS');
const metObjects    = $('metObjects');
const metTracks     = $('metTracks');
const metInference  = $('metInference');
const detectionsList= $('detectionsList');
const detCount      = $('detCount');
const classesList   = $('classesList');
const totalDetected = $('totalDetected');
const totalClasses  = $('totalClasses');
const maxSimul      = $('maxSimul');
const toastEl       = $('toast');

// State
let trackingOn   = true;
let isRunning    = false;
let statsInterval= null;
let modelReady   = false;

// ── Poll stats from backend ────────────────────────────────
async function pollStats() {
  try {
    const res  = await fetch('/api/stats');
    const data = await res.json();

    // Model ready check
    if (!modelReady && data.model_ready) {
      modelReady = true;
      onModelReady();
    } else if (!modelReady) {
      idleSub.textContent = data.model_status || 'Loading…';
      return;
    }

    if (!data.running) return;

    // Metrics
    metFPS.textContent       = data.fps;
    metObjects.textContent   = data.objects;
    metTracks.textContent    = trackingOn ? data.tracks : '--';
    metInference.textContent = data.inference_ms;

    // Detections list
    detCount.textContent = data.objects;
    renderDetections(data.detections);
    renderClassSummary(data.class_counts);

    // Totals
    totalDetected.textContent = data.total_detected.toLocaleString();
    totalClasses.textContent  = data.unique_classes;
    maxSimul.textContent      = data.max_simultaneous;

  } catch (e) {
    console.warn('Stats poll error:', e);
  }
}

function onModelReady() {
  webcamBtn.disabled  = false;
  idleTitle.textContent = 'Ready to Detect';
  idleSub.textContent   = 'Start webcam or upload a video to begin';
  modelBarWrap.style.display = 'none';
  idleFeatures.style.display = 'flex';
  setStatus('ready', 'Model Ready', '#047857', '#ecfdf5', '#a7f3d0');
  showToast('✅ YOLOv8n model ready!');
}

// ── Render helpers ─────────────────────────────────────────
function renderDetections(dets) {
  if (!dets || !dets.length) {
    detectionsList.innerHTML = '<div class="no-det">No detections</div>';
    return;
  }
  detectionsList.innerHTML = dets.map(d => {
    const idTag = d.track_id != null
      ? `<span class="det-id">#${d.track_id}</span>` : '';
    return `
      <div class="det-item">
        <div class="det-color" style="background:${d.color}"></div>
        <span class="det-label">${d.label}</span>
        ${idTag}
        <span class="det-conf">${d.conf}%</span>
      </div>`;
  }).join('');
}

function renderClassSummary(counts) {
  if (!counts || !Object.keys(counts).length) {
    classesList.innerHTML = '<div class="no-det">Awaiting…</div>';
    return;
  }
  const max = Math.max(...Object.values(counts));
  classesList.innerHTML = Object.entries(counts)
    .sort((a,b) => b[1]-a[1]).slice(0,8)
    .map(([label, cnt]) => {
      const pct = Math.round((cnt/max)*100);
      const color = getLabelColor(label);
      return `
        <div class="class-row">
          <span class="class-name">${label}</span>
          <div class="class-bar-wrap">
            <div class="class-bar" style="width:${pct}%;background:${color}"></div>
          </div>
          <span class="class-cnt">${cnt}</span>
        </div>`;
    }).join('');
}

// Quick color lookup (matches Python detector.py, clean modern palette)
const CLASS_COLORS = {
  person:'#2563eb',car:'#0284c7',bicycle:'#0d9488',motorcycle:'#ea580c',
  bus:'#10b981',truck:'#d97706',cat:'#e11d48',dog:'#0284c7',bird:'#059669',
  horse:'#16a34a',sheep:'#ca8a04',cow:'#dc2626',elephant:'#0891b2',
  laptop:'#0284c7','cell phone':'#ca8a04',chair:'#2563eb',bottle:'#0284c7',
};
function getLabelColor(label) {
  return CLASS_COLORS[label] || '#2563eb';
}

// ── Controls ───────────────────────────────────────────────
webcamBtn.addEventListener('click', async () => {
  if (!modelReady) { showToast('⚠️ Model still loading…'); return; }
  const res  = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ source: 'webcam' }),
  });
  const data = await res.json();
  if (data.ok) {
    startStream();
    showToast('📷 Webcam started');
  } else {
    showToast(`❌ ${data.msg}`);
  }
});

videoUpload.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  if (!modelReady) { showToast('⚠️ Model still loading…'); return; }

  showToast('⏳ Uploading video…');
  const form = new FormData();
  form.append('video', file);

  const up = await fetch('/api/upload', { method:'POST', body: form });
  const ud = await up.json();
  if (!ud.ok) { showToast(`❌ ${ud.msg}`); return; }

  const res  = await fetch('/api/start', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ source:'file', path: ud.path }),
  });
  const data = await res.json();
  if (data.ok) {
    startStream();
    showToast('🎬 Video detection started');
  } else {
    showToast(`❌ ${data.msg}`);
  }
  e.target.value = '';
});

stopBtn.addEventListener('click', async () => {
  await fetch('/api/stop', { method:'POST' });
  stopStream();
  showToast('⏹ Stopped');
});

confSlider.addEventListener('input', () => {
  const val = confSlider.value;
  confValue.textContent = val + '%';
  fetch('/api/settings', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ confidence: val / 100 }),
  });
});

toggleTrackBtn.addEventListener('click', () => {
  trackingOn = !trackingOn;
  toggleTrackBtn.textContent = '';
  // Rebuild btn content
  const svg = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 10h14M10 3l7 7-7 7"/></svg>`;
  toggleTrackBtn.innerHTML = svg + (trackingOn ? ' Tracking ON' : ' Tracking OFF');
  toggleTrackBtn.classList.toggle('off', !trackingOn);
  fetch('/api/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ tracking: trackingOn }),
  });
  showToast(trackingOn ? '✅ Tracking enabled' : '⚪ Tracking disabled');
});

// ── Stream control ─────────────────────────────────────────
function startStream() {
  isRunning = true;
  idleScreen.classList.add('hidden');
  // Force reload of MJPEG stream
  streamImg.src = '/video_feed?' + Date.now();
  streamImg.classList.remove('hidden');
  scanOverlay.style.display = 'block';
  webcamBtn.disabled  = true;
  stopBtn.disabled    = false;
  setStatus('running', 'Detecting…', '#1d4ed8', '#eff6ff', '#bfdbfe');
}

function stopStream() {
  isRunning = false;
  streamImg.classList.add('hidden');
  streamImg.src = '';
  scanOverlay.style.display = 'none';
  idleScreen.classList.remove('hidden');
  idleTitle.textContent = 'Ready to Detect';
  idleSub.textContent   = 'Start webcam or upload a video to begin';
  webcamBtn.disabled  = false;
  stopBtn.disabled    = true;
  setStatus('ready', 'Model Ready', '#047857', '#ecfdf5', '#a7f3d0');
  // Reset sidebar
  metFPS.textContent = metObjects.textContent = metTracks.textContent = metInference.textContent = '--';
  detectionsList.innerHTML = '<div class="no-det">No detections yet</div>';
  classesList.innerHTML = '<div class="no-det">Awaiting detections…</div>';
  detCount.textContent = '0';
}

// ── Status badge ───────────────────────────────────────────
function setStatus(state, text, color, bg = '#ecfdf5', border = '#a7f3d0') {
  statusText.textContent = text;
  statusBadge.style.cssText = `background:${bg};border:1px solid ${border};color:${color}`;
  statusDot.style.background = color;
}

// ── Toast ──────────────────────────────────────────────────
let _tt;
function showToast(msg, dur=2800) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(_tt);
  _tt = setTimeout(() => toastEl.classList.remove('show'), dur);
}

// ── Start polling ──────────────────────────────────────────
setStatus('loading', 'Loading Model…', '#d97706', '#fffbeb', '#fde68a');
statsInterval = setInterval(pollStats, 400);
