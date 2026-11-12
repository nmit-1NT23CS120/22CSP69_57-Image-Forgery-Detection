// ForensicAI — batch.js

const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const zipInput      = document.getElementById('zipInput');
const pickImagesBtn = document.getElementById('pickImagesBtn');
const pickZipBtn    = document.getElementById('pickZipBtn');

const queueSection  = document.getElementById('queueSection');
const queueList     = document.getElementById('queueList');
const queueTitle    = document.getElementById('queueTitle');
const clearQueueBtn = document.getElementById('clearQueueBtn');
const runBatchBtn   = document.getElementById('runBatchBtn');

const loadingSection = document.getElementById('loadingSection');
const loadingText    = document.getElementById('loadingText');
const summaryBar      = document.getElementById('summaryBar');
const resultsSection  = document.getElementById('resultsSection');
const resultsGrid     = document.getElementById('resultsGrid');
const skipNote        = document.getElementById('skipNote');

const modalOverlay  = document.getElementById('modalOverlay');
const modalContent  = document.getElementById('modalContent');
const modalCloseBtn = document.getElementById('modalCloseBtn');

const exportCsvBtn = document.getElementById('exportCsvBtn');
const printBtn     = document.getElementById('printBtn');

let imageQueue   = [];   // array of File
let zipQueue     = null; // single File or null
let currentBatch = null; // { batch_id, results: [...] }

// ── Picking files ───────────────────────────────────────────
pickImagesBtn.addEventListener('click', () => fileInput.click());
pickZipBtn.addEventListener('click', () => zipInput.click());
fileInput.addEventListener('change', e => addImages(Array.from(e.target.files)));
zipInput.addEventListener('change', e => { if (e.target.files[0]) setZip(e.target.files[0]); });

// ── Drag & drop (accepts a mix of images and/or one zip) ────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const files = Array.from(e.dataTransfer.files);
  const zips  = files.filter(f => f.name.toLowerCase().endsWith('.zip'));
  const imgs  = files.filter(f => f.type.startsWith('image/'));
  if (zips.length) setZip(zips[0]);
  if (imgs.length) addImages(imgs);
});
dropZone.addEventListener('click', e => {
  if (e.target.tagName !== 'BUTTON') fileInput.click();
});

function addImages(files) {
  imageQueue = imageQueue.concat(files);
  renderQueue();
}
function setZip(file) {
  zipQueue = file;
  renderQueue();
}
function removeImage(idx) {
  imageQueue.splice(idx, 1);
  renderQueue();
}
function removeZip() {
  zipQueue = null;
  renderQueue();
}
clearQueueBtn.addEventListener('click', () => {
  imageQueue = [];
  zipQueue = null;
  fileInput.value = '';
  zipInput.value = '';
  renderQueue();
});

function renderQueue() {
  const total = imageQueue.length + (zipQueue ? 1 : 0);
  if (total === 0) {
    queueSection.style.display = 'none';
    return;
  }
  queueSection.style.display = 'block';
  queueTitle.textContent = imageQueue.length + (zipQueue ? ' image(s) + 1 zip queued' : ' image(s) queued');

  let html = '';
  if (zipQueue) {
    html += '<div class="queue-item">🗂️ ' + escapeHtml(zipQueue.name) +
      ' <span class="rm" data-zip="1">✕</span></div>';
  }
  imageQueue.forEach((f, i) => {
    html += '<div class="queue-item">🖼️ ' + escapeHtml(f.name) +
      ' <span class="rm" data-idx="' + i + '">✕</span></div>';
  });
  queueList.innerHTML = html;

  queueList.querySelectorAll('.rm').forEach(el => {
    el.addEventListener('click', () => {
      if (el.dataset.zip) removeZip();
      else removeImage(parseInt(el.dataset.idx, 10));
    });
  });
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Run batch ────────────────────────────────────────────────
runBatchBtn.addEventListener('click', runBatch);

async function runBatch() {
  const total = imageQueue.length + (zipQueue ? 1 : 0);
  if (total === 0) return;

  // Snapshot exactly what we're submitting, then clear the live queue right
  // away. Without this, picking new images after a run finished would get
  // added on top of whatever was still sitting in imageQueue from the last
  // run, silently growing the batch across multiple attempts.
  const submittedImages = imageQueue;
  const submittedZip    = zipQueue;
  imageQueue = [];
  zipQueue   = null;
  fileInput.value = '';
  zipInput.value  = '';

  queueSection.style.display = 'none';
  loadingSection.style.display = 'block';
  loadingText.textContent = 'Analyzing ' + submittedImages.length + ' image(s)' +
    (submittedZip ? ' + 1 zip' : '') + '...';
  resultsSection.style.display = 'none';
  summaryBar.style.display = 'none';

  const fd = new FormData();
  submittedImages.forEach(f => fd.append('images', f));
  if (submittedZip) fd.append('zipfile', submittedZip);

  try {
    const res  = await fetch('/batch_analyze', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      loadingSection.style.display = 'none';
      // Restore the queue so a failed request doesn't force re-selecting everything.
      imageQueue = submittedImages;
      zipQueue   = submittedZip;
      renderQueue();
      queueSection.style.display = 'block';
      return;
    }

    currentBatch = data;
    loadingSection.style.display = 'none';
    renderResults(data);

  } catch (err) {
    alert('Batch analysis failed. Make sure the Flask server is running.');
    loadingSection.style.display = 'none';
    imageQueue = submittedImages;
    zipQueue   = submittedZip;
    renderQueue();
    queueSection.style.display = 'block';
  }
}

// ── Render results grid + summary ───────────────────────────
function renderResults(data) {
  const results = data.results || [];

  // Summary
  const forgedCount = results.filter(r => r.predicted === 'FORGED').length;
  const authCount   = results.length - forgedCount;
  const avgForged   = results.length
    ? (results.reduce((s, r) => s + r.forged_prob, 0) / results.length).toFixed(1)
    : 0;

  document.getElementById('sumTotal').textContent  = results.length;
  document.getElementById('sumForged').textContent = forgedCount;
  document.getElementById('sumAuth').textContent   = authCount;
  document.getElementById('sumAvg').textContent    = avgForged + '%';
  summaryBar.style.display = 'block';

  // Skipped note
  if (data.skipped && data.skipped.length) {
    skipNote.style.display = 'block';
    skipNote.textContent = '⚠ Skipped ' + data.skipped.length + ' unreadable file(s): ' + data.skipped.join(', ');
  } else {
    skipNote.style.display = 'none';
  }
  if (data.truncated) {
    skipNote.style.display = 'block';
    skipNote.textContent = (skipNote.textContent ? skipNote.textContent + ' — ' : '') +
      '⚠ Batch capped at 100 images; extra files were not processed.';
  }

  // Cards
  let html = '';
  if (results.length === 0) {
    html = '<div class="empty-state">No results.</div>';
  }
  results.forEach(r => {
    const isForged = r.predicted === 'FORGED';
    const tag = isForged
      ? '<span class="tag tag-forged">FORGED</span>'
      : '<span class="tag tag-auth">AUTHENTIC</span>';
    html +=
      '<div class="rcard" data-image-id="' + r.image_id + '">' +
        '<img src="' + r.img_url + '" alt="' + escapeHtml(r.filename) + '">' +
        '<div class="rcard-body">' +
          '<div class="rcard-name" title="' + escapeHtml(r.filename) + '">' + escapeHtml(r.filename) + '</div>' +
          '<div class="rcard-row">' + tag +
            '<span class="rcard-conf">' + r.forged_prob + '%</span></div>' +
          '<div class="rcard-row">' +
            '<span class="trust-pill" style="background:' + r.trust_color + '22;color:' + r.trust_color + '">' + r.trust + '</span>' +
          '</div>' +
          '<button class="full-btn" data-image-id="' + r.image_id + '">🔬 Full Analyze</button>' +
        '</div>' +
      '</div>';
  });
  resultsGrid.innerHTML = html;
  resultsSection.style.display = 'block';

  resultsGrid.querySelectorAll('.full-btn').forEach(btn => {
    btn.addEventListener('click', () => runFullAnalyze(btn.dataset.imageId, btn));
  });
}

// ── Full analyze on demand ──────────────────────────────────
async function runFullAnalyze(imageId, btnEl) {
  if (!currentBatch) return;
  const origLabel = btnEl.innerHTML;
  btnEl.disabled = true;
  btnEl.innerHTML = '<span class="spinner small"></span> Analyzing...';

  try {
    const res  = await fetch('/batch_full/' + currentBatch.batch_id + '/' + imageId, { method: 'POST' });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      btnEl.disabled = false;
      btnEl.innerHTML = origLabel;
      return;
    }
    showModal(data);
  } catch (err) {
    alert('Full analysis failed.');
  } finally {
    btnEl.disabled = false;
    btnEl.innerHTML = origLabel;
  }
}

function showModal(d) {
  const isForged   = d.predicted === 'FORGED';
  const verdictCls = isForged ? 'forged' : 'authentic';
  const verdictIcon = isForged ? '⚠️' : '✅';

  let modelRows = '';
  const modelComp = d.model_comparison || {};
  Object.keys(modelComp).forEach(name => {
    const r = modelComp[name];
    const isOurs = name.includes('Hybrid') || name.includes('Ours');
    const tag = r.predicted === 'FORGED'
      ? '<span class="tag tag-forged">FORGED</span>'
      : '<span class="tag tag-auth">AUTHENTIC</span>';
    modelRows += '<tr><td>' + name + (isOurs ? ' ⭐' : '') + '</td><td>' + tag +
      '</td><td style="font-family:\'JetBrains Mono\',monospace">' + r.confidence + '%</td></tr>';
  });

  let platformHtml = '';
  const platforms = d.platform_results || {};
  Object.keys(platforms).forEach(pname => {
    const pdata = platforms[pname];
    const hybridR = pdata.hybrid;
    platformHtml += '<div class="platform-card"><div class="platform-name">' +
      pname.charAt(0).toUpperCase() + pname.slice(1) + '</div>';
    if (hybridR) {
      const t = hybridR.predicted === 'FORGED'
        ? '<span style="color:#ef4444;font-weight:600">FORGED</span>'
        : '<span style="color:#22c55e;font-weight:600">AUTHENTIC</span>';
      platformHtml += '<div>' + t + ' · ' + hybridR.confidence + '%</div>';
    } else {
      platformHtml += '<div style="color:var(--muted)">No data</div>';
    }
    platformHtml += '</div>';
  });

  const heatmapHtml = d.heatmap_url
    ? '<img src="' + d.heatmap_url + '" alt="Grad-CAM heatmap">'
    : '<div class="no-heatmap">No heatmap available</div>';

  const exif = d.exif_analysis || {};
  let exifHtml = '';
  if (exif.available === false) {
    exifHtml = '<div class="no-heatmap">' + (exif.note || 'EXIF analysis not available') + '</div>';
  } else if (exif.predicted) {
    const t = exif.predicted === 'FORGED'
      ? '<span style="color:#ef4444;font-weight:700">FORGED</span>'
      : '<span style="color:#22c55e;font-weight:700">AUTHENTIC</span>';
    exifHtml = '<div>' + t + ' · ' + exif.forged_prob + '% · Has EXIF: ' +
      (exif.has_exif ? 'Yes' : 'No') + (exif.flagged_editor_software ? ' · ⚠ Editor software detected' : '') + '</div>';
  }

  modalContent.innerHTML =
    '<div class="result-title" style="font-weight:700;font-size:1.2rem;margin-bottom:16px;">' +
      escapeHtml(d.filename || 'Analysis Result') + '</div>' +
    '<div class="verdict ' + verdictCls + '"><div class="verdict-inner">' +
      '<div class="verdict-icon">' + verdictIcon + '</div><div>' +
        '<div class="verdict-result">' + d.predicted + '</div>' +
        '<div class="verdict-conf">' + d.forged_prob + '% forged probability · ' + d.trust + '</div>' +
      '</div></div></div>' +
    '<div class="two-col">' +
      '<div class="mcard"><div class="mcard-header">Grad-CAM Heatmap</div>' + heatmapHtml + '</div>' +
      '<div class="mcard"><div class="mcard-header">Model Comparison</div>' +
        '<table class="model-table"><thead><tr><th>Model</th><th>Verdict</th><th>Conf.</th></tr></thead>' +
        '<tbody>' + modelRows + '</tbody></table></div>' +
    '</div>' +
    '<div class="mcard" style="margin-bottom:16px;"><div class="mcard-header">Platform Compression Test</div>' +
      '<div class="platform-grid">' + platformHtml + '</div></div>' +
    '<div class="mcard"><div class="mcard-header">EXIF Metadata Analysis</div>' + exifHtml + '</div>';

  modalOverlay.style.display = 'block';
}

modalCloseBtn.addEventListener('click', () => modalOverlay.style.display = 'none');
modalOverlay.addEventListener('click', e => { if (e.target === modalOverlay) modalOverlay.style.display = 'none'; });

// ── Export CSV (client-side, from the quick results already in memory) ──
exportCsvBtn.addEventListener('click', () => {
  if (!currentBatch) return;
  const rows = [['filename', 'predicted', 'forged_prob', 'auth_prob', 'trust']];
  currentBatch.results.forEach(r => {
    rows.push([r.filename, r.predicted, r.forged_prob, r.auth_prob, r.trust]);
  });
  const csv = rows.map(row => row.map(csvEscape).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'forensicai_batch_report.csv';
  a.click();
  URL.revokeObjectURL(url);
});

function csvEscape(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

// ── Print / Save as PDF (fills a hidden print-only report, then window.print) ──
printBtn.addEventListener('click', () => {
  if (!currentBatch) return;
  document.getElementById('printMeta').textContent =
    currentBatch.results.length + ' image(s) analyzed — ' + new Date().toLocaleString();
  const tbody = document.getElementById('printTableBody');
  tbody.innerHTML = currentBatch.results.map(r =>
    '<tr><td>' + escapeHtml(r.filename) + '</td><td>' + r.predicted + '</td><td>' +
    r.forged_prob + '%</td><td>' + r.auth_prob + '%</td><td>' + r.trust + '</td></tr>'
  ).join('');
  window.print();
});
