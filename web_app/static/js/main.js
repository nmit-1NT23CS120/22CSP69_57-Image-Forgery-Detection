// ForensicAI — main.js

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewSec  = document.getElementById('previewSection');
const loadingSec  = document.getElementById('loadingSection');
const previewImg  = document.getElementById('previewImg');
const previewName = document.getElementById('previewName');
const previewSize = document.getElementById('previewSize');
const analyzeBtn  = document.getElementById('analyzeBtn');
let   selectedFile = null;

// ── Drag & Drop ──────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) handleFile(f);
});
dropZone.addEventListener('click', e => {
  if (e.target.tagName !== 'BUTTON') fileInput.click();
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src  = url;
  previewName.textContent = file.name;
  previewSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
  previewSec.style.display = 'block';
  loadingSec.style.display = 'none';
  previewSec.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Loading animation ────────────────────────────────────────
function animateSteps() {
  const steps = ['step1','step2','step3','step4','step5'];
  let i = 0;
  steps.forEach(s => document.getElementById(s)?.classList.remove('active'));
  const iv = setInterval(() => {
    if (i < steps.length) {
      document.getElementById(steps[i])?.classList.add('active');
      i++;
    } else clearInterval(iv);
  }, 900);
}

// ── Analyze ──────────────────────────────────────────────────
async function analyzeImage() {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  previewSec.style.display  = 'none';
  loadingSec.style.display  = 'block';
  loadingSec.scrollIntoView({ behavior: 'smooth' });
  animateSteps();

  const fd = new FormData();
  fd.append('image', selectedFile);

  try {
    const res  = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      loadingSec.style.display = 'none';
      previewSec.style.display = 'block';
      analyzeBtn.disabled = false;
      return;
    }

    // Store results and navigate to results page
    sessionStorage.setItem('forensicResults', JSON.stringify(data));
    window.location.href = '/results';

  } catch (err) {
    alert('Analysis failed. Make sure Flask server is running.');
    loadingSec.style.display = 'none';
    previewSec.style.display = 'block';
    analyzeBtn.disabled = false;
  }
}

// ── Animate stats on scroll ──────────────────────────────────
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.querySelectorAll('.acc-fill').forEach(bar => {
        const w = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => bar.style.width = w, 100);
      });
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.models-grid').forEach(el => observer.observe(el));
