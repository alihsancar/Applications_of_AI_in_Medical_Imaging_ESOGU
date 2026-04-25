// static/js/main.js — DeepEmbryo UI Scripts

/* ─── Loading Overlay ─────────────────────────────── */

function showLoading(text = 'Embriyo analiz ediliyor...', sub = 'Lütfen bekleyin, bu birkaç saniye sürebilir.') {
  const overlay = document.getElementById('loadingOverlay');
  if (!overlay) return;
  overlay.querySelector('.loading-text').textContent = text;
  overlay.querySelector('.loading-sub').textContent  = sub;
  overlay.classList.add('active');
}

function hideLoading() {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) overlay.classList.remove('active');
}

/* ─── Single Image Drop Zone ──────────────────────── */

function initDropZone(zoneId, inputId, previewId, defaultId) {
  const zone     = document.getElementById(zoneId);
  const input    = document.getElementById(inputId);
  const preview  = document.getElementById(previewId);
  const def      = document.getElementById(defaultId);
  if (!zone || !input) return;

  // Drag events
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleSingleFile(files[0], input, preview, def);
  });

  // Input change
  input.addEventListener('change', () => {
    if (input.files.length > 0) handleSingleFile(input.files[0], input, preview, def);
  });

  // Click zone (not on input)
  zone.addEventListener('click', e => {
    if (e.target !== input) input.click();
  });
}

function handleSingleFile(file, input, preview, def) {
  if (!file.type.startsWith('image/')) {
    showToast('Lütfen geçerli bir görsel dosyası seçin.', 'error');
    return;
  }

  // Transfer to input
  const dt = new DataTransfer();
  dt.items.add(file);
  input.files = dt.files;

  // Show preview
  const reader = new FileReader();
  reader.onload = e => {
    const img = preview ? preview.querySelector('img') : null;
    if (img) img.src = e.target.result;

    const nameEl = preview ? preview.querySelector('.preview-name') : null;
    if (nameEl) nameEl.textContent = file.name;

    const sizeEl = preview ? preview.querySelector('.preview-size') : null;
    if (sizeEl) sizeEl.textContent = formatBytes(file.size);

    if (preview) preview.classList.add('visible');
    if (def)     def.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

/* ─── Batch File List ─────────────────────────────── */

function initBatchZone(zoneId, inputId, listId, countId) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const list  = document.getElementById(listId);
  const count = document.getElementById(countId);
  if (!zone || !input) return;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    addFilesToBatch(e.dataTransfer.files, input, list, count);
  });

  input.addEventListener('change', () => {
    addFilesToBatch(input.files, input, list, count);
  });

  zone.addEventListener('click', e => {
    if (e.target !== input) input.click();
  });
}

let batchFiles = [];

function addFilesToBatch(newFiles, input, list, count) {
  const imageTypes = ['image/png','image/jpeg','image/bmp','image/tiff'];
  for (const f of newFiles) {
    if (imageTypes.includes(f.type) && !batchFiles.find(x => x.name === f.name)) {
      batchFiles.push(f);
    }
  }

  // Update input
  const dt = new DataTransfer();
  batchFiles.forEach(f => dt.items.add(f));
  input.files = dt.files;

  // Render list
  if (list) {
    list.innerHTML = '';
    batchFiles.forEach((f, i) => {
      const li = document.createElement('li');
      li.className = 'file-item';
      li.innerHTML = `
        <span class="file-icon">🔬</span>
        <span class="file-name">${escapeHtml(f.name)}</span>
        <span class="file-size">${formatBytes(f.size)}</span>
        <button type="button" class="btn btn-sm btn-danger" onclick="removeBatchFile(${i}, '${inputId}', '${listId}', '${countId}')" style="padding:0.2rem 0.5rem;">✕</button>
      `;
      list.appendChild(li);
    });
  }

  if (count) count.textContent = batchFiles.length;
}

// Make inputId etc. available globally for remove function
let _batchInputId, _batchListId, _batchCountId;

function removeBatchFile(index, inputId, listId, countId) {
  batchFiles.splice(index, 1);
  const input = document.getElementById(inputId);
  const list  = document.getElementById(listId);
  const count = document.getElementById(countId);
  addFilesToBatch([], input, list, count);
}

/* ─── Probability Bars Animation ─────────────────── */

function animateProbBars() {
  document.querySelectorAll('.prob-bar-fill[data-width]').forEach(bar => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width + '%';
    }, 100);
  });
}

/* ─── Flash Messages Auto-dismiss ────────────────── */

function initFlashMessages() {
  document.querySelectorAll('.alert[data-auto-dismiss]').forEach(el => {
    setTimeout(() => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(-8px)';
      el.style.transition = 'all 0.4s ease';
      setTimeout(() => el.remove(), 400);
    }, 4000);
  });
}

/* ─── Toast Notifications ─────────────────────────── */

function showToast(message, type = 'info') {
  const icons = { info: 'ℹ️', success: '✅', error: '❌', warning: '⚠️' };
  const container = document.getElementById('toastContainer') || createToastContainer();
  const toast = document.createElement('div');
  toast.className = `alert alert-${type === 'error' ? 'danger' : type}`;
  toast.style.cssText = 'margin:0; min-width:280px; animation: slideIn 0.3s ease;';
  toast.innerHTML = `<span class="alert-icon">${icons[type] || 'ℹ️'}</span> ${escapeHtml(message)}`;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; setTimeout(() => toast.remove(), 300); }, 3500);
}

function createToastContainer() {
  const div = document.createElement('div');
  div.id = 'toastContainer';
  div.style.cssText = 'position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;display:flex;flex-direction:column;gap:0.5rem;';
  document.body.appendChild(div);
  return div;
}

/* ─── Confirm Delete ──────────────────────────────── */

function confirmDelete(formId) {
  if (confirm('Bu kaydı silmek istediğinizden emin misiniz?')) {
    document.getElementById(formId).submit();
  }
}

/* ─── Submit with Loading ─────────────────────────── */

function submitWithLoading(formId, loadingText, loadingSub) {
  const form = document.getElementById(formId);
  if (!form) return;
  form.addEventListener('submit', e => {
    const fileInput = form.querySelector('input[type="file"]');
    if (fileInput && fileInput.files.length === 0) {
      e.preventDefault();
      showToast('Lütfen önce bir görsel seçin.', 'warning');
      return;
    }
    showLoading(loadingText, loadingSub);
  });
}

/* ─── Helpers ─────────────────────────────────────── */

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

/* ─── DOM Ready ───────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  initFlashMessages();
  animateProbBars();

  // Set active nav link
  const path = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href && (path === href || (href !== '/' && path.startsWith(href)))) {
      link.classList.add('active');
    }
  });
});
