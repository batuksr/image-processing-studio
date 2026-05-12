/**
 * Image Processing Studio – Full Frontend Logic
 * Vanilla JS, no external libraries.
 */

// ── STATE MANAGEMENT ─────────────────────────────────────────────────────────
const state = {
  originalImage: null,
  currentImage: null,
  secondImageB64: null,
  activeOp: null,
  historyStack: [],
  historyIndex: -1,
  isCompareMode: false,
};

const resultCache = new Map(); // LRU cache for last 10 results

// ── İŞLEM KATALOĞU ───────────────────────────────────────────────────────────
const OPERATIONS = [
  {
    group: 'Temel İşlemler', icon: '🔧',
    ops: [
      { key: 'grayscale', label: 'Gri Dönüşüm', icon: '⬜', desc: 'Görüntüyü parlaklık değerlerine dönüştürür (Y = 0.299R + 0.587G + 0.114B).', params: [] },
      { key: 'binary', label: 'Binary Dönüşüm', icon: '◼', desc: 'Piksel yoğunluk eşiğine göre görüntüyü siyah-beyaza çevirir.',
        params: [
          { id:'threshold', label:'Eşik Değeri', type:'range', min:0, max:255, step:1, default:128 },
          { id:'preview', label:'Canlı Önizleme', type:'checkbox', default:false, action:'auto_apply' }
        ] 
      },
      { key: 'rotate', label: 'Görüntü Döndürme', icon: '🔄', desc: 'Bilineer interpolasyon kullanarak görüntüyü döndürür.',
        params: [
          { id:'angle', label:'Açı (°)', type:'range', min:-180, max:180, step:1, default:90 },
          { id:'expand', label:'Tuvali Genişlet', type:'checkbox', default:true }
        ] 
      },
      { key: 'crop', label: 'Görüntü Kırpma', icon: '✂️', desc: 'X/Y koordinatları ve boyutlar kullanarak görüntüyü kırpar.',
        params: [
          { id:'x', label:'X Başlangıç', type:'number', default:0 },
          { id:'y', label:'Y Başlangıç', type:'number', default:0 },
          { id:'width', label:'Genişlik', type:'number', default:256 },
          { id:'height', label:'Yükseklik', type:'number', default:256 }
        ] 
      },
      { key: 'zoom', label: 'Yaklaştırma / Uzaklaştırma', icon: '🔍', desc: 'En yakın komşu yöntemi ile görüntüyü X ve Y eksenlerinde bağımsız olarak ölçekler.',
        params: [
          { id:'scale_x', label:'X Ölçeği', type:'range', min:0.1, max:5.0, step:0.1, default:1.5 },
          { id:'scale_y', label:'Y Ölçeği', type:'range', min:0.1, max:5.0, step:0.1, default:1.5 }
        ] 
      },
    ],
  },
  {
    group: 'Histogram', icon: '📊',
    ops: [
      { key: 'hist_stretch', label: 'Histogram Germe', icon: '📈', desc: 'Piksel yoğunluklarını doğrusal olarak 0-255 aralığına genişletir.', params: [] },
      { key: 'hist_equalize', label: 'Histogram Eşitleme', icon: '⚖️', desc: 'Kümülatif Dağılım Fonksiyonu (CDF) kullanarak histogramı düzleştirir.', params: [] },
    ],
  },
  {
    group: 'Aritmetik İşlemler', icon: '➕', requiresSecond: true,
    ops: [
      { key: 'add_images', label: 'İki Resim Toplama', icon: '➕', desc: 'İki görüntüyü alfa ağırlığı ile piksel piksel toplar.',
        params: [{ id:'alpha', label:'Alfa (ağırlık)', type:'range', min:0, max:1, step:0.01, default:0.5 }] },
      { key: 'multiply_images', label: 'İki Resim Çarpma', icon: '✖️', desc: 'İki görüntünün piksel yoğunluklarını çarpar.', params: [] },
      { key: 'subtract_images', label: 'Mutlak Fark', icon: '➖', desc: 'Piksel yoğunluklarını çıkarır ve mutlak değer alır.', params: [] },
    ]
  },
  {
    group: 'Renk Uzayı Dönüşümleri', icon: '🎨',
    ops: [
      { key: 'color_space_unified', label: 'Dönüştür & Kanalları Göster', icon: '🌈', desc: 'Renk uzayını dönüştürür ve bireysel kanalları ayrı olarak görüntüler.',
        params: [
          { id:'conversion', label:'Dönüşüm Modu', type:'select', default:'rgb_to_hsv_display', options:[
              {val:'rgb_to_hsv_display', text:'RGB → HSV'},
              {val:'rgb_to_ycbcr', text:'RGB → YCbCr'},
              {val:'rgb_to_lab', text:'RGB → L*a*b*'},
              {val:'pseudo_color_jet', text:'Jet (Sözde Renk)'},
              {val:'pseudo_color_hot', text:'Hot (Sözde Renk)'},
              {val:'pseudo_color_cool', text:'Cool (Sözde Renk)'}
            ]
          }
        ] 
      },
    ],
  },
  {
    group: 'Filtreler', icon: '🌀',
    ops: [
      { key: 'gaussian_filter', label: 'Konvolüsyon İşlemi (Gauss)', icon: '🫧', desc: 'Gauss dağılımı çekirdeği kullanarak görüntüyü yumuşatır.',
        params: [
          { id:'kernel_size', label:'Çekirdek Boyutu', type:'select', default:5, options:[{val:3,text:'3x3'},{val:5,text:'5x5'},{val:7,text:'7x7'},{val:9,text:'9x9'},{val:11,text:'11x11'}] },
          { id:'sigma', label:'Sigma (σ)', type:'range', min:0.1, max:5.0, step:0.1, default:1.0 }
        ] 
      },
      { key: 'blur_unified', label: 'Görüntü Filtre Uygulaması (Blurring)', icon: '📦', desc: 'Genel kutu veya Gauss yumuşatma filtresi uygular.',
        params: [
          { id:'method', label:'Yöntem', type:'radio', default:'box', options:[{val:'box',text:'Kutu (Ortalama)'},{val:'gaussian',text:'Gauss'}] },
          { id:'kernel_size', label:'Çekirdek Boyutu', type:'range', min:3, max:21, step:2, default:5 }
        ] 
      },
    ],
  },
  {
    group: 'Kenar Bulma', icon: '🔲',
    ops: [
      { key: 'sobel_edge', label: 'Kenar Bulma (Sobel)', icon: '🔲', desc: '3x3 gradyan operatörleri ile görüntüdeki kenarları tespit eder.',
        params: [
          { id:'threshold', label:'Kenar Eşiği', type:'range', min:0, max:255, step:1, default:50 },
          { id:'show_dir', label:'Gradyan Yönünü Göster', type:'checkbox', default:false }
        ] 
      },
    ],
  },
  {
    group: 'Gürültü', icon: '📡',
    ops: [
      { key: 'add_salt_pepper', label: 'Görüntüye Gürültü Ekleme (Tuz&Biber)', icon: '🧂', desc: 'Pikselleri rastgele minimum ve maksimum yoğunluk değerleriyle bozar.',
        params: [
          { id:'density', label:'Yoğunluk', type:'range', min:0.01, max:0.3, step:0.01, default:0.05 }
        ] 
      },
      { key: 'denoise_unified', label: 'Gürültü Temizleme (Mean/Median)', icon: '〽️', desc: 'Doğrusal (ortalama) veya doğrusal olmayan (medyan) tekniklerle gürültüyü giderir.',
        params: [
          { id:'method', label:'Yöntem', type:'radio', default:'median', options:[{val:'mean',text:'Ortalama Filtre'},{val:'median',text:'Medyan Filtre'}] },
          { id:'kernel_size', label:'Çekirdek Boyutu', type:'select', default:3, options:[{val:3,text:'3x3'},{val:5,text:'5x5'},{val:7,text:'7x7'}] }
        ] 
      },
    ],
  },
  {
    group: 'Morfolojik İşlemler', icon: '🧬',
    ops: [
      { key: 'morphology_unified', label: 'Morfolojik Filtreler', icon: '⊕', desc: 'Görüntü şekline dayalı doğrusal olmayan morfolojik işlemler uygular.',
        params: [
          { id:'op', label:'İşlem', type:'radio', default:'dilate', options:[
              {val:'dilate',text:'Genişletme'},{val:'erode',text:'Aşındırma'},{val:'opening',text:'Açma'},{val:'closing',text:'Kapama'}
            ]
          },
          { id:'shape', label:'Yapısal Eleman Şekli', type:'select', default:'rect', options:[
              {val:'rect',text:'Dikdörtgen'},{val:'ellipse',text:'Elips'},{val:'cross',text:'Artı'}
            ]
          },
          { id:'kernel_size', label:'Eleman Boyutu', type:'select', default:3, options:[
              {val:3,text:'3x3'},{val:5,text:'5x5'},{val:7,text:'7x7'},{val:9,text:'9x9'}
            ]
          }
        ] 
      },
    ],
  },
];

// ── DOM REFERENCES ───────────────────────────────────────────────────────────
const elOriginalImg    = document.getElementById('original-img');
const elProcessedImg   = document.getElementById('processed-img');
const elOriginalEmpty  = document.getElementById('original-empty');
const elProcessedEmpty = document.getElementById('processed-empty');
const elOriginalInfo   = document.getElementById('original-info');
const elProcessedInfo  = document.getElementById('processed-info');
const elOpList         = document.getElementById('op-list');
const elParamsContent  = document.getElementById('params-content');
const elApplyBtn       = document.getElementById('apply-btn');
const elHistoryStrip   = document.getElementById('history-thumbs');
const elLoadingOverlay = document.getElementById('loading-overlay');
const elSecondSection  = document.getElementById('second-image-section');
const elSecondInput    = document.getElementById('second-image-input');
const elSecondPreview  = document.getElementById('second-preview');
const elToastContainer = document.getElementById('toast-container');
const elUploadInput    = document.getElementById('upload-input');
const elDownloadBtn    = document.getElementById('download-btn');
const elResetBtn       = document.getElementById('reset-btn');
const elUseProcessed   = document.getElementById('use-processed-btn');

const viewBtns         = document.querySelectorAll('.view-controls .icon-btn');
const canvasStage      = document.querySelector('.canvas-stage');
const processedPane    = document.querySelector('.processed-pane');

// Setup Histogram Canvas
const histContainer = document.querySelector('.histogram-mini');
const histBars = document.querySelector('.hist-bars');
if (histBars) histBars.remove();
const histCanvas = document.createElement('canvas');
histCanvas.width = 120;
histCanvas.height = 36;
histCanvas.style.cssText = 'width:120px; height:36px; border-radius:4px; background:#1a1a24; border:1px solid var(--border);';
histContainer.appendChild(histCanvas);

// Setup Time Badge
const timeBadge = document.createElement('div');
timeBadge.style.cssText = 'position:absolute; bottom:16px; right:16px; background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.5); color:#a5b4fc; font-family:var(--font-mono); font-size:11px; padding:4px 8px; border-radius:4px; backdrop-filter:blur(4px); display:none; z-index:10;';
processedPane.appendChild(timeBadge);

// Setup Compare Slider Handle
const compareHandle = document.createElement('div');
compareHandle.className = 'compare-handle';
compareHandle.style.cssText = 'display:none; position:absolute; top:0; bottom:0; left:50%; width:4px; background:var(--accent-indigo); z-index:100; cursor:ew-resize; transform:translateX(-50%); box-shadow:0 0 10px rgba(0,0,0,0.5);';
const compareKnob = document.createElement('div');
compareKnob.style.cssText = 'position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); width:24px; height:24px; border-radius:50%; background:#fff; border:4px solid var(--accent-indigo); box-shadow:0 2px 8px rgba(0,0,0,0.4);';
compareHandle.appendChild(compareKnob);
canvasStage.style.position = 'relative';
canvasStage.appendChild(compareHandle);

// Feature Specific Canvases
const cdfCanvas = document.createElement('canvas');
cdfCanvas.style.cssText = 'width:100%; height:120px; border:1px solid var(--border); border-radius:4px; margin-top:10px; display:none;';
const channelBox = document.createElement('div');
channelBox.style.cssText = 'display:flex; gap:8px; margin-top:10px; display:none;';
const morphPreview = document.createElement('canvas');
morphPreview.style.cssText = 'width:60px; height:60px; border:1px solid var(--border); image-rendering:pixelated; margin-top:8px; display:none;';

// ── FILE HANDLING & DRAG-DROP ────────────────────────────────────────────────
const dropZone = document.getElementById('upload-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent-indigo)'; });
dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '');
dropZone.addEventListener('drop', async e => {
  e.preventDefault(); dropZone.style.borderColor = '';
  if (e.dataTransfer.files[0]) await uploadFile(e.dataTransfer.files[0]);
});
elUploadInput.addEventListener('change', async e => { if (e.target.files[0]) await uploadFile(e.target.files[0]); });

window.addEventListener('paste', async e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (let index in items) {
    if (items[index].kind === 'file' && items[index].type.startsWith('image/')) {
      await uploadFile(items[index].getAsFile());
      break;
    }
  }
});

async function uploadFile(file) {
  if (file.size > 10 * 1024 * 1024) { showToast('Dosya çok büyük (max 10MB)', 'error'); return; }
  if (!['image/png', 'image/jpeg'].includes(file.type)) { showToast('Geçersiz dosya türü (sadece PNG/JPG)', 'error'); return; }

  setLoading(true);
  const fd = new FormData();
  fd.append('image', file);
  try {
    const res = await fetch('/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Yükleme başarısız oldu');
    
    state.originalImage = data.image_data;
    state.currentImage = data.image_data;
    state.historyStack = [];
    state.historyIndex = -1;
    resultCache.clear();
    elHistoryStrip.innerHTML = '';
    
    pushHistory(data.image_data);
    showImage(elOriginalImg, elOriginalEmpty, data.image_data);
    clearProcessed();
    elOriginalInfo.textContent = `${data.width} × ${data.height}`;
    updateHistogram();
    showToast(`${file.name} yüklendi`, 'success');
  } catch (err) { showToast(err.message, 'error'); } 
  finally { setLoading(false); }
}

// Second Image Upload (for arithmetic)
const secondDropZone = document.getElementById('second-upload-zone');
secondDropZone.addEventListener('dragover', e => { e.preventDefault(); secondDropZone.style.borderColor = 'var(--accent-purple)'; });
secondDropZone.addEventListener('dragleave', () => secondDropZone.style.borderColor = '');
secondDropZone.addEventListener('drop', e => {
  e.preventDefault(); secondDropZone.style.borderColor = '';
  if (e.dataTransfer.files[0]) handleSecondImage(e.dataTransfer.files[0]);
});
elSecondInput.addEventListener('change', e => {
  if (e.target.files[0]) handleSecondImage(e.target.files[0]);
});
function handleSecondImage(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width; canvas.height = img.height;
      canvas.getContext('2d').drawImage(img, 0, 0);
      state.secondImageB64 = canvas.toDataURL('image/png').split(',')[1];
      elSecondPreview.src = canvas.toDataURL('image/png');
      elSecondPreview.style.display = 'block';
      secondDropZone.style.animation = ''; // Stop pulsing if it was
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
}

// ── SIDEBAR RENDERING ────────────────────────────────────────────────────────
function buildSidebar() {
  elOpList.innerHTML = '';
  OPERATIONS.forEach(group => {
    const groupEl = document.createElement('div');
    groupEl.className = 'op-group';
    const header = document.createElement('div');
    header.className = 'op-group-header open';
    header.innerHTML = `<span class="group-icon">${group.icon}</span>${group.group}<span class="chevron">▶</span>`;
    const items = document.createElement('div');
    items.className = 'op-group-items open';

    header.addEventListener('click', () => { 
      header.classList.toggle('open'); 
      items.classList.toggle('open'); 
      if (header.classList.contains('open')) groupEl.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    });

    group.ops.forEach((op, idx) => {
      const btn = document.createElement('button');
      btn.className = 'op-btn';
      btn.id = `op-${op.key}`;
      btn.innerHTML = `<span class="op-icon">${op.icon}</span>${op.label}`;
      btn.title = op.desc; // Native tooltip
      btn.style.animationDelay = `${idx * 0.03}s`;
      btn.classList.add('animate-slide-in');
      btn.addEventListener('click', () => selectOperation(op, group));
      items.appendChild(btn);
    });
    groupEl.appendChild(header);
    groupEl.appendChild(items);
    elOpList.appendChild(groupEl);
  });
}

function selectOperation(op, group) {
  document.querySelectorAll('.op-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`op-${op.key}`).classList.add('active');
  state.activeOp = op.key;
  elSecondSection.classList.toggle('visible', group.requiresSecond || false);
  renderParams(op);
}

let debounceTimer;
function debouncedApply() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => { elApplyBtn.click(); }, 300);
}

function renderParams(op) {
  elParamsContent.innerHTML = '';
  
  const header = document.createElement('div');
  header.style.cssText = 'display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;';
  header.innerHTML = `<h3 style="font-size:13px; color:var(--accent-purple); margin:0;">${op.label}</h3><span title="${op.desc}" style="cursor:help; color:var(--text-muted); font-size:14px;">❓</span>`;
  elParamsContent.appendChild(header);

  if (!op.params || op.params.length === 0) {
    elParamsContent.insertAdjacentHTML('beforeend', '<div class="empty-params" style="margin-top:20px;">Parametre gerekmiyor. Sadece uygulayın.</div>');
  }

  cdfCanvas.style.display = 'none'; channelBox.style.display = 'none'; morphPreview.style.display = 'none';
  elParamsContent.appendChild(cdfCanvas); elParamsContent.appendChild(channelBox); elParamsContent.appendChild(morphPreview);
  if (op.key === 'hist_equalize') cdfCanvas.style.display = 'block';

  (op.params || []).forEach(p => {
    const group = document.createElement('div'); group.className = 'param-group';

    if (p.type === 'range') {
      const label = document.createElement('div'); label.className = 'param-label';
      label.innerHTML = `<span>${p.label}</span> <span class="val-display" id="val-${p.id}">${p.default}</span>`;
      const input = document.createElement('input');
      input.type = 'range'; input.id = `param-${p.id}`; input.min = p.min; input.max = p.max; input.step = p.step; input.value = p.default;
      input.addEventListener('input', () => { 
        document.getElementById(`val-${p.id}`).textContent = input.value; 
        
        // Instant visual previews
        if (op.key === 'rotate' && p.id === 'angle' && state.currentImage) {
          elProcessedImg.style.transform = `rotate(${input.value}deg)`;
        }
        if (op.key === 'zoom' && state.currentImage) {
          const sx = document.getElementById('param-scale_x')?.value || 1;
          const sy = document.getElementById('param-scale_y')?.value || 1;
          elProcessedImg.style.transform = `scale(${sx}, ${sy})`;
        }
      });
      // Send to backend only on drag release (change)
      input.addEventListener('change', () => {
        elProcessedImg.style.transform = ''; // reset CSS transform so backend handles it
        if(document.querySelector('#param-preview')?.checked) debouncedApply();
      });
      group.appendChild(label); group.appendChild(input);
    } 
    else if (p.type === 'number') {
      const label = document.createElement('div'); label.className = 'param-label'; label.innerHTML = `<span>${p.label}</span>`;
      const input = document.createElement('input'); input.type = 'number'; input.id = `param-${p.id}`; input.value = p.default;
      group.appendChild(label); group.appendChild(input);
    } 
    else if (p.type === 'select') {
      const label = document.createElement('div'); label.className = 'param-label'; label.innerHTML = `<span>${p.label}</span>`;
      const sel = document.createElement('select'); sel.id = `param-${p.id}`;
      p.options.forEach(opt => {
        const o = document.createElement('option'); o.value = opt.val; o.textContent = opt.text;
        if (opt.val == p.default) o.selected = true; sel.appendChild(o);
      });
      if (op.key === 'morphology_unified') sel.addEventListener('change', drawMorphPreview);
      sel.addEventListener('change', () => { if(document.querySelector('#param-preview')?.checked) debouncedApply(); });
      group.appendChild(label); group.appendChild(sel);
    } 
    else if (p.type === 'radio') {
      const label = document.createElement('div'); label.className = 'param-label'; label.innerHTML = `<span>${p.label}</span>`;
      group.appendChild(label);
      const radioContainer = document.createElement('div'); radioContainer.style.cssText = 'display:flex; gap:12px; font-size:12px; color:var(--text-main); margin-top:4px;';
      p.options.forEach(opt => {
        const lbl = document.createElement('label'); lbl.style.cssText = 'display:flex; align-items:center; gap:4px; cursor:pointer;';
        const rad = document.createElement('input'); rad.type = 'radio'; rad.name = `param-${p.id}`; rad.value = opt.val;
        if (opt.val == p.default) rad.checked = true;
        rad.addEventListener('change', () => { if(document.querySelector('#param-preview')?.checked) debouncedApply(); });
        lbl.appendChild(rad); lbl.appendChild(document.createTextNode(opt.text)); radioContainer.appendChild(lbl);
      });
      group.appendChild(radioContainer);
    }
    else if (p.type === 'checkbox') {
      const wrap = document.createElement('label'); wrap.style.cssText = 'display:flex; align-items:center; gap:8px; font-size:13px; color:var(--text-main); cursor:pointer; margin-top:10px;';
      const cb = document.createElement('input'); cb.type = 'checkbox'; cb.id = `param-${p.id}`; cb.checked = p.default;
      cb.addEventListener('change', () => { if (p.action === 'auto_apply' && cb.checked) debouncedApply(); });
      wrap.appendChild(cb); wrap.appendChild(document.createTextNode(p.label)); group.appendChild(wrap);
    }
    elParamsContent.appendChild(group);
  });

  if (op.key === 'morphology_unified') { morphPreview.style.display = 'block'; drawMorphPreview(); }
}

function collectParams() {
  const params = {};
  document.querySelectorAll('[id^="param-"]').forEach(el => {
    const key = el.id.replace('param-', '');
    if (el.type === 'checkbox') params[key] = el.checked;
    else if (el.type === 'range' || el.type === 'number') params[key] = parseFloat(el.value);
    else params[key] = el.value;
  });
  document.querySelectorAll('input[type="radio"]:checked').forEach(el => { params[el.name.replace('param-', '')] = el.value; });
  return params;
}

// ── OPERATION API CALL ───────────────────────────────────────────────────────
elApplyBtn.addEventListener('click', async () => {
  if (!state.currentImage && !state.originalImage) { showToast('Önce bir görüntü yükleyin.', 'error'); return; }
  if (!state.activeOp) { showToast('Bir işlem seçin.', 'error'); return; }
  
  // Check arithmetic op requirement
  const group = OPERATIONS.find(g => g.ops.some(o => o.key === state.activeOp));
  if (group?.requiresSecond && !state.secondImageB64) {
    showToast('Bu işlem için ikinci bir görüntü yükleyin.', 'error');
    secondDropZone.style.animation = 'pulse 1.5s infinite';
    return;
  }

  setLoading(true);
  try {
    const params = collectParams();
    let sourceImage = state.currentImage || state.originalImage;
    if (state.secondImageB64) params.image2 = state.secondImageB64;
    
    let opKey = state.activeOp;
    if (opKey === 'sobel_edge' && params.show_dir) opKey = 'gradient_direction';

    // Caching check
    const cacheKey = `${opKey}_${JSON.stringify(params)}_${sourceImage.slice(0, 50)}`;
    if (resultCache.has(cacheKey)) {
      handleSuccess(resultCache.get(cacheKey));
      setLoading(false);
      return;
    }

    const body = { operation: opKey, params, image_data: sourceImage };
    const res = await fetch('/apply', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Processing failed');
    if (data.warning) showToast(data.warning, 'info');

    // Add to LRU cache
    if (resultCache.size > 10) resultCache.delete(resultCache.keys().next().value);
    resultCache.set(cacheKey, data);

    handleSuccess(data);
  } catch (err) { showToast(err.message, 'error'); } 
  finally { setLoading(false); }
});

function handleSuccess(data) {
  state.currentImage = data.image_data;
  elProcessedImg.style.opacity = 0;
  showImage(elProcessedImg, elProcessedEmpty, data.image_data);
  setTimeout(() => elProcessedImg.style.opacity = 1, 50);
  
  elProcessedInfo.textContent = `${data.width} × ${data.height}`;
  timeBadge.textContent = `${data.time_ms}ms`;
  timeBadge.style.display = 'block';
  
  pushHistory(data.image_data);
  updateHistogram();

  if (state.activeOp === 'hist_equalize') drawCDF(data.image_data);
  if (state.activeOp === 'color_space_unified') renderChannels(data.image_data);
  
  showToast(`Success`, 'success');
}

// ── MISC VISUALS ─────────────────────────────────────────────────────────────
function drawMorphPreview() {
  const sizeSelect = document.getElementById('param-kernel_size'); const shapeSelect = document.getElementById('param-shape');
  if (!sizeSelect || !shapeSelect) return;
  const size = parseInt(sizeSelect.value); const shape = shapeSelect.value;
  morphPreview.width = size; morphPreview.height = size;
  const ctx = morphPreview.getContext('2d');
  ctx.fillStyle = '#0a0a0f'; ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#a78bfa';
  const c = Math.floor(size/2);
  for(let r=0; r<size; r++) {
    for(let col=0; col<size; col++) {
      let active = false;
      if (shape === 'rect') active = true;
      else if (shape === 'cross') active = (r === c || col === c);
      else if (shape === 'ellipse') active = (Math.pow(r-c, 2) + Math.pow(col-c, 2)) <= Math.pow(c, 2);
      if (active) ctx.fillRect(col, r, 1, 1);
    }
  }
}

async function drawCDF(b64) {
  const hist = await computeHistArray(b64);
  const total = hist.reduce((a,b)=>a+b, 0);
  let cdf = [], sum = 0;
  for(let v of hist) { sum += v; cdf.push(sum / total); }
  cdfCanvas.width = 256; cdfCanvas.height = 100;
  const ctx = cdfCanvas.getContext('2d'); ctx.clearRect(0,0,256,100);
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2; ctx.beginPath();
  for(let i=0; i<256; i++) {
    const x = i; const y = 100 - (cdf[i] * 100);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();
}

function renderChannels(b64) {
  channelBox.innerHTML = ''; channelBox.style.display = 'flex';
  const img = new Image();
  img.onload = () => {
    const labels = ['Ch 1', 'Ch 2', 'Ch 3'];
    for(let c=0; c<3; c++) {
      const wrap = document.createElement('div'); wrap.style.cssText = 'flex:1; text-align:center; font-size:10px; color:var(--text-muted);';
      const cvs = document.createElement('canvas'); cvs.width = img.width; cvs.height = img.height; cvs.style.cssText = 'width:100%; border-radius:4px; margin-bottom:4px;';
      const ctx = cvs.getContext('2d'); ctx.drawImage(img, 0, 0);
      const dat = ctx.getImageData(0,0,cvs.width,cvs.height); const px = dat.data;
      for(let i=0; i<px.length; i+=4) { px[i] = px[i+c]; px[i+1] = px[i+c]; px[i+2] = px[i+c]; }
      ctx.putImageData(dat, 0, 0); wrap.appendChild(cvs); wrap.appendChild(document.createTextNode(labels[c])); channelBox.appendChild(wrap);
    }
  };
  img.src = `data:image/png;base64,${b64}`;
}

// ── HISTORY (UNDO/REDO) ──────────────────────────────────────────────────────
function pushHistory(b64) {
  state.historyStack = state.historyStack.slice(0, state.historyIndex + 1);
  state.historyStack.push(b64); state.historyIndex++; renderHistoryStrip();
}

function renderHistoryStrip() {
  elHistoryStrip.innerHTML = '';
  state.historyStack.forEach((b64, idx) => {
    const img = document.createElement('img'); img.src = `data:image/png;base64,${b64}`; img.className = `history-thumb ${idx === state.historyIndex ? 'active' : ''}`;
    img.addEventListener('click', () => loadHistoryState(idx)); elHistoryStrip.appendChild(img);
  });
  elHistoryStrip.scrollLeft = elHistoryStrip.scrollWidth;
}

function loadHistoryState(index) {
  if (index < 0 || index >= state.historyStack.length) return;
  state.historyIndex = index; state.currentImage = state.historyStack[index];
  if (index === 0) clearProcessed(); else showImage(elProcessedImg, elProcessedEmpty, state.currentImage);
  renderHistoryStrip(); updateHistogram();
}

window.addEventListener('keydown', e => {
  if (e.key === '?') toggleShortcuts();
  if (e.ctrlKey && e.key.toLowerCase() === 'z') { e.preventDefault(); if (state.historyIndex > 0) loadHistoryState(state.historyIndex - 1); }
  if (e.ctrlKey && e.key.toLowerCase() === 'y') { e.preventDefault(); if (state.historyIndex < state.historyStack.length - 1) loadHistoryState(state.historyIndex + 1); }
});

// ── COMPARE SLIDER ───────────────────────────────────────────────────────────
viewBtns.forEach((btn, idx) => {
  btn.addEventListener('click', () => {
    viewBtns.forEach(b => b.classList.remove('active')); btn.classList.add('active');
    state.isCompareMode = (idx === 1); toggleCompareMode();
  });
});

let isDraggingCompare = false;
function toggleCompareMode() {
  if (state.isCompareMode && state.currentImage && state.currentImage !== state.originalImage) {
    canvasStage.style.display = 'block';
    document.querySelector('.original-pane').style.cssText = 'position: absolute; inset: 24px; z-index: 1;';
    processedPane.style.cssText = 'position: absolute; inset: 24px; z-index: 2; clip-path: polygon(50% 0, 100% 0, 100% 100%, 50% 100%); background: transparent; border: none;';
    [elOriginalImg, elProcessedImg].forEach(img => { img.style.width = '100%'; img.style.height = '100%'; img.style.objectFit = 'contain'; });
    compareHandle.style.display = 'block'; compareHandle.style.left = '50%';
  } else {
    canvasStage.style.display = 'flex';
    document.querySelector('.original-pane').style.cssText = '';
    processedPane.style.cssText = ''; compareHandle.style.display = 'none';
  }
}

compareHandle.addEventListener('mousedown', () => isDraggingCompare = true);
window.addEventListener('mouseup', () => isDraggingCompare = false);
window.addEventListener('mousemove', e => {
  if (!isDraggingCompare || !state.isCompareMode) return;
  const rect = canvasStage.getBoundingClientRect();
  let x = Math.max(24, Math.min(e.clientX - rect.left, rect.width - 24));
  const percent = (x / rect.width) * 100;
  compareHandle.style.left = `${percent}%`;
  processedPane.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
});

// ── HISTOGRAM ────────────────────────────────────────────────────────────────
async function computeHistArray(b64) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0);
      const data = ctx.getImageData(0,0,c.width,c.height).data;
      const hist = new Array(256).fill(0);
      for(let i=0; i<data.length; i+=4) hist[Math.round(0.299*data[i] + 0.587*data[i+1] + 0.114*data[i+2])]++;
      resolve(hist);
    };
    img.src = `data:image/png;base64,${b64}`;
  });
}

async function updateHistogram() {
  if (!state.originalImage) return;
  const histOrig = await computeHistArray(state.originalImage);
  const histCur = (state.currentImage && state.currentImage !== state.originalImage) ? await computeHistArray(state.currentImage) : null;
  const ctx = histCanvas.getContext('2d'); const w = histCanvas.width; const h = histCanvas.height;
  ctx.clearRect(0, 0, w, h);
  const globalMax = Math.max(Math.max(...histOrig), histCur ? Math.max(...histCur) : 0);
  if (globalMax === 0) return;
  
  ctx.globalCompositeOperation = 'screen';
  ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
  for(let i=0; i<256; i++) ctx.fillRect((i/256)*w, h - (histOrig[i]/globalMax*h), w/256, (histOrig[i]/globalMax*h));
  if (histCur) {
    ctx.fillStyle = 'rgba(99, 102, 241, 0.8)';
    for(let i=0; i<256; i++) ctx.fillRect((i/256)*w, h - (histCur[i]/globalMax*h), w/256, (histCur[i]/globalMax*h));
  }
}

// ── UTILS ────────────────────────────────────────────────────────────────────
elDownloadBtn.addEventListener('click', () => {
  if (!state.currentImage) return;
  const a = document.createElement('a'); a.href = `data:image/png;base64,${state.currentImage}`; a.download = 'result.png'; a.click();
});
document.addEventListener('keydown', async e => {
  if (e.ctrlKey && e.key.toLowerCase() === 'c' && state.currentImage) {
    try {
      const res = await fetch(`data:image/png;base64,${state.currentImage}`);
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': await res.blob() })]);
      showToast('Panoya kopyalandı', 'success');
    } catch (err) {}
  }
});
elResetBtn.addEventListener('click', () => {
  if (state.originalImage) {
    elProcessedImg.style.transition = 'opacity 0.3s';
    elProcessedImg.style.opacity = 0;
    setTimeout(() => { loadHistoryState(0); elProcessedImg.style.opacity = 1; }, 300);
    showToast('Orijinale Döndürüldü.', 'info');
  }
});
elUseProcessed.addEventListener('click', () => {
  if (!state.currentImage || state.currentImage === state.originalImage) return;
  state.originalImage = state.currentImage; showImage(elOriginalImg, elOriginalEmpty, state.currentImage); clearProcessed();
  state.historyStack = [state.currentImage]; state.historyIndex = 0; renderHistoryStrip(); updateHistogram();
});

function showImage(imgEl, emptyEl, b64) { imgEl.src = `data:image/png;base64,${b64}`; imgEl.style.display = 'block'; emptyEl.style.display = 'none'; toggleCompareMode(); }
function clearProcessed() { elProcessedImg.style.display = 'none'; elProcessedEmpty.style.display = 'flex'; elProcessedInfo.textContent = '-- x --'; timeBadge.style.display = 'none'; toggleCompareMode(); }
function setLoading(on) { state.loading = on; elLoadingOverlay.classList.toggle('visible', on); elApplyBtn.disabled = on; }
function showToast(msg, type = 'info') {
  const toast = document.createElement('div'); toast.className = `toast ${type}`; toast.innerHTML = `<span>${msg}</span>`;
  elToastContainer.appendChild(toast); setTimeout(() => { toast.style.animation = 'toastSlideIn 0.3s reverse forwards'; setTimeout(() => toast.remove(), 300); }, 4000); // 4s timeout
}

function toggleShortcuts() {
  const modal = document.getElementById('shortcuts-modal');
  if (modal) { modal.remove(); return; }
  const m = document.createElement('div'); m.id = 'shortcuts-modal';
  m.style.cssText = 'position:fixed; top:50%; left:50%; transform:translate(-50%, -50%); background:var(--bg-panel); border:1px solid var(--border); padding:24px; border-radius:8px; z-index:9999; box-shadow:var(--shadow-panel); width:300px; color:var(--text-main);';
  m.innerHTML = `<h3 style="margin:0 0 16px 0; color:var(--accent-purple);">Klavye Kısayolları</h3>
    <ul style="list-style:none; padding:0; margin:0; font-size:13px; line-height:2;">
      <li><kbd style="background:var(--bg-main); padding:2px 6px; border-radius:4px;">Ctrl+Z</kbd> Geri Al</li>
      <li><kbd style="background:var(--bg-main); padding:2px 6px; border-radius:4px;">Ctrl+Y</kbd> Yinele</li>
      <li><kbd style="background:var(--bg-main); padding:2px 6px; border-radius:4px;">Ctrl+C</kbd> Sonucu kopyala</li>
      <li><kbd style="background:var(--bg-main); padding:2px 6px; border-radius:4px;">Ctrl+V</kbd> Görüntü yapıştır</li>
      <li><kbd style="background:var(--bg-main); padding:2px 6px; border-radius:4px;">?</kbd> Kısayolları aç/kapat</li>
    </ul>`;
  document.body.appendChild(m);
}

buildSidebar();
