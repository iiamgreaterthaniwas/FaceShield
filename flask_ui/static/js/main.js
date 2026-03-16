/* ── Tab 切换 ─────────────────────────────── */
document.querySelectorAll('.nav-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const target = tab.dataset.tab;
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + target).classList.add('active');
  });
});

/* ── Radio 按钮组 ─────────────────────────── */
function initRadioGroup(groupId) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.querySelectorAll('.tag').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}
initRadioGroup('algoGroup');
initRadioGroup('expAlgoGroup');

function getRadioVal(groupId) {
  const active = document.querySelector(`#${groupId} .tag.active`);
  return active ? active.dataset.val : 'pgd';
}

/* ── Toggle ───────────────────────────────── */
let adaptiveOn = true;
let compareOn  = false;
let fawkesOn   = false;
let dflOn      = false;

function _syncToggle(id, state) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle('on', state);
}

// main.js 在 </body> 前加载，DOM 已 ready，直接调用无需等事件
function _initToggles() {
  _syncToggle('adaptiveToggle', adaptiveOn);
  _syncToggle('compareToggle',  compareOn);
  _syncToggle('fawkesToggle',   fawkesOn);
  _syncToggle('dflToggle',      dflOn);
}
_initToggles();

function toggleAdaptive() {
  adaptiveOn = !adaptiveOn;
  _syncToggle('adaptiveToggle', adaptiveOn);
}
function toggleCompare() {
  compareOn = !compareOn;
  _syncToggle('compareToggle', compareOn);
}
function toggleFawkes() {
  fawkesOn = !fawkesOn;
  _syncToggle('fawkesToggle', fawkesOn);
}
function toggleDFL() {
  dflOn = !dflOn;
  _syncToggle('dflToggle', dflOn);
}

/* ── 源人脸上传 & 拖拽 ───────────────────── */
const dropZone  = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
let uploadedFile = null;
let _currentTaskId = null;
let _pollTimer = null;
let _previewDebounce = null;

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

function handleFile(file) {
  uploadedFile = file;
  const url = URL.createObjectURL(file);
  dropZone.innerHTML = `<img src="${url}" style="object-fit:cover">`;
  document.getElementById('origResult').innerHTML =
    `<img src="${url}" style="width:100%;height:auto;display:block">`;
  // 上传图片只预览，不自动提交，等用户点击"生成保护图像"
}

/* ── 目标人脸上传（SimSwap 用）────────────── */
const targetDropZone  = document.getElementById('targetDropZone');
const targetFileInput = document.getElementById('targetFileInput');
let targetFile = null;       // null = 使用后端默认目标人脸

if (targetDropZone) {
  targetDropZone.addEventListener('dragover', e => { e.preventDefault(); targetDropZone.classList.add('dragover'); });
  targetDropZone.addEventListener('dragleave', () => targetDropZone.classList.remove('dragover'));
  targetDropZone.addEventListener('drop', e => {
    e.preventDefault();
    targetDropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleTargetFile(file);
  });
  targetDropZone.addEventListener('click', () => targetFileInput.click());
}

if (targetFileInput) {
  targetFileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleTargetFile(e.target.files[0]);
  });
}

function handleTargetFile(file) {
  targetFile = file;
  const url = URL.createObjectURL(file);
  targetDropZone.innerHTML = `<img src="${url}" style="object-fit:cover;width:100%;height:100%;border-radius:8px">`;
  // 目标人脸只更新预览，不自动触发攻击测试
}

function clearTargetFile() {
  targetFile = null;
  if (targetDropZone) {
    targetDropZone.innerHTML = `
      <div class="upload-icon" style="font-size:20px;opacity:0.4">👤</div>
      <div class="upload-hint" style="font-size:11px">点击或拖拽目标人脸<br>（不选则用默认人脸）</div>`;
  }
  // 清除目标人脸，不自动触发
}

/* ── 参数变化只更新滑块显示值，不自动触发保护 */
['epsilon', 'numSteps'].forEach(id => {
  const el = document.getElementById(id);
  // 仅保留原有的 input 显示更新逻辑，无 scheduleAutoPreview
});

/* ── 异步保护主逻辑 ────────────────────────── */
async function runProtect() {
  if (!uploadedFile) { alert('请先上传图片'); return; }

  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }

  const btn = document.getElementById('protectBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> 提交任务...';
  _setProtectProgress(5);

  const dlBtn = document.getElementById('downloadBtn');
  if (dlBtn) dlBtn.style.display = 'none';

  const formData = new FormData();
  formData.append('image',       uploadedFile);
  formData.append('epsilon',     document.getElementById('epsilon').value);
  formData.append('num_steps',   document.getElementById('numSteps').value);
  formData.append('attack_type', getRadioVal('algoGroup'));
  formData.append('adaptive',    adaptiveOn.toString());

  // ← 新增：附加目标人脸（有则上传，无则后端用默认）
  if (targetFile) {
    formData.append('target_image', targetFile);
  }

  try {
    const res  = await fetch('/api/protect_async', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) { alert('提交失败: ' + data.error); _resetProtectBtn(); return; }

    _currentTaskId = data.task_id;
    btn.innerHTML = '<div class="spin"></div> 生成中...';
    _setProtectProgress(20);

    // 轮询任务状态（每 600ms）
    _pollTimer = setInterval(async () => {
      try {
        const poll   = await fetch(`/api/task_status/${_currentTaskId}`);
        const status = await poll.json();
        _setProtectProgress(status.progress || 20);

        // 显示后端状态提示（SimSwap 运行时会有 "SimSwap 换脸中…" 等提示）
        if (status.status_hint) {
          btn.innerHTML = `<div class="spin"></div> ${status.status_hint}`;
        }

        if (status.status === 'done') {
          clearInterval(_pollTimer); _pollTimer = null;
          _onProtectDone(status.result);
        } else if (status.status === 'error') {
          clearInterval(_pollTimer); _pollTimer = null;
          alert('处理失败: ' + status.error);
          _resetProtectBtn();
        }
      } catch (_) { /* 网络抖动时继续轮询 */ }
    }, 600);

  } catch (e) {
    alert('请求失败: ' + e);
    _resetProtectBtn();
  }
}

// 保存最近一次保护结果，供攻击测试使用
let _lastProtectResult = null;

function _onProtectDone(result) {
  _lastProtectResult = result;  // 缓存结果

  setImgResult('advResult',  result.protected);
  setImgResult('pertResult', result.perturbation);
  setImgResult('epsResult',  result.epsilon_map);

  // 下载按钮
  const dlBtn = document.getElementById('downloadBtn');
  if (dlBtn && result.protected) {
    const bytes = Uint8Array.from(atob(result.protected), c => c.charCodeAt(0));
    const blob  = new Blob([bytes], { type: 'image/png' });
    dlBtn.href  = URL.createObjectURL(blob);
    dlBtn.style.display = 'inline-flex';
  }

  // 指标卡
  const m = result.metrics;
  document.getElementById('v-psnr').textContent = m.psnr + ' dB';
  document.getElementById('v-ssim').textContent = m.ssim;
  document.getElementById('v-linf').textContent = m.l_inf;
  document.getElementById('v-time').textContent = m.elapsed + 's';
  const vCos = document.getElementById('v-cos');
  if (vCos && m.id_cos_sim !== undefined) {
    vCos.textContent = m.id_cos_sim;
    const cosCard = document.getElementById('m-cos');
    if (cosCard) cosCard.classList.add(m.id_cos_sim < 0.5 ? 'good' : 'warn');
  }
  colorMetric('m-psnr', m.psnr >= 30);
  colorMetric('m-ssim', m.ssim >= 0.95);
  document.getElementById('metricsCard').style.display = 'block';

  // 换脸攻击区：重置为等待用户点击状态
  document.getElementById('swapCompareRow').style.display = 'none';
  document.getElementById('swapComparePlaceholder').style.display = 'block';
  document.getElementById('swapComparePlaceholder').innerHTML =
    '保护图已生成，点击下方按钮开始换脸攻击测试';
  const attackBtn = document.getElementById('attackTestBtn');
  if (attackBtn) attackBtn.style.display = 'inline-flex';
  const modeBadge = document.getElementById('swapModeBadge');
  if (modeBadge) modeBadge.style.display = 'none';

  _setProtectProgress(100);
  _resetProtectBtn();
}

/* ── 换脸攻击测试（独立触发，真正调用后端）─────────────── */
async function runAttackTest() {
  if (!_currentTaskId) { alert('请先生成保护图像'); return; }

  const btn = document.getElementById('attackTestBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> 换脸测试中...';

  // 隐藏旧结果，显示占位提示
  document.getElementById('swapCompareRow').style.display = 'none';
  document.getElementById('swapComparePlaceholder').style.display = 'block';
  document.getElementById('swapComparePlaceholder').innerHTML =
    '<div class="spin" style="display:inline-block;margin-right:6px"></div> SimSwap 换脸中，请稍候...';

  try {
    const formData = new FormData();
    formData.append('task_id', _currentTaskId);
    // 若用户此时有新目标人脸可一并上传（复用 targetFile）
    if (targetFile) {
      formData.append('target_image', targetFile);
    }

    const res  = await fetch('/api/attack_test', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      document.getElementById('swapComparePlaceholder').innerHTML =
        '⚠️ 换脸测试失败: ' + data.error;
      btn.disabled = false;
      btn.innerHTML = '🔁 重新测试';
      return;
    }

    // 展示换脸对比图
    if (data.swap_orig && data.swap_protected) {
      setImgResult('swapOrigResult', data.swap_orig);
      setImgResult('swapProtResult', data.swap_protected);
      document.getElementById('swapCompareRow').style.display = 'grid';
      document.getElementById('swapComparePlaceholder').style.display = 'none';

      const modeBadge = document.getElementById('swapModeBadge');
      if (modeBadge) {
        modeBadge.textContent = data.swap_mode === 'simswap'
          ? '✅ SimSwap' : '⚠️ 热力图模式（SimSwap 未运行）';
        modeBadge.className = data.swap_mode === 'simswap'
          ? 'badge badge-green' : 'badge badge-gold';
        modeBadge.style.display = 'inline-flex';
      }
      const origCaption = document.getElementById('swapOrigCaption');
      const protCaption = document.getElementById('swapProtCaption');
      if (data.swap_mode === 'simswap') {
        if (origCaption) origCaption.textContent = '原图换脸结果（身份被提取 ✗）';
        if (protCaption) protCaption.textContent = '保护图换脸结果（身份被干扰 ✓）';
      } else {
        if (origCaption) origCaption.textContent = '原图（绿框=正常输入）';
        if (protCaption) protCaption.textContent = '保护图（热力图=扰动区域）';
      }
    } else {
      document.getElementById('swapComparePlaceholder').innerHTML =
        '⚠️ 换脸数据不可用（SimSwap 未配置或运行失败）';
    }
  } catch (e) {
    document.getElementById('swapComparePlaceholder').innerHTML =
      '⚠️ 请求失败: ' + e;
  }

  btn.disabled = false;
  btn.innerHTML = '🔁 重新测试';
}

function _setProtectProgress(pct) {
  const bar  = document.getElementById('protectProgressBar');
  const text = document.getElementById('protectProgressText');
  if (bar)  { bar.style.width = pct + '%'; bar.parentElement.style.display = 'block'; }
  if (text) { text.style.display = 'block'; text.textContent = pct < 100 ? `处理中 ${pct}%` : '✅ 完成'; }
}

function _resetProtectBtn() {
  const btn = document.getElementById('protectBtn');
  btn.disabled = false;
  btn.innerHTML = '<span>🚀</span> 生成保护图像';
}

function setImgResult(id, base64) {
  document.getElementById(id).innerHTML =
    `<img src="data:image/png;base64,${base64}" style="width:100%;height:auto;display:block">`;
}

function colorMetric(id, good) {
  const el = document.getElementById(id);
  el.classList.remove('good', 'warn');
  el.classList.add(good ? 'good' : 'warn');
}

/* ── 数据集准备 ───────────────────────────── */
function appendLog(termId, text) {
  const term = document.getElementById(termId);
  let colored = text
    .replace(/(✅[^\n]*)/g, '<span class="log-success">$1</span>')
    .replace(/(❌[^\n]*)/g, '<span class="log-error">$1</span>')
    .replace(/(⚠️[^\n]*)/g, '<span class="log-warn">$1</span>')
    .replace(/(🚀[^\n]*)/g, '<span class="log-accent">$1</span>')
    .replace(/(\[[\d\/]+\][^\n]*)/g, '<span class="log-accent">$1</span>');
  term.innerHTML += colored;
  term.scrollTop = term.scrollHeight;
}

async function prepareDataset() {
  const src        = document.getElementById('srcFolder').value.trim();
  const imgSize    = parseInt(document.getElementById('prepImgSize').value);
  const maxSamples = parseInt(document.getElementById('maxSamples').value);

  if (!src) { alert('请输入文件夹路径'); return; }

  const log = document.getElementById('datasetLog');
  log.innerHTML = '';
  document.getElementById('prepProgressTrack').style.display = 'block';
  document.getElementById('prepProgressText').style.display  = 'block';
  document.getElementById('prepProgressBar').style.width = '0%';

  try {
    const res = await fetch('/api/prepare_dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ src_folder: src, img_size: imgSize, max_samples: maxSamples }),
    });

    const reader  = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer    = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const msg = line.slice(6).trim();
        if (!msg) continue;

        if (msg.startsWith('PROGRESS:')) {
          const parts = msg.split(':');
          const pct   = parseInt(parts[1]);
          const info  = parts[2] || '';
          document.getElementById('prepProgressBar').style.width = pct + '%';
          document.getElementById('prepProgressText').textContent = '进度 ' + pct + '%  ' + info;
        } else if (msg === 'DONE') {
          document.getElementById('prepProgressBar').style.width = '100%';
          document.getElementById('prepProgressText').textContent = '✅ 处理完成';
        } else {
          appendLog('datasetLog', msg + '\n');
        }
      }
    }
  } catch (e) {
    appendLog('datasetLog', '❌ 请求失败: ' + e + '\n');
  }
}

async function checkDataset() {
  const res  = await fetch('/api/dataset_status');
  const data = await res.json();
  const card = document.getElementById('datasetStatusCard');
  const cont = document.getElementById('datasetStatusContent');
  card.style.display = 'block';

  if (data.ready) {
    cont.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:8px">
        <div style="display:flex;justify-content:space-between">
          <span style="color:var(--text2);font-size:13px">状态</span>
          <span class="badge badge-green">✅ 就绪</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:var(--text2);font-size:13px">图片数量</span>
          <span style="font-family:var(--mono);color:var(--accent)">${data.count} 张</span>
        </div>
        <div style="display:flex;justify-content:space-between">
          <span style="color:var(--text2);font-size:13px">图片尺寸</span>
          <span style="font-family:var(--mono);color:var(--text)">${data.size}</span>
        </div>
        <div style="font-family:var(--mono);font-size:11px;color:var(--text3);margin-top:4px">${data.path}</div>
      </div>`;
  } else {
    cont.innerHTML = `<span class="badge badge-red">❌ ${data.message}</span>`;
  }
}

/* ── 批量实验 ─────────────────────────────── */
let expEventSource   = null;
let _experimentRunning = false;

async function runExperiment() {
  if (expEventSource) expEventSource.close();

  const btn = document.getElementById('expBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> 运行中...';
  document.getElementById('expRunningBadge').style.display = 'inline-flex';
  document.getElementById('expLog').innerHTML = '';
  _experimentRunning = true;
  document.getElementById('expProgressBar').style.width = '5%';

  const payload = {
    epsilon:        parseInt(document.getElementById('expEpsilon').value),
    num_steps:      parseInt(document.getElementById('expSteps').value),
    attack_type:    getRadioVal('expAlgoGroup'),
    num_samples:    parseInt(document.getElementById('expN').value),
    compare_mode:   compareOn,
    include_fawkes: fawkesOn,
    include_dfl:    dflOn,
  };

  const startRes = await fetch('/api/run_experiment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  if (!startRes.ok) {
    const err = await startRes.json();
    alert('启动失败: ' + err.error);
    btn.disabled = false;
    btn.innerHTML = '🚀 开始实验';
    return;
  }

  let progress  = 5;
  let retryCount = 0;
  const MAX_RETRY = 20;

  function connectSSE() {
    expEventSource = new EventSource('/api/experiment_log');

    expEventSource.onmessage = e => {
      const msg = e.data;
      retryCount = 0;

      if (msg === 'DONE') {
        expEventSource.close();
        _experimentRunning = false;
        btn.disabled = false;
        btn.innerHTML = '开始实验';
        document.getElementById('expRunningBadge').style.display = 'none';
        document.getElementById('expProgressBar').style.width = '100%';
        appendLog('expLog', '\n✅ 实验完成，正在自动刷新图表...\n');
        // 自动刷新图表，延迟500ms确保后端文件写入完成
        setTimeout(() => {
          loadResults();
          // 自动切换到实验结果 Tab
          document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
          document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
          const resultTab = document.querySelector('.nav-tab[data-tab="results"]');
          const resultPanel = document.getElementById('tab-results');
          if (resultTab) resultTab.classList.add('active');
          if (resultPanel) resultPanel.classList.add('active');
        }, 500);
        return;
      }
      if (msg === 'HEARTBEAT') return;

      appendLog('expLog', msg);
      if (msg.includes('Batch') && progress < 95) {
        progress = Math.min(progress + 2, 95);
        document.getElementById('expProgressBar').style.width = progress + '%';
      }
    };

    expEventSource.onerror = () => {
      expEventSource.close();
      retryCount++;
      if (retryCount <= MAX_RETRY && _experimentRunning) {
        setTimeout(connectSSE, 1500);
      } else {
        _experimentRunning = false;
        btn.disabled = false;
        btn.innerHTML = '开始实验';
        document.getElementById('expRunningBadge').style.display = 'none';
      }
    };
  }

  connectSSE();
}

/* ── 查看结果 ─────────────────────────────── */
async function loadResults() {
  const res  = await fetch('/api/results');
  const data = await res.json();

  if (data.line_chart) {
    document.getElementById('lineChartSlot').innerHTML =
      `<img src="data:image/png;base64,${data.line_chart}" class="chart-img">`;
  }
  if (data.heatmap) {
    document.getElementById('heatmapSlot').innerHTML =
      `<img src="data:image/png;base64,${data.heatmap}" class="chart-img">`;
  }
  const fawkesSlot = document.getElementById('fawkesChartSlot');
  if (fawkesSlot) {
    if (data.fawkes_chart) {
      fawkesSlot.closest('.card').style.display = 'block';
      fawkesSlot.innerHTML = `<img src="data:image/png;base64,${data.fawkes_chart}" class="chart-img">`;
    } else {
      fawkesSlot.closest('.card').style.display = 'none';
    }
  }
  const dflSlot = document.getElementById('dflChartSlot');
  if (dflSlot) {
    if (data.dfl_chart) {
      dflSlot.closest('.card').style.display = 'block';
      dflSlot.innerHTML = `<img src="data:image/png;base64,${data.dfl_chart}" class="chart-img">`;
    } else {
      dflSlot.closest('.card').style.display = 'none';
    }
  }
  if (data.files && data.files.length > 0) {
    document.getElementById('resultFileList').innerHTML =
      data.files.map(f => `<div style="padding:4px 0;border-bottom:1px solid var(--border)">📄 ${f}</div>`).join('');
  } else {
    document.getElementById('resultFileList').textContent = '暂无结果文件，请先运行实验';
  }
}

/* ── 性能测试 ─────────────────────────────── */
async function runPerfTest() {
  const btn = document.getElementById('perfBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spin"></div> 测试中...';

  try {
    const res  = await fetch('/api/perf_test', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ n: parseInt(document.getElementById('perfN').value) }),
    });
    const data = await res.json();
    document.getElementById('perf-avg').textContent    = data.avg + 's';
    document.getElementById('perf-fps').textContent    = data.fps;
    document.getElementById('perf-total').textContent  = data.total + 's';
    document.getElementById('perf-device').textContent = data.device;
    document.getElementById('perfResultCard').style.display = 'block';
  } catch (e) {
    alert('测试失败: ' + e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ 开始测试';
  }
}