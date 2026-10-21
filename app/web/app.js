/* ==========================================================================
 * audio_processer · 本地 Web UI（原生 JS，无框架 / 无构建 / 无外部依赖）
 *
 * 只调用固定 API 契约：
 *   GET    /api/health
 *   GET    /api/jobs
 *   POST   /api/jobs                      （JSON {"path": "data/"} 扫描）
 *   POST   /api/jobs?filename=x.m4a       （application/octet-stream 裸流上传）
 *   POST   /api/jobs/{id}/run|retry
 *   GET    /api/jobs/{id}                 （job/report/stages/artifacts/transcript/corrections）
 *   GET    /api/jobs/{id}/audio           （clean.m4a，支持 Range，直接喂 <audio>）
 *   GET    /api/jobs/{id}/export?format=md|srt|txt
 *   POST   /api/jobs/{id}/transcript      （{"segments":[{"index":0,"text":"..."}]}）
 *   GET    /api/events                    （SSE：event: snapshot，data: {summary,jobs}）
 *
 * 安全约定：所有来自服务端/用户的文本一律走 textContent / value / title 属性，
 * 绝不把 user data 拼进 innerHTML。
 * ========================================================================== */

(function () {
  'use strict';

  /* ---------------------------------------------------------------- 常量 */

  var PAGE_SIZE = 100;
  var FALLBACK_POLL_MS = 8000;
  var HEALTH_POLL_MS = 30000;

  var STAGES = ['decode', 'analyze', 'denoise', 'trim', 'loudness',
                'asr', 'diarize', 'correct', 'export'];

  var STATUS_LABELS = {
    pending: '待处理',
    running: '运行中',
    done: '已完成',
    failed: '失败',
    skipped: '已跳过'
  };

  /* ---------------------------------------------------------------- 状态 */

  var state = {
    jobs: [],
    summary: null,
    selectedId: null,
    detail: null,
    transcript: [],
    edits: new Map(),        // index -> 修改后的 text（仅未保存的脏数据）
    page: 0,
    saving: false,
    loadingDetail: false,
    loadToken: 0,
    lastListSig: '',
    lastDetailJobSig: '',
    sseOpen: false,
    sseSource: null,
    toastTimer: null
  };

  /* ---------------------------------------------------------------- DOM 小工具 */

  function $(id) {
    return document.getElementById(id);
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clearNode(node) {
    if (!node) return;
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function setText(node, text) {
    if (node) node.textContent = text === undefined || text === null ? '' : String(text);
  }

  function toInt(value) {
    var n = Number(value);
    return isFinite(n) ? Math.trunc(n) : null;
  }

  function toNumber(value) {
    var n = Number(value);
    return isFinite(n) ? n : null;
  }

  function firstLine(text) {
    var s = String(text === undefined || text === null ? '' : text);
    return s.split(/\r?\n/)[0];
  }

  function truncate(text, limit) {
    var s = String(text === undefined || text === null ? '' : text);
    if (s.length <= limit) return s;
    return s.slice(0, Math.max(0, limit - 1)) + '…';
  }

  function baseName(path) {
    var s = String(path === undefined || path === null ? '' : path);
    var parts = s.split(/[\\/]/);
    return parts[parts.length - 1] || '';
  }

  function shortId(id) {
    var s = String(id === undefined || id === null ? '' : id);
    var head = s.split('_')[0];
    return head || s.slice(0, 8);
  }

  function statusKey(status) {
    var s = String(status === undefined || status === null ? '' : status);
    return Object.prototype.hasOwnProperty.call(STATUS_LABELS, s) ? s : 'pending';
  }

  /* ---------------------------------------------------------------- 格式化 */

  function pad2(n) {
    return String(n).padStart(2, '0');
  }

  // 秒 -> 12:34 / 1:02:03（用于转写时间轴与时长）
  function fmtClock(value) {
    var sec = Math.max(0, Math.floor(Number(value) || 0));
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    var s = sec % 60;
    if (h > 0) return h + ':' + pad2(m) + ':' + pad2(s);
    return m + ':' + pad2(s);
  }

  // Unix 时间戳（秒或毫秒）或 ISO 字符串 -> 本地 YYYY-MM-DD HH:mm
  function fmtDateTime(value) {
    var date = null;
    if (typeof value === 'number' && isFinite(value)) {
      date = new Date(value < 1e12 ? value * 1000 : value);
    } else if (typeof value === 'string' && value) {
      var parsed = new Date(value);
      if (!isNaN(parsed.getTime())) date = parsed;
    }
    if (!date || isNaN(date.getTime())) return '';
    return date.getFullYear() + '-' + pad2(date.getMonth() + 1) + '-' + pad2(date.getDate()) +
      ' ' + pad2(date.getHours()) + ':' + pad2(date.getMinutes());
  }

  // updated_at - created_at（任务耗时）
  function fmtElapsed(job) {
    if (!job) return '';
    var start = toInt(job.created_at);
    var end = toInt(job.updated_at);
    if (start === null || end === null) return '';
    var sec = Math.max(0, end - start);
    if (sec < 60) return sec + ' 秒';
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    var s = sec % 60;
    if (h > 0) return h + ' 小时 ' + m + ' 分';
    return m + ' 分 ' + s + ' 秒';
  }

  function fmtBytes(value) {
    var n = Number(value);
    if (!isFinite(n) || n < 0) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    if (n < 1024 * 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + ' MB';
    return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  /* ---------------------------------------------------------------- Toast / 徽标 */

  function toast(message, kind) {
    var node = $('toast');
    if (!node) return;
    node.textContent = String(message === undefined || message === null ? '' : message);
    node.className = 'toast' + (kind ? ' toast-' + kind : '');
    node.hidden = false;
    if (state.toastTimer) clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(function () {
      node.hidden = true;
    }, kind === 'error' ? 7000 : 3800);
  }

  function setBadge(node, kind, text) {
    if (!node) return;
    node.className = 'badge badge-' + kind;
    node.textContent = text;
  }

  function makeBadge(text, kind, title) {
    var node = el('span', 'badge badge-' + kind, text);
    if (title) node.title = String(title);
    return node;
  }

  /* ---------------------------------------------------------------- HTTP */

  function apiFetch(path, options) {
    var opts = options || {};
    return fetch(path, opts).then(function (res) {
      return res.text().then(function (body) {
        var data = null;
        if (body) {
          try { data = JSON.parse(body); } catch (err) { data = null; }
        }
        if (!res.ok) {
          var detail = data && (data.detail || data.error || data.message);
          var message = '';
          if (typeof detail === 'string') message = detail;
          else if (detail) {
            try { message = JSON.stringify(detail); } catch (err) { message = String(detail); }
          }
          var error = new Error('HTTP ' + res.status + (message ? '：' + message : ''));
          error.status = res.status;
          throw error;
        }
        return data;
      });
    }, function () {
      var error = new Error('无法连接本地服务，请确认 start_web.bat 已启动');
      error.status = 0;
      throw error;
    });
  }

  /* ---------------------------------------------------------------- 汇总 / 任务列表 */

  function normalizeCounts(summary) {
    var source = summary && typeof summary === 'object' ? (summary.counts || summary) : {};
    var counts = { pending: 0, running: 0, done: 0, failed: 0, skipped: 0 };
    Object.keys(counts).forEach(function (key) {
      var value = toInt(source[key]);
      counts[key] = value === null ? 0 : value;
    });
    var total = toInt(source.total);
    counts.total = total === null ? counts.pending + counts.running + counts.done +
      counts.failed + counts.skipped : total;
    return counts;
  }

  function renderSummary() {
    var counts = normalizeCounts(state.summary);
    var box = $('summaryChips');
    if (!box) return;
    clearNode(box);
    var defs = [
      ['pending', '待处理'],
      ['running', '运行中'],
      ['done', '已完成'],
      ['failed', '失败']
    ];
    defs.forEach(function (pair) {
      var key = pair[0];
      var chip = el('span', 'chip chip-' + key);
      chip.appendChild(el('span', null, pair[1]));
      chip.appendChild(el('b', null, String(counts[key])));
      chip.title = pair[1] + '：' + counts[key];
      box.appendChild(chip);
    });
    setText($('jobTotal'), String(counts.total));
  }

  function jobSignature(job) {
    return [job.id, job.status, job.stage, job.updated_at, job.error].join('|');
  }

  function renderJobList() {
    var list = $('jobList');
    if (!list) return;
    clearNode(list);

    if (!state.jobs.length) {
      var empty = el('div', 'list-empty');
      empty.appendChild(el('div', null, '还没有任务'));
      empty.appendChild(el('div', null, '输入目录后点「扫描登记」，或点「上传音频…」。'));
      list.appendChild(empty);
      return;
    }

    state.jobs.forEach(function (job) {
      list.appendChild(renderJobItem(job));
    });
  }

  function renderJobItem(job) {
    var status = statusKey(job.status);
    var active = job.id === state.selectedId;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'job-item' + (active ? ' is-active' : '');
    btn.setAttribute('role', 'option');
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
    btn.dataset.jobId = job.id;
    btn.title = job.id + (job.error ? '\n错误：' + job.error : '');

    var row = el('div', 'job-row');
    row.appendChild(el('span', 'status-dot status-' + status));
    row.appendChild(el('span', 'job-id', '#' + shortId(job.id)));
    row.appendChild(el('span', 'job-duration', fmtElapsed(job)));
    btn.appendChild(row);

    btn.appendChild(el('div', 'job-item-title', baseName(job.source) || job.id));

    var sub = el('div', 'job-item-sub');
    sub.appendChild(el('span', null, STATUS_LABELS[status] || status));
    if (job.stage) {
      sub.appendChild(el('span', null, '·'));
      sub.appendChild(el('span', 'stage-name', String(job.stage)));
    }
    var attempts = toInt(job.attempts) || 0;
    if (attempts > 0) sub.appendChild(el('span', null, '· 第 ' + attempts + ' 次'));
    btn.appendChild(sub);

    if (job.error) {
      btn.appendChild(el('div', 'job-item-error', truncate(firstLine(job.error), 60)));
    }

    btn.addEventListener('click', function () {
      selectJob(job.id);
    });
    return btn;
  }

  /* ---------------------------------------------------------------- 详情视图切换 */

  function setDetailView(mode, message) {
    var empty = $('detailEmpty');
    var loading = $('detailLoading');
    var error = $('detailError');
    var body = $('detailBody');
    if (empty) empty.hidden = mode !== 'empty';
    if (loading) loading.hidden = mode !== 'loading';
    if (error) error.hidden = mode !== 'error';
    if (body) body.hidden = mode !== 'body';
    if (mode === 'error') setText($('detailErrorMsg'), message || '未知错误');

    if (mode === 'error' || mode === 'loading') {
      var player = $('player');
      if (player) {
        player.pause();
        player.removeAttribute('src');
        try { player.load(); } catch (err) { /* ignore */ }
        delete player.dataset.jobId;
      }
    }
  }

  function selectJob(id, options) {
    var opts = options || {};
    if (!id) return;
    if (state.selectedId === id && !opts.force) return;

    if (state.edits.size > 0) {
      var ok = window.confirm('当前任务有 ' + state.edits.size + ' 处未保存的修改，确定切换任务吗？');
      if (!ok) {
        renderJobList();
        return;
      }
    }

    state.selectedId = id;
    state.detail = null;
    state.transcript = [];
    state.edits.clear();
    state.page = 0;
    state.loadToken += 1;
    state.lastDetailJobSig = '';
    setText($('saveStatus'), '');

    renderJobList();
    setDetailView('loading');
    loadDetail(id, { reset: true, force: true });

    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, '', '#' + encodeURIComponent(id));
    }
  }

  function clearSelection() {
    state.selectedId = null;
    state.detail = null;
    state.transcript = [];
    state.edits.clear();
    state.page = 0;
    state.lastDetailJobSig = '';
    setText($('saveStatus'), '');
    renderJobList();
    setDetailView('empty');
    if (window.history && window.history.replaceState) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  }

  /* ---------------------------------------------------------------- 详情加载与渲染 */

  function loadDetail(id, options) {
    var opts = options || {};
    if (!id) return Promise.resolve();
    if (opts.silent && state.loadingDetail) return Promise.resolve();

    var token = ++state.loadToken;
    state.loadingDetail = true;
    if (!opts.silent && !opts.reset) setDetailView('loading');

    return apiFetch('/api/jobs/' + encodeURIComponent(id)).then(function (data) {
      if (token !== state.loadToken || state.selectedId !== id) return;
      state.detail = data && typeof data === 'object' ? data : {};
      state.transcript = Array.isArray(state.detail.transcript) ? state.detail.transcript : [];

      // 丢弃失效的未保存修改（任务/段表被外部改动时）
      state.edits.forEach(function (value, index) {
        var seg = state.transcript[index];
        if (!seg || String(seg.text) === String(value)) state.edits.delete(index);
      });

      var pageCount = Math.max(1, Math.ceil(state.transcript.length / PAGE_SIZE));
      if (opts.reset) state.page = 0;
      state.page = Math.min(Math.max(0, state.page), pageCount - 1);

      var job = state.detail.job || {};
      state.lastDetailJobSig = [job.status, job.stage, job.updated_at].join('|');

      setDetailView('body');
      renderDetail();
    }).catch(function (err) {
      if (token !== state.loadToken || state.selectedId !== id) return;
      if (opts.silent) {
        toast('刷新详情失败：' + err.message, 'error');
        return;
      }
      setDetailView('error', err.message);
    }).then(function () {
      if (token === state.loadToken) state.loadingDetail = false;
    });
  }

  function renderDetail() {
    var detail = state.detail || {};
    var job = detail.job || {};
    var artifacts = Array.isArray(detail.artifacts) ? detail.artifacts : [];

    renderJobHeader(job);
    renderStageChips(detail, job);
    renderAudio(job, artifacts);
    renderReportLine(detail.report, detail.corrections);
    renderArtifacts(artifacts);
    renderExports(job, artifacts);
    renderTranscript();
    renderPagination();
    updateDirtyState();
  }

  function renderJobHeader(job) {
    var id = job.id || state.selectedId || '';
    setText($('jobTitle'), baseName(job.source) || id);
    setText($('jobSource'), job.source ? '源文件：' + job.source : '任务：' + id);

    var badges = $('jobBadges');
    if (!badges) return;
    clearNode(badges);

    var status = statusKey(job.status);
    var kindMap = { pending: 'muted', running: 'info', done: 'ok', failed: 'error', skipped: 'muted' };
    badges.appendChild(makeBadge(STATUS_LABELS[status] || String(job.status || '未知'), kindMap[status]));
    if (id) badges.appendChild(makeBadge('#' + shortId(id), 'muted', id));

    if (job.stage) badges.appendChild(makeBadge('阶段：' + job.stage, 'info'));
    var elapsed = fmtElapsed(job);
    if (elapsed) badges.appendChild(makeBadge('耗时：' + elapsed, 'muted'));
    var attempts = toInt(job.attempts) || 0;
    if (attempts > 0) badges.appendChild(makeBadge('尝试 ' + attempts + ' 次', 'warn'));

    var created = fmtDateTime(job.created_at);
    if (created) badges.appendChild(makeBadge('创建 ' + created, 'muted'));
    var updated = fmtDateTime(job.updated_at);
    if (updated) badges.appendChild(makeBadge('更新 ' + updated, 'muted'));

    if (job.error) {
      var errorBadge = makeBadge('错误：' + truncate(firstLine(job.error), 90), 'error', job.error);
      badges.appendChild(errorBadge);
    }

    var running = status === 'running';
    var runBtn = $('runBtn');
    var retryBtn = $('retryBtn');
    if (runBtn) {
      runBtn.disabled = running || status === 'done';
      runBtn.textContent = running ? '运行中…' : '开始处理';
    }
    if (retryBtn) retryBtn.disabled = running;
  }

  function renderStageChips(detail, job) {
    var box = $('stageChips');
    if (!box) return;
    clearNode(box);

    var statusMap = {};
    function absorb(source) {
      if (!source) return;
      if (Array.isArray(source)) {
        source.forEach(function (item) {
          if (!item) return;
          var name = item.stage || item.name;
          if (!name) return;
          var value = item.status;
          if (typeof value !== 'string') value = (item.status && item.status.status) || 'pending';
          statusMap[name] = value;
        });
      } else if (typeof source === 'object') {
        Object.keys(source).forEach(function (name) {
          var value = source[name];
          if (typeof value === 'string') statusMap[name] = value;
          else if (value && typeof value === 'object') statusMap[name] = value.status || 'pending';
        });
      }
    }
    absorb(detail && detail.stages);
    absorb(job && job.stages);

    var names = Object.keys(statusMap);
    if (!names.length) {
      box.appendChild(el('span', 'muted small', '暂无阶段记录'));
      return;
    }
    names.sort(function (a, b) {
      var ia = STAGES.indexOf(a);
      var ib = STAGES.indexOf(b);
      if (ia < 0) ia = STAGES.length;
      if (ib < 0) ib = STAGES.length;
      if (ia !== ib) return ia - ib;
      return a < b ? -1 : a > b ? 1 : 0;
    });

    names.forEach(function (name) {
      var status = statusKey(statusMap[name]);
      var chip = el('span', 'stage-chip stage-' + status,
        name + ' · ' + (STATUS_LABELS[status] || status));
      chip.title = name + '：' + String(statusMap[name]);
      box.appendChild(chip);
    });
  }

  function renderAudio(job, artifacts) {
    var player = $('player');
    if (!player) return;
    var hasClean = (artifacts || []).some(function (item) {
      return item && item.name === 'clean.m4a';
    });
    var hint = $('audioHint');
    var jobId = job.id || state.selectedId;

    if (hasClean && jobId) {
      player.hidden = false;
      if (player.dataset.jobId !== jobId) {
        player.dataset.jobId = jobId;
        player.src = '/api/jobs/' + encodeURIComponent(jobId) + '/audio';
        try { player.load(); } catch (err) { /* ignore */ }
      }
      if (hint) hint.textContent = 'clean.m4a（降噪裁剪版）';
    } else {
      player.pause();
      player.removeAttribute('src');
      delete player.dataset.jobId;
      try { player.load(); } catch (err) { /* ignore */ }
      player.hidden = true;
      if (hint) {
        hint.textContent = job.status === 'running'
          ? '音频生成中…（跑完 render 阶段后刷新）'
          : '音频尚未生成，跑完流水线后可用';
      }
    }
  }

  function renderReportLine(report, corrections) {
    var box = $('reportLine');
    if (!box) return;
    clearNode(box);

    var items = [];
    if (report && typeof report === 'object') {
      var duration = toNumber(report.duration_s);
      if (duration !== null) items.push(['素材时长', fmtClock(duration)]);
      if (report.output && typeof report.output === 'object') {
        var outDuration = toNumber(report.output.duration_s);
        if (outDuration !== null) items.push(['成品时长', fmtClock(outDuration)]);
        var size = toNumber(report.output.size_bytes);
        if (size !== null) items.push(['成品大小', fmtBytes(size)]);
      }
      if (report.asr && typeof report.asr === 'object') {
        var segs = toInt(report.asr.n_segments);
        var chars = toInt(report.asr.chars);
        var rtf = toNumber(report.asr.rtf);
        var asrText = [];
        if (segs !== null) asrText.push(segs + ' 段');
        if (chars !== null) asrText.push(chars + ' 字');
        if (rtf !== null) asrText.push('RTF ' + rtf.toFixed(3));
        if (asrText.length) items.push(['ASR', asrText.join(' / ')]);
      }
      if (report.speakers && typeof report.speakers === 'object') {
        var n = toInt(report.speakers.n);
        if (n !== null) items.push(['说话人', n + ' 人（' + (report.speakers.method || 'none') + '）']);
      }
      if (report.correct && typeof report.correct === 'object') {
        if (report.correct.enabled) {
          var edits = toInt(report.correct.edits);
          var rejected = toInt(report.correct.rejected);
          var cost = toNumber(report.correct.cost_cny);
          var parts = [(edits === null ? 0 : edits) + ' 处'];
          if (rejected) parts.push('回退 ' + rejected + ' 块');
          if (cost !== null) parts.push('¥' + cost.toFixed(3));
          items.push(['纠错', parts.join(' / ')]);
        } else {
          items.push(['纠错', '未启用（无 key 自动降级，不阻塞）']);
        }
      }
    }
    if (corrections && typeof corrections === 'object') {
      var total = toInt(corrections.total);
      if (total !== null) items.push(['纠错条目', total + ' 条']);
    }

    if (!items.length) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    items.forEach(function (pair) {
      var item = el('span', 'report-item');
      item.appendChild(el('span', null, pair[0] + '：'));
      item.appendChild(el('b', null, pair[1]));
      box.appendChild(item);
    });
  }

  function renderArtifacts(artifacts) {
    var list = $('artifactList');
    if (!list) return;
    clearNode(list);

    if (!Array.isArray(artifacts) || !artifacts.length) {
      list.appendChild(el('li', 'artifact-empty', '暂无产物'));
      return;
    }
    artifacts.forEach(function (item) {
      if (!item) return;
      var li = el('li', 'artifact-item');
      li.appendChild(el('span', 'artifact-name', String(item.name || '?')));
      var size = toNumber(item.size);
      li.appendChild(el('span', 'artifact-size', size === null ? '' : fmtBytes(size)));
      var mtime = fmtDateTime(item.mtime);
      if (mtime) li.title = '修改时间：' + mtime;
      list.appendChild(li);
    });
  }

  function setExportLink(node, format, jobId, enabled) {
    if (!node) return;
    if (enabled && jobId) {
      node.href = '/api/jobs/' + encodeURIComponent(jobId) + '/export?format=' + format;
      node.removeAttribute('aria-disabled');
      node.style.pointerEvents = '';
      node.style.opacity = '';
    } else {
      node.removeAttribute('href');
      node.setAttribute('aria-disabled', 'true');
      node.style.pointerEvents = 'none';
      node.style.opacity = '0.45';
    }
  }

  function renderExports(job, artifacts) {
    var jobId = job.id || state.selectedId;
    var names = new Set((artifacts || []).map(function (item) {
      return item && item.name ? String(item.name) : '';
    }));
    setExportLink($('exportMd'), 'md', jobId, names.has('transcript.md'));
    setExportLink($('exportSrt'), 'srt', jobId, names.has('transcript.srt'));
    setExportLink($('exportTxt'), 'txt', jobId, names.has('transcript.txt'));
  }

  /* ---------------------------------------------------------------- 转写表格 */

  function transcriptRow(seg, index) {
    var tr = document.createElement('tr');

    var tdTime = document.createElement('td');
    var timeBtn = el('button', 'time-btn', '[' + fmtClock(seg.start) + ']');
    timeBtn.type = 'button';
    timeBtn.dataset.index = String(index);
    var title = '跳转播放：' + fmtClock(seg.start) + ' → ' + fmtClock(seg.end);
    if (seg.orig_start !== undefined && seg.orig_start !== null) {
      title += '（原片 ' + fmtClock(seg.orig_start) + '）';
    }
    timeBtn.title = title;
    tdTime.appendChild(timeBtn);
    tr.appendChild(tdTime);

    var tdSpeaker = document.createElement('td');
    var speaker = el('span', 'speaker-name', seg.speaker === undefined || seg.speaker === null
      ? '—' : String(seg.speaker));
    tdSpeaker.appendChild(speaker);
    tr.appendChild(tdSpeaker);

    var tdRaw = document.createElement('td');
    tdRaw.appendChild(el('div', 'raw-text', seg.text_raw === undefined || seg.text_raw === null
      ? '' : String(seg.text_raw)));
    if (Array.isArray(seg.flags) && seg.flags.length) {
      var flagNames = seg.flags.map(function (flag) { return String(flag); });
      var flagBadge = el('span', 'flag-badge', '已标记');
      flagBadge.title = 'flags：' + flagNames.join(', ');
      tdRaw.appendChild(flagBadge);
    }
    tr.appendChild(tdRaw);

    var tdText = document.createElement('td');
    var textarea = document.createElement('textarea');
    textarea.className = 'edit-text';
    textarea.rows = 2;
    textarea.spellcheck = false;
    textarea.dataset.index = String(index);
    var edited = state.edits.has(index);
    textarea.value = edited ? String(state.edits.get(index)) : String(seg.text === undefined || seg.text === null ? '' : seg.text);
    if (edited) textarea.classList.add('is-dirty');
    tdText.appendChild(textarea);
    tr.appendChild(tdText);

    return tr;
  }

  function autoSizeTextarea(node) {
    if (!node || node.offsetParent === null) return;
    node.style.height = 'auto';
    var next = Math.max(46, Math.min(node.scrollHeight + 2, 160));
    node.style.height = next + 'px';
  }

  function renderTranscript() {
    var body = $('transcriptBody');
    if (!body) return;
    clearNode(body);

    var total = state.transcript.length;
    setText($('transcriptCount'), total ? '· ' + total + ' 段' : '');

    var start = state.page * PAGE_SIZE;
    var end = Math.min(start + PAGE_SIZE, total);
    for (var i = start; i < end; i += 1) {
      body.appendChild(transcriptRow(state.transcript[i], i));
    }

    var textareas = body.querySelectorAll('textarea.edit-text');
    Array.prototype.forEach.call(textareas, autoSizeTextarea);

    var empty = $('transcriptEmpty');
    if (empty) empty.hidden = total > 0;
    var wrap = document.querySelector('.table-wrap');
    if (wrap) wrap.hidden = total === 0;
    document.querySelectorAll('[data-page-role="bar"]').forEach(function (bar) {
      bar.hidden = total === 0;
    });
  }

  function pageWindow(current, pageCount) {
    var wanted = {};
    if (pageCount <= 9) {
      for (var i = 0; i < pageCount; i += 1) wanted[i] = true;
    } else {
      [0, pageCount - 1, current - 2, current - 1, current, current + 1, current + 2]
        .forEach(function (page) {
          if (page >= 0 && page < pageCount) wanted[page] = true;
        });
    }
    return Object.keys(wanted).map(Number).sort(function (a, b) { return a - b; });
  }

  function buildPageButtons(container, current, pageCount) {
    clearNode(container);
    var pages = pageWindow(current, pageCount);
    var previous = -1;
    pages.forEach(function (page) {
      if (previous >= 0 && page - previous > 1) {
        container.appendChild(el('span', 'page-gap', '…'));
      }
      var btn = el('button', 'page-btn' + (page === current ? ' is-current' : ''), String(page + 1));
      btn.type = 'button';
      btn.dataset.page = String(page);
      container.appendChild(btn);
      previous = page;
    });
  }

  function renderPagination() {
    var total = state.transcript.length;
    var pageCount = Math.max(1, Math.ceil(total / PAGE_SIZE));
    var current = Math.min(Math.max(0, state.page), pageCount - 1);

    document.querySelectorAll('[data-page-role="prev"]').forEach(function (btn) {
      btn.disabled = current <= 0;
    });
    document.querySelectorAll('[data-page-role="next"]').forEach(function (btn) {
      btn.disabled = current >= pageCount - 1;
    });
    var info = total ? '第 ' + (current + 1) + ' / ' + pageCount + ' 页' : '第 0 / 0 页';
    document.querySelectorAll('[data-page-role="info"]').forEach(function (node) {
      node.textContent = info;
    });
    document.querySelectorAll('[data-page-role="total"]').forEach(function (node) {
      node.textContent = '共 ' + total + ' 行 · 每页 ' + PAGE_SIZE + ' 行';
    });
    document.querySelectorAll('[data-page-role="pages"]').forEach(function (node) {
      buildPageButtons(node, current, pageCount);
    });
  }

  function goToPage(page) {
    var pageCount = Math.max(1, Math.ceil(state.transcript.length / PAGE_SIZE));
    var next = Math.min(Math.max(0, toInt(page) || 0), pageCount - 1);
    if (next === state.page) return;
    state.page = next;
    renderTranscript();
    renderPagination();
  }

  function updateDirtyState() {
    var count = state.edits.size;
    var hint = $('saveHint');
    var saveBtn = $('saveBtn');
    if (hint) hint.hidden = count === 0;
    if (saveBtn) {
      saveBtn.disabled = count === 0 || state.saving;
      saveBtn.textContent = state.saving
        ? '保存中…'
        : (count ? '保存修改 (' + count + ')' : '保存修改');
    }
  }

  function seekAudio(start) {
    var player = $('player');
    if (!player || player.hidden || !player.src) {
      toast('音频尚未生成，暂不能跳转播放', 'warn');
      return;
    }
    var target = Math.max(0, Number(start) || 0);
    var apply = function () {
      try { player.currentTime = target; } catch (err) { /* ignore */ }
      var played = player.play();
      if (played && typeof played.catch === 'function') played.catch(function () { /* 自动播放被拦截时忽略 */ });
    };
    if (player.readyState >= 1) apply();
    else player.addEventListener('loadedmetadata', apply, { once: true });
  }

  /* ---------------------------------------------------------------- 操作 */

  function startJob(mode) {
    var id = state.selectedId;
    if (!id) return;
    var isRetry = mode === 'retry';
    var path = '/api/jobs/' + encodeURIComponent(id) + (isRetry ? '/retry' : '/run');
    var runBtn = $('runBtn');
    var retryBtn = $('retryBtn');
    if (runBtn) runBtn.disabled = true;
    if (retryBtn) retryBtn.disabled = true;

    apiFetch(path, { method: 'POST' }).then(function () {
      toast(isRetry ? '已提交重试，任务即将开始' : '已开始处理，进度会实时刷新', 'ok');
      return refreshJobs({ silent: true }).then(function () {
        return loadDetail(id, { silent: true, force: true });
      });
    }).catch(function (err) {
      var message = err.status === 409 ? '任务正在运行中，请稍候' : err.message;
      toast((isRetry ? '重试失败：' : '启动失败：') + message, 'error');
      if (state.detail && state.detail.job) renderJobHeader(state.detail.job);
    });
  }

  function saveEdits() {
    if (!state.selectedId || state.edits.size === 0 || state.saving) return;
    var jobId = state.selectedId;
    var changes = [];
    state.edits.forEach(function (text, index) {
      changes.push({ index: index, text: String(text) });
    });
    state.saving = true;
    updateDirtyState();
    setText($('saveStatus'), '保存中…');

    apiFetch('/api/jobs/' + encodeURIComponent(jobId) + '/transcript', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segments: changes })
    }).then(function () {
      state.edits.clear();
      var message = '已保存 ' + changes.length + ' 段修改，md / srt / txt 已重新导出';
      toast(message, 'ok');
      setText($('saveStatus'), '已保存 ' + changes.length + ' 段');
      return refreshJobs({ silent: true }).then(function () {
        return loadDetail(jobId, { silent: true, force: true });
      });
    }).catch(function (err) {
      setText($('saveStatus'), '保存失败');
      var message = err.status === 409 ? '任务正在运行中，暂不能修改转写' : err.message;
      toast('保存失败：' + message, 'error');
    }).then(function () {
      state.saving = false;
      updateDirtyState();
    });
  }

  function doScan() {
    var input = $('dirInput');
    var dir = input ? input.value.trim() : '';
    if (!dir) {
      toast('请输入要扫描的本地目录，例如 data/', 'warn');
      if (input) input.focus();
      return;
    }
    var btn = $('scanBtn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = '扫描中…';
    }

    apiFetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: dir })
    }).then(function (data) {
      var created = toInt(data && data.created) || 0;
      var existing = toInt(data && data.existing) || 0;
      toast('扫描完成：新登记 ' + created + ' 个，已存在 ' + existing + ' 个',
        created > 0 ? 'ok' : 'warn');
      return refreshJobs();
    }).catch(function (err) {
      toast('扫描失败：' + err.message, 'error');
    }).then(function () {
      if (btn) {
        btn.disabled = false;
        btn.textContent = '扫描登记';
      }
    });
  }

  function uploadFiles(files) {
    if (!files || !files.length) return;
    var btn = $('uploadBtn');
    var list = Array.prototype.slice.call(files);
    var lastJobId = null;

    if (btn) {
      btn.disabled = true;
      btn.textContent = '上传中 1/' + list.length + '…';
    }

    function uploadOne(index) {
      if (index >= list.length) {
        if (btn) {
          btn.disabled = false;
          btn.textContent = '上传音频…';
        }
        refreshJobs().then(function () {
          if (lastJobId) selectJob(lastJobId, { force: true });
        });
        return;
      }
      var file = list[index];
      if (btn) btn.textContent = '上传中 ' + (index + 1) + '/' + list.length + '…';

      apiFetch('/api/jobs?filename=' + encodeURIComponent(file.name), {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: file
      }).then(function (data) {
        if (data && data.job_id) lastJobId = data.job_id;
        toast('已上传：' + file.name, 'ok');
      }).catch(function (err) {
        toast('上传失败（' + file.name + '）：' + err.message, 'error');
      }).then(function () {
        uploadOne(index + 1);
      });
    }

    uploadOne(0);
  }

  /* ---------------------------------------------------------------- 数据刷新 / SSE */

  function applyJobsPayload(payload, options) {
    var opts = options || {};
    var data = payload && typeof payload === 'object' ? payload : {};
    state.jobs = Array.isArray(data.jobs) ? data.jobs : [];
    state.summary = data.summary || null;

    var signature = state.jobs.map(jobSignature).join('~');
    if (!opts.skipIfSame || signature !== state.lastListSig) {
      state.lastListSig = signature;
      renderSummary();
      renderJobList();
    }

    if (!state.selectedId) return;

    var selected = null;
    state.jobs.forEach(function (job) {
      if (job.id === state.selectedId) selected = job;
    });
    if (!selected) {
      clearSelection();
      return;
    }

    if (!state.detail) return;
    var detailSig = [selected.status, selected.stage, selected.updated_at].join('|');
    if (detailSig !== state.lastDetailJobSig && !state.loadingDetail) {
      loadDetail(selected.id, { silent: true });
    }
  }

  function refreshJobs(options) {
    return apiFetch('/api/jobs').then(function (data) {
      applyJobsPayload(data, options);
      return data;
    }).catch(function (err) {
      if (!options || !options.silent) toast('刷新任务列表失败：' + err.message, 'error');
      return null;
    });
  }

  function refreshHealth() {
    return apiFetch('/api/health').then(function (data) {
      var version = data && data.version !== undefined && data.version !== null
        ? String(data.version) : '';
      setBadge($('healthBadge'), 'ok', '服务正常' + (version ? ' · v' + version : ''));
    }).catch(function () {
      setBadge($('healthBadge'), 'error', '服务不可用');
    });
  }

  function handleSnapshot(payload) {
    state.sseOpen = true;
    setBadge($('sseBadge'), 'ok', '实时：已连接');
    applyJobsPayload(payload, { skipIfSame: true });
  }

  function connectSSE() {
    setBadge($('sseBadge'), 'muted', '实时：连接中…');
    if (typeof window.EventSource !== 'function') {
      setBadge($('sseBadge'), 'warn', '实时：浏览器不支持');
      return;
    }
    var source = new window.EventSource('/api/events');
    state.sseSource = source;

    source.addEventListener('open', function () {
      state.sseOpen = true;
      setBadge($('sseBadge'), 'ok', '实时：已连接');
    });
    source.addEventListener('snapshot', function (event) {
      var payload = null;
      try { payload = JSON.parse(event.data); } catch (err) { return; }
      handleSnapshot(payload);
    });
    source.addEventListener('error', function () {
      state.sseOpen = false;
      setBadge($('sseBadge'), 'warn', '实时：重连中…');
    });
  }

  /* ---------------------------------------------------------------- 事件绑定 / 启动 */

  function bindEvents() {
    var scanBtn = $('scanBtn');
    if (scanBtn) scanBtn.addEventListener('click', doScan);

    var dirInput = $('dirInput');
    if (dirInput) {
      dirInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter') doScan();
      });
    }

    var uploadBtn = $('uploadBtn');
    var fileInput = $('fileInput');
    if (uploadBtn && fileInput) {
      uploadBtn.addEventListener('click', function () { fileInput.click(); });
      fileInput.addEventListener('change', function () {
        var files = fileInput.files ? Array.prototype.slice.call(fileInput.files) : [];
        fileInput.value = '';
        uploadFiles(files);
      });
    }

    var refreshBtn = $('refreshBtn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', function () {
        refreshHealth();
        refreshJobs().then(function (data) {
          if (data && state.selectedId) loadDetail(state.selectedId, { silent: true, force: true });
        });
      });
    }

    var runBtn = $('runBtn');
    if (runBtn) runBtn.addEventListener('click', function () { startJob('run'); });
    var retryBtn = $('retryBtn');
    if (retryBtn) retryBtn.addEventListener('click', function () { startJob('retry'); });

    var detailRetryBtn = $('detailRetryBtn');
    if (detailRetryBtn) {
      detailRetryBtn.addEventListener('click', function () {
        if (state.selectedId) loadDetail(state.selectedId, { force: true });
      });
    }

    var saveBtn = $('saveBtn');
    if (saveBtn) saveBtn.addEventListener('click', saveEdits);

    document.querySelectorAll('[data-page-role="prev"]').forEach(function (btn) {
      btn.addEventListener('click', function () { goToPage(state.page - 1); });
    });
    document.querySelectorAll('[data-page-role="next"]').forEach(function (btn) {
      btn.addEventListener('click', function () { goToPage(state.page + 1); });
    });

    document.addEventListener('click', function (event) {
      var target = event.target;
      if (!target || !target.closest) return;
      var pageBtn = target.closest('.page-btn[data-page]');
      if (!pageBtn) return;
      var page = toInt(pageBtn.dataset.page);
      if (page !== null) goToPage(page);
    });

    var body = $('transcriptBody');
    if (body) {
      body.addEventListener('click', function (event) {
        var target = event.target;
        if (!target || !target.closest) return;
        var timeBtn = target.closest('.time-btn');
        if (!timeBtn) return;
        var index = toInt(timeBtn.dataset.index);
        if (index === null || !state.transcript[index]) return;
        seekAudio(state.transcript[index].start);
      });

      body.addEventListener('input', function (event) {
        var target = event.target;
        if (!target || !target.classList || !target.classList.contains('edit-text')) return;
        var index = toInt(target.dataset.index);
        var seg = index === null ? null : state.transcript[index];
        if (!seg) return;
        var value = String(target.value);
        var original = String(seg.text === undefined || seg.text === null ? '' : seg.text);
        if (value === original) {
          state.edits.delete(index);
          target.classList.remove('is-dirty');
        } else {
          state.edits.set(index, value);
          target.classList.add('is-dirty');
        }
        updateDirtyState();
      });
    }

    window.addEventListener('beforeunload', function (event) {
      if (state.edits.size > 0) {
        event.preventDefault();
        event.returnValue = '';
      }
    });
  }

  function restoreFromHash() {
    var hash = String(window.location.hash || '').replace(/^#/, '');
    if (!hash) return;
    var id = '';
    try { id = decodeURIComponent(hash); } catch (err) { id = hash; }
    var exists = state.jobs.some(function (job) { return job.id === id; });
    if (exists) selectJob(id, { force: true });
  }

  function init() {
    bindEvents();
    connectSSE();

    var list = $('jobList');
    if (list) list.appendChild(el('div', 'list-empty', '正在加载任务列表…'));

    refreshHealth();
    refreshJobs().then(function () {
      restoreFromHash();
    });

    // 兜底轮询：SSE 掉线时也能看到进度（EventSource 自己也会重连）
    setInterval(function () {
      if (!state.sseOpen) refreshJobs({ silent: true });
    }, FALLBACK_POLL_MS);
    setInterval(refreshHealth, HEALTH_POLL_MS);
  }

  init();
})();
