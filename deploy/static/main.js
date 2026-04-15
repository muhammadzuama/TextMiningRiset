/* ── Theme toggle ──────────────────────────────────────────────────────── */
(function() {
  const saved = localStorage.getItem('kuhp-theme');
  if (saved === 'light') document.body.classList.add('light-mode');
})();

document.getElementById('theme-toggle').addEventListener('click', () => {
  const isLight = document.body.classList.toggle('light-mode');
  localStorage.setItem('kuhp-theme', isLight ? 'light' : 'dark');
});

/* ── Markdown → HTML renderer ──────────────────────────────────────────── */
function renderMarkdown(raw) {
  let md = raw;

  // Escape HTML first to prevent XSS
  md = md
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // --- Block elements (order matters) ---

  // Horizontal rules
  md = md.replace(/^[-*_]{3,}\s*$/gm, '<hr>');

  // Headings: ### ## #
  md = md.replace(/^#{3}\s+(.+)$/gm, '<h3>$1</h3>');
  md = md.replace(/^#{2}\s+(.+)$/gm, '<h2>$1</h2>');
  md = md.replace(/^#{1}\s+(.+)$/gm, '<h1>$1</h1>');

  // Blockquotes
  md = md.replace(/^&gt;\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  // Merge consecutive blockquotes
  md = md.replace(/<\/blockquote>\s*<blockquote>/g, '<br>');

  // // Ordered lists: lines starting with "1. " "2. " etc.
  // md = md.replace(/^(\d+)\.\s+(.+)$/gm, '<li data-ol>$2</li>');

  // // Unordered lists: lines starting with - * +
  // md = md.replace(/^[-*+]\s+(.+)$/gm, '<li>$2</li>');

  // Wrap consecutive <li data-ol> in <ol>
  md = md.replace(/(<li data-ol>.*?<\/li>\s*)+/gs, (match) => {
    const items = match.replace(/ data-ol/g, '');
    return '<ol>' + items + '</ol>';
  });

  // Wrap remaining <li> (not already inside ol) in <ul>
  md = md.replace(/(<li>.*?<\/li>\s*)+/gs, (match) => {
    if (/<ol>/.test(match)) return match;
    return '<ul>' + match + '</ul>';
  });

  // --- Inline elements ---

  // Bold+italic: ***text***
  md = md.replace(/\*{3}(.+?)\*{3}/g, '<strong><em>$1</em></strong>');

  // Bold: **text** or __text__
  md = md.replace(/\*{2}(.+?)\*{2}/g, '<strong>$1</strong>');
  md = md.replace(/_{2}(.+?)_{2}/g, '<strong>$1</strong>');

  // Italic: *text* or _text_
  md = md.replace(/\*(.+?)\*/g, '<em>$1</em>');
  md = md.replace(/_(.+?)_/g, '<em>$1</em>');

  // Inline code
  md = md.replace(/`(.+?)`/g, '<code>$1</code>');

  // --- Paragraphs ---
  // Split on double newlines for paragraphs, but skip block tags
  const blockTags = /^<(h[1-3]|ul|ol|li|hr|blockquote|div)/;
  const lines = md.split(/\n\n+/);
  md = lines.map(chunk => {
    chunk = chunk.trim();
    if (!chunk) return '';
    if (blockTags.test(chunk)) return chunk;
    // Single newlines within a paragraph → line break (but not inside lists)
    chunk = chunk.replace(/\n(?!<li)/g, '<br>');
    return '<p>' + chunk + '</p>';
  }).join('\n');

  // --- Special: Referensi section ---
  // Detect lines like "- Pasal 219 KUHP" inside a reference block
  md = md.replace(
    /(<p>.*?[Rr]eferensi.*?<\/p>)([\s\S]*?)(?=<p>|$)/,
    (_, heading, refs) => {
      const refItems = refs.match(/<p>[-–]\s*(.*?)<\/p>/g) || [];
      if (!refItems.length) return _ ;
      const refHtml = refItems.map(item => {
        const text = item.replace(/<\/?p>/g, '').replace(/^[-–]\s*/, '');
        return `<div class="ref-item"><span class="ref-badge">Ref</span><span>${text}</span></div>`;
      }).join('');
      return heading + '<div class="ref-section">' + refHtml + '</div>';
    }
  );

  return md;
}

/* ── Tab navigation ────────────────────────────────────────────────────── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    if (tab === 'riwayat') loadHistory();
  });
});

/* ── Sample pills ──────────────────────────────────────────────────────── */
document.querySelectorAll('.sample-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    const ta = document.getElementById('question-input');
    ta.value = pill.dataset.q;
    updateCharCount();
    ta.focus();
  });
});

/* ── Char counter ──────────────────────────────────────────────────────── */
const ta = document.getElementById('question-input');
const counter = document.getElementById('char-count');
function updateCharCount() { counter.textContent = ta.value.length; }
ta.addEventListener('input', updateCharCount);

/* ── Refresh LLM model list from Ollama ───────────────────────────────── */
document.getElementById('refresh-llm').addEventListener('click', async () => {
  const btn  = document.getElementById('refresh-llm');
  const hint = document.getElementById('llm-hint');
  const sel  = document.getElementById('llm-select');
  btn.style.opacity = '0.4';
  hint.textContent = 'Memuat daftar model dari Ollama...';
  try {
    const res  = await fetch('/llm-models');
    const list = await res.json();
    const prev = sel.value;
    sel.innerHTML = list.map(m =>
      `<option value="${escHtml(m.name)}">${escHtml(m.name)}${m.description ? ' \u2014 ' + escHtml(m.description) : ''}</option>`
    ).join('');
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
    hint.textContent = `${list.length} model ditemukan di Ollama.`;
  } catch(e) {
    hint.textContent = 'Gagal menghubungi Ollama. Pastikan server berjalan.';
  } finally {
    btn.style.opacity = '1';
  }
});

/* ── State helpers ─────────────────────────────────────────────────────── */
function showState(state) {
  ['idle','loading','content','error'].forEach(s => {
    document.getElementById(`answer-${s}`).classList.toggle('hidden', s !== state);
  });
}

/* ── Loading steps animation ──────────────────────────────────────────── */
let stepTimer = null;
function runLoadingSteps() {
  const steps = ['step-1','step-2','step-3'];
  steps.forEach(id => document.getElementById(id).classList.remove('active','done'));
  document.getElementById('step-1').classList.add('active');
  let i = 0;
  stepTimer = setInterval(() => {
    if (i < steps.length - 1) {
      document.getElementById(steps[i]).classList.remove('active');
      document.getElementById(steps[i]).classList.add('done');
      i++;
      document.getElementById(steps[i]).classList.add('active');
    }
  }, 2500);
}
function stopLoadingSteps() {
  clearInterval(stepTimer);
  ['step-1','step-2','step-3'].forEach(id => {
    document.getElementById(id).classList.remove('active');
    document.getElementById(id).classList.add('done');
  });
}

/* ── Submit ────────────────────────────────────────────────────────────── */
document.getElementById('submit-btn').addEventListener('click', async () => {
  const question = ta.value.trim();
  const modelKey = document.querySelector('input[name="model"]:checked')?.value;
  const llmModel = document.getElementById('llm-select')?.value || 'gemma3:4b';

  if (!question) {
    ta.focus();
    ta.style.borderColor = 'var(--red)';
    setTimeout(() => ta.style.borderColor = '', 1500);
    return;
  }

  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Menganalisis...';
  showState('loading');
  runLoadingSteps();

  const startTime = Date.now();
  try {
    const res  = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model_key: modelKey, llm_model: llmModel })
    });
    const data = await res.json();
    stopLoadingSteps();

    if (!res.ok || data.error) {
      document.getElementById('error-text').textContent = data.error || 'Terjadi kesalahan.';
      showState('error');
      return;
    }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    document.getElementById('meta-llm').textContent    = data.llm_model || llmModel;
    document.getElementById('meta-model').textContent  = data.model_key;
    document.getElementById('meta-time').textContent   = `${elapsed}s`;

    // Render markdown → HTML
    document.getElementById('answer-text').innerHTML = renderMarkdown(data.answer);
    showState('content');

  } catch (err) {
    stopLoadingSteps();
    document.getElementById('error-text').textContent = 'Gagal menghubungi server: ' + err.message;
    showState('error');
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Analisis Hukum';
  }
});

/* ── Copy button ───────────────────────────────────────────────────────── */
document.getElementById('copy-btn').addEventListener('click', () => {
  // Copy plain text (strip HTML tags)
  const text = document.getElementById('answer-text').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Disalin!';
    setTimeout(() => {
      btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Salin`;
    }, 2000);
  });
});

/* ── History ───────────────────────────────────────────────────────────── */
async function loadHistory() {
  const container = document.getElementById('history-container');
  container.innerHTML = '<div class="history-empty">Memuat...</div>';
  try {
    const res  = await fetch('/history');
    const data = await res.json();
    if (!data.rows || data.rows.length === 0) {
      container.innerHTML = '<div class="history-empty">Belum ada riwayat pertanyaan.</div>';
      return;
    }
    container.innerHTML = '';
    [...data.rows].reverse().forEach(row => {
      const card = document.createElement('div');
      card.className = 'history-card';
      const ts       = row.timestamp ? new Date(row.timestamp).toLocaleString('id-ID') : '';
      const answerId = 'ans-' + Math.random().toString(36).slice(2);
      card.innerHTML = `
        <div class="history-card-header">
          <span class="history-badge history-badge-llm">${escHtml(row.model_llm || '-')}</span>
          <span class="history-badge">${escHtml(row.model_embedding || '')}</span>
          <span class="history-ts">${escHtml(ts)}</span>
        </div>
        <div class="history-question">${escHtml(row.question || '')}</div>
        <div class="history-answer" id="${answerId}">${escHtml(row.answer || '')}</div>
        <button class="history-expand">Lihat selengkapnya</button>
      `;
      const ansDiv = card.querySelector(`#${answerId}`);
      const expBtn = card.querySelector('.history-expand');
      let expanded = false;
      expBtn.addEventListener('click', () => {
        expanded = !expanded;
        ansDiv.style.webkitLineClamp = expanded ? 'unset' : '3';
        ansDiv.style.overflow        = expanded ? 'visible' : 'hidden';
        expBtn.textContent = expanded ? 'Sembunyikan' : 'Lihat selengkapnya';
      });
      container.appendChild(card);
    });
  } catch (e) {
    container.innerHTML = '<div class="history-empty">Gagal memuat riwayat.</div>';
  }
}

document.getElementById('refresh-history').addEventListener('click', loadHistory);

/* ── Utilities ─────────────────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}