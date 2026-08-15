/* Nitkellmu — Speak Maltese
   Single-page client. No build step, no dependencies. */

const $ = (id) => document.getElementById(id);

const state = {
  caps: null,
  settings: { voice: 'mt-MT-GraceNeural', rate: 0.95, show_english: true, autoplay: true },
  queue: [],
  qIndex: 0,
  card: null,
  revealed: false,
  attempted: false,
};

/* ── API helpers ──────────────────────────────────────────────────────── */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body instanceof FormData ? {} : { 'content-type': 'application/json' },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(detail);
  }
  return res.headers.get('content-type')?.includes('application/json') ? res.json() : res;
}

const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) });

function toast(msg, ms = 3200) {
  const el = $('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, ms);
}

function escapeHtml(s = '') {
  return s.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* ── Audio ─────────────────────────────────────────────────────────────── */

let currentAudio = null;

function speak(text, { rate } = {}) {
  if (!text) return Promise.resolve();
  const r = rate ?? state.settings.rate;
  const url = `/api/tts?text=${encodeURIComponent(text)}&rate=${r}&voice=${encodeURIComponent(state.settings.voice)}`;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  const audio = new Audio(url);
  currentAudio = audio;
  return audio.play().catch((err) => {
    // Autoplay policies block the first sound until the user interacts.
    if (err.name !== 'NotAllowedError') toast('Audio unavailable — check TTS setup');
  });
}

/* ── Recording ─────────────────────────────────────────────────────────── */

/* The microphone stream is acquired once and kept open for the session.
   getUserMedia costs 100-500ms, and previously that was paid *after* the button
   was pressed — so the first syllable was simply never recorded. That is what
   turned "Bonġu" into "onġi" and "Silas" into "salas": the recogniser was not
   mishearing the onset, it never received it. */
let sharedStream = null;

async function ensureStream() {
  if (sharedStream?.active) return sharedStream;
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Microphone not available');
  sharedStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  return sharedStream;
}

/** Warm the mic on the first interaction so the first recording isn't the slow one. */
function prewarmMic() {
  ensureStream().catch(() => { /* permission comes later, on first real use */ });
}

class Recorder {
  constructor() { this.chunks = []; this.rec = null; }

  async start() {
    const stream = await ensureStream();
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      .find((m) => MediaRecorder.isTypeSupported(m)) || '';
    this.chunks = [];
    this.rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    this.rec.ondataavailable = (e) => { if (e.data.size) this.chunks.push(e.data); };
    // Small timeslice so audio is flushed continuously rather than in one lump
    // at stop, which shortens the gap between releasing and having a blob.
    this.rec.start(200);
    this.startedAt = performance.now();
  }

  async stop() {
    if (!this.rec) return null;
    const done = new Promise((resolve) => { this.rec.onstop = resolve; });
    // Let the tail of the final word land before cutting the recorder.
    await new Promise((r) => setTimeout(r, 250));
    this.rec.stop();
    await done;
    // Keep the stream open for the next utterance — only the recorder stops.
    const blob = new Blob(this.chunks, { type: this.rec.mimeType || 'audio/webm' });
    const ms = performance.now() - this.startedAt;
    this.rec = null;
    return ms < 350 || blob.size < 1200 ? null : blob;
  }
}

async function transcribe(blob, target) {
  const fd = new FormData();
  fd.append('audio', blob, 'speech.webm');
  if (target) fd.append('target', target);
  return api('/api/stt', { method: 'POST', body: fd });
}

/** Wires push-to-talk (hold) and tap-to-toggle onto a mic button. */
function bindMic(button, { onResult, onStatus, target }) {
  const recorder = new Recorder();
  let active = false;
  let holdTimer = null;
  let isHold = false;

  const setBusy = (busy) => button.classList.toggle('is-busy', busy);

  async function begin() {
    if (active || button.classList.contains('is-busy')) return;
    try {
      await recorder.start();
      active = true;
      button.classList.add('is-recording');
      onStatus?.('Listening… (release to send)');
    } catch (err) {
      toast(err.message);
    }
  }

  async function end() {
    if (!active) return;
    active = false;
    button.classList.remove('is-recording');
    setBusy(true);
    onStatus?.('Transcribing…');
    try {
      const blob = await recorder.stop();
      if (!blob) { onStatus?.('Too short — hold a little longer'); return; }
      const result = await transcribe(blob, typeof target === 'function' ? target() : target);
      onStatus?.('');
      await onResult(result);
    } catch (err) {
      onStatus?.('');
      toast(`Could not transcribe: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  let pendingStop = false;

  button.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (active) { pendingStop = true; return; }   // second tap of a toggle
    isHold = false;
    pendingStop = false;
    // Start capturing immediately. Waiting even 140ms here clipped the first
    // syllable off every utterance; hold-vs-tap is resolved on release instead,
    // which costs no audio.
    begin();
    holdTimer = setTimeout(() => { isHold = true; }, 140);
  });

  const release = () => {
    clearTimeout(holdTimer);
    if (pendingStop) { pendingStop = false; end(); return; }
    if (isHold) { end(); return; }   // held down: push-to-talk, release sends
    // Quick tap: keep recording until the next tap.
  };
  button.addEventListener('pointerup', (e) => { e.preventDefault(); release(); });
  button.addEventListener('pointerleave', () => { if (active && isHold) end(); });

  return { begin, end, isActive: () => active };
}

/* ── Boot ──────────────────────────────────────────────────────────────── */

async function boot() {
  let data;
  try {
    data = await api('/api/bootstrap');
  } catch (err) {
    toast(`Backend unreachable: ${err.message}`, 8000);
    return;
  }
  state.caps = data.capabilities;
  state.settings = { ...state.settings, ...data.settings };

  applySettings();
  renderCaps();
  updateCounts(data.counts);
  $('levelChip').textContent = data.profile.level;

  await loadDrills();
  loadGrammar();
}

function applySettings() {
  $('voiceSelect').value = state.settings.voice;
  $('rateRange').value = state.settings.rate;
  $('rateLabel').textContent = `${Number(state.settings.rate).toFixed(2)}×`;
  $('showEnglish').checked = state.settings.show_english;
  $('autoplay').checked = state.settings.autoplay;
}

function renderCaps() {
  const c = state.caps;
  const mark = (ok) => (ok ? '<span class="ok">✓</span>' : '<span class="off">✗</span>');
  $('capsBox').innerHTML = `
    <div>${mark(c.tts.length)} <b>Speech out</b> — ${c.tts.join(', ') || 'none'}</div>
    <div>${mark(c.stt.length)} <b>Speech in</b> — ${c.stt.join(', ') || 'none'}</div>`;
}

function updateCounts(counts) {
  if (!counts) return;
  const badge = $('dueBadge');
  const n = counts.due + Math.min(counts.new, 12);
  badge.textContent = n;
  badge.hidden = n === 0;
}

/* ── Drill: scripted conversation ──────────────────────────────────────────
   No model in the loop. The reply is chosen by phonetic match on the server in
   under a millisecond, and its audio is already cached, so the only wait is
   speech recognition. */

const drill = { dialogue: null, node: null, busy: false, attempts: 0 };

async function loadDrills() {
  const { dialogues } = await api('/api/drill/dialogues');
  const sel = $('drillSelect');
  sel.innerHTML = dialogues
    .map((d) => `<option value="${d.id}">${d.name} — ${d.name_en} · ${d.level}</option>`)
    .join('');
  sel.onchange = () => startDrill(sel.value);
  if (!drill.dialogue) await startDrill(dialogues[0]?.id);
}

async function startDrill(id) {
  if (!id) return;
  drill.dialogue = id;
  $('drillChat').innerHTML = '';
  const node = await post('/api/drill/start', { dialogue: id });
  presentDrillNode(node);
}

function presentDrillNode(node) {
  drill.node = node.node;
  drill.attempts = 0;
  $('drillExpect').textContent = node.expect_en ? `→ ${node.expect_en}` : '';
  drillBubble('tutor', node.say_mt, node.say_en);
  if (state.settings.autoplay) speak(node.say_mt);
}

function drillBubble(role, mt, en, extraClass = '') {
  const el = document.createElement('div');
  el.className = `turn ${role} ${extraClass}`;
  el.innerHTML = `
    <div class="bubble">
      <p class="mt">${escapeHtml(mt || '')}</p>
      ${en ? `<p class="en" ${state.settings.show_english ? '' : 'hidden'}>${escapeHtml(en)}</p>` : ''}
      ${role === 'tutor' && mt ? `<div class="bubble-tools">
          <button class="tool" data-play>🔊 Play</button>
          <button class="tool" data-slow>🐢 Slow</button>
        </div>` : ''}
    </div>`;
  if (role === 'tutor' && mt) {
    el.querySelector('[data-play]').onclick = () => speak(mt);
    el.querySelector('[data-slow]').onclick = () => speak(mt, { rate: 0.7 });
  }
  $('drillChat').append(el);
  $('drillChat').scrollTop = $('drillChat').scrollHeight;
  return el;
}

async function answerDrill(said) {
  said = (said || '').trim();
  if (!said || drill.busy) return;
  drill.busy = true;
  $('drillInput').value = '';
  drillBubble('user', said, '');

  try {
    const t0 = performance.now();
    const r = await post('/api/drill/answer', {
      dialogue: drill.dialogue, node: drill.node, said, attempts: drill.attempts,
    });
    if (!r.advance) drill.attempts += 1;
    const ms = Math.round(performance.now() - t0);

    const tone = r.moved_on ? 'near' : { correct: 'ok', close: 'near', wrong: 'bad' }[r.verdict];
    const mark = r.moved_on ? '→' : { correct: '✓', close: '≈', wrong: '✗' }[r.verdict];
    const el = drillBubble('tutor', r.reply_mt, r.reply_en);
    el.querySelector('.bubble').insertAdjacentHTML('afterbegin',
      `<p class="drill-verdict ${tone}">${mark} ${r.moved_on ? 'moving on' : r.verdict} · ${Math.round(r.score * 100)}% · ${ms}ms</p>`);

    if (r.say_this_mt) {
      el.querySelector('.bubble').insertAdjacentHTML('beforeend',
        `<p class="drill-target">${escapeHtml(r.say_this_mt)}
           <em>${escapeHtml(r.say_this_en || '')}</em></p>`);
    }

    if (state.settings.autoplay) await speak(r.reply_mt);

    if (r.advance && r.next) {
      setTimeout(() => presentDrillNode(r.next), 450);
    } else if (r.finished) {
      $('drillExpect').textContent = '';
      drillBubble('tutor', 'Spiċċajna. Prosit!', 'We’re done. Well done!');
      updateCounts((await api('/api/bootstrap')).counts);
    }
  } catch (err) {
    toast(err.message);
  } finally {
    drill.busy = false;
  }
}

/* ── Review ────────────────────────────────────────────────────────────── */

async function loadQueue() {
  const data = await api('/api/queue?limit=25');
  state.queue = data.cards;
  state.qIndex = 0;
  updateCounts(data.counts);
  showCard();
}

function showCard() {
  const empty = $('reviewEmpty');
  const card = $('reviewCard');
  if (state.qIndex >= state.queue.length) {
    state.card = null;
    card.hidden = true;
    empty.hidden = false;
    empty.querySelector('h2').textContent = state.queue.length
      ? 'Sew! Spiċċajt.' : "Xejn x'tirrepeti bħalissa";
    empty.querySelector('p').textContent = state.queue.length
      ? 'Session done — everything due has been reviewed. Come back later, or start a conversation.'
      : 'Nothing due right now. Have a conversation — new words you meet there get scheduled automatically — or pull in some new material below.';
    return;
  }

  empty.hidden = true;
  card.hidden = false;
  state.revealed = false;
  state.attempted = false;

  const c = state.queue[state.qIndex];
  state.card = c;

  $('cardMode').textContent = { produce: 'say it', recognise: 'recall', listen: 'listen' }[c.mode] || c.mode;
  $('cardTopic').textContent = c.topic || '';
  $('cardState').textContent = c.state === 'new' ? 'new' : c.state;
  $('cardProgress').textContent = `${state.qIndex + 1} / ${state.queue.length}`;

  const labels = {
    produce: 'Say this in Maltese',
    recognise: 'What does this mean?',
    listen: 'Listen, then repeat',
  };
  $('promptLabel').textContent = labels[c.mode] || '';

  // Front of the card depends on the retrieval direction being trained.
  if (c.mode === 'produce') {
    $('cardPrompt').textContent = c.en;
    $('cardSub').textContent = c.kind === 'phrase' ? 'whole phrase' : (c.pos || '');
    $('cardPlay').hidden = true;
  } else if (c.mode === 'recognise') {
    $('cardPrompt').textContent = c.mt;
    $('cardSub').textContent = '';
    $('cardPlay').hidden = false;
  } else {
    $('cardPrompt').textContent = c.mt;
    $('cardSub').textContent = c.literal ? `literally: ${c.literal}` : '';
    $('cardPlay').hidden = false;
    if (state.settings.autoplay) speak(c.mt);
  }

  $('cardAnswer').hidden = true;
  $('attemptBox').hidden = true;
  $('gradeRow').hidden = true;
  $('revealBtn').hidden = false;
  $('speakRow').hidden = c.mode === 'recognise';
  $('cardInput').value = '';

  for (const g of [1, 2, 3, 4]) $(`int${g}`).textContent = c.intervals?.[g] || '';
}

function reveal() {
  if (!state.card || state.revealed) return;
  state.revealed = true;
  const c = state.card;
  $('answerMt').textContent = c.mt;
  $('answerEn').textContent = c.en;
  $('answerNote').textContent = [c.literal ? `literally: ${c.literal}` : '', c.note || '']
    .filter(Boolean).join(' · ');
  $('answerExample').textContent = c.example_mt
    ? `${c.example_mt} — ${c.example_en || ''}` : '';
  $('cardAnswer').hidden = false;
  $('gradeRow').hidden = false;
  $('revealBtn').hidden = true;
  $('speakRow').hidden = true;
  if (state.settings.autoplay && !state.attempted) speak(c.mt);
}

async function gradeAttempt(said) {
  const c = state.card;
  if (!c || !said?.trim()) return;
  state.attempted = true;
  const a = await post('/api/attempt', { said, target: c.mt });
  // A chunk ending in "-" is a fused article (mill-, id-); keep it glued to the next.
  $('attemptDiff').innerHTML = a.diff.map((d, i) => {
    const prev = a.diff[i - 1];
    const glue = prev && (prev.target || prev.said).endsWith('-') ? '' : ' ';
    let html;
    if (d.op === 'equal') html = `<span class="eq">${escapeHtml(d.target)}</span>`;
    else if (d.op === 'sub') html = `<span class="bad">${escapeHtml(d.said)}</span> <span class="good">${escapeHtml(d.target)}</span>`;
    else if (d.op === 'del') html = `<span class="bad">${escapeHtml(d.said)}</span>`;
    else html = `<span class="miss">${escapeHtml(d.target)}</span>`;
    return (i ? glue : '') + html;
  }).join('');
  const pct = Math.round(a.score * 100);
  $('attemptVerdict').textContent = {
    perfect: `Perfetta! ${pct}% — nothing to fix.`,
    close: `Qrib ħafna — ${pct}%. Check the highlighted words.`,
    partial: `${pct}%. Listen again and have another go.`,
    off: `${pct}%. Let's hear the model version.`,
  }[a.verdict];
  $('attemptBox').hidden = false;
  reveal();
  if (a.score < 0.85) speak(c.mt, { rate: 0.75 });
  // Pre-select the auto-grade so a single click confirms it.
  document.querySelectorAll('.grade').forEach((b) => {
    b.style.outline = Number(b.dataset.grade) === a.grade ? `2px solid var(--sea)` : '';
  });
}

async function submitGrade(grade) {
  const c = state.card;
  if (!c) return;
  const said = $('cardInput').value.trim() || undefined;
  try {
    const res = await post('/api/review', {
      card_id: c.id, grade, mode: c.mode, said,
    });
    updateCounts(res.counts);
    // A lapsed card comes back later in the same session.
    if (grade === 1) state.queue.push({ ...c, intervals: c.intervals });
  } catch (err) {
    toast(err.message);
  }
  document.querySelectorAll('.grade').forEach((b) => { b.style.outline = ''; });
  state.qIndex += 1;
  showCard();
}

/* ── Progress ──────────────────────────────────────────────────────────── */

async function loadStats() {
  const s = await api('/api/stats');
  const speakPct = s.speaking.attempts
    ? Math.round((s.speaking.correct / s.speaking.attempts) * 100) : 0;
  const cards = [
    ['Words learned', s.learned],
    ['Solid (3wk+)', s.solid],
    ['Due now', s.due],
    ['Not started', s.new],
    ['Day streak', s.streak],
    ['Spoken accuracy', `${speakPct}%`],
  ];
  $('statGrid').innerHTML = cards
    .map(([label, v]) => `<div class="stat"><b>${v}</b><span>${label}</span></div>`).join('');

  const max = Math.max(1, ...s.history.map((h) => h.n));
  $('historyChart').innerHTML = s.history
    .map((h) => `<i style="height:${Math.max(4, (h.n / max) * 100)}%" title="${h.d}: ${h.n} reviews, ${Math.round((h.retention || 0) * 100)}% recalled"></i>`)
    .join('') || '<span style="color:var(--text-dim);font-size:.85rem">No reviews yet.</span>';

  $('topicList').innerHTML = s.topics.map((t) => {
    const pct = t.total ? Math.round((t.learned / t.total) * 100) : 0;
    return `<div class="topic-row">
        <span>${escapeHtml(t.topic)}</span>
        <span class="topic-bar"><i style="width:${pct}%"></i></span>
        <span class="num">${t.learned}/${t.total}</span>
      </div>`;
  }).join('');

  $('weakList').innerHTML = s.weak.length
    ? s.weak.map((w) => `<li><span class="mt">${escapeHtml(w.mt)}</span><span class="n">${escapeHtml(w.en)} · ${w.lapses} slips</span></li>`).join('')
    : '<li class="n">Nothing sticky yet.</li>';

  $('errorList').innerHTML = s.error_kinds.length
    ? s.error_kinds.map((e) => `<li><span>${escapeHtml(e.kind)}</span><span class="n">${e.n}</span></li>`).join('')
    : '<li class="n">No corrections logged yet.</li>';
}

/* ── Reference ─────────────────────────────────────────────────────────── */

async function loadGrammar() {
  try {
    const { markdown } = await api('/api/grammar');
    $('grammarPane').innerHTML = renderMarkdown(markdown);
  } catch {
    $('grammarPane').textContent = 'Reference unavailable.';
  }
}

/** Deliberately small markdown subset — enough for data/grammar_notes.md. */
function renderMarkdown(md) {
  const lines = md.split('\n');
  const out = [];
  let inTable = false;
  let inList = null;
  let inComment = false;

  const inline = (s) => escapeHtml(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[\s(])\*([^*]+)\*/g, '$1<em>$2</em>');

  const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = null; } };
  const closeTable = () => { if (inTable) { out.push('</tbody></table>'); inTable = false; } };

  for (const raw of lines) {
    const line = raw.trimEnd();

    // Authoring notes live in HTML comments and are not for the learner.
    if (inComment) { if (line.includes('-->')) inComment = false; continue; }
    if (line.trimStart().startsWith('<!--')) {
      if (!line.includes('-->')) inComment = true;
      continue;
    }

    if (/^\|[\s:|-]+\|$/.test(line)) continue;                    // table separator
    if (line.startsWith('|')) {
      const cells = line.slice(1, -1).split('|').map((c) => c.trim());
      if (!inTable) {
        closeList();
        out.push(`<table><thead><tr>${cells.map((c) => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>`);
        inTable = true;
      } else {
        out.push(`<tr>${cells.map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`);
      }
      continue;
    }
    closeTable();

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }

    const ol = line.match(/^(\d+)\.\s+(.*)$/);
    if (ol) {
      if (inList !== 'ol') { closeList(); out.push('<ol>'); inList = 'ol'; }
      out.push(`<li>${inline(ol[2])}</li>`);
      continue;
    }
    const ul = line.match(/^[-*]\s+(.*)$/);
    if (ul) {
      if (inList !== 'ul') { closeList(); out.push('<ul>'); inList = 'ul'; }
      out.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }

    if (!line.trim()) { closeList(); continue; }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  return out.join('\n');
}

/* ── Wiring ────────────────────────────────────────────────────────────── */

function switchView(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  if (name === 'review' && !state.card) loadQueue().catch((e) => toast(e.message));
  if (name === 'progress') loadStats().catch((e) => toast(e.message));
  if (name === 'drill' && !drill.dialogue) loadDrills().catch((e) => toast(e.message));
}

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => switchView(t.dataset.view));
});


$('drillSend').addEventListener('click', () => answerDrill($('drillInput').value));
$('drillInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') answerDrill($('drillInput').value);
});
$('drillRestart').addEventListener('click', () => startDrill(drill.dialogue));

bindMic($('drillMic'), {
  onStatus: (s) => { $('drillStatus').textContent = s || 'Hold the mic and answer'; },
  onResult: async (res) => {
    if (!res.text) { toast('Nothing heard — try again'); return; }
    await answerDrill(res.text);
  },
});

$('cardPlay').addEventListener('click', () => state.card && speak(state.card.mt));
$('revealBtn').addEventListener('click', reveal);
$('skipBtn').addEventListener('click', reveal);
$('startNewBtn').addEventListener('click', () => loadQueue().catch((e) => toast(e.message)));

$('cardInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') gradeAttempt($('cardInput').value);
});

bindMic($('cardMic'), {
  target: () => state.card?.mt,
  onStatus: () => {},
  onResult: async (res) => {
    $('cardInput').value = res.text;
    await gradeAttempt(res.text);
  },
});

document.querySelectorAll('.grade').forEach((b) => {
  b.addEventListener('click', () => submitGrade(Number(b.dataset.grade)));
});

/* Settings */
$('settingsBtn').addEventListener('click', () => $('settingsDialog').showModal());
$('rateRange').addEventListener('input', (e) => {
  state.settings.rate = Number(e.target.value);
  $('rateLabel').textContent = `${state.settings.rate.toFixed(2)}×`;
});
$('settingsDialog').addEventListener('close', () => {
  state.settings.voice = $('voiceSelect').value;
  state.settings.show_english = $('showEnglish').checked;
  state.settings.autoplay = $('autoplay').checked;
  post('/api/settings', state.settings).catch(() => {});
});

/* Keyboard shortcuts */
document.addEventListener('keydown', (e) => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
  const reviewing = $('view-review').classList.contains('is-active');

  if (reviewing && !typing) {
    if (e.code === 'Space') { e.preventDefault(); state.revealed ? submitGrade(3) : reveal(); return; }
    if (['1', '2', '3', '4'].includes(e.key) && state.revealed) { submitGrade(Number(e.key)); return; }
    if (e.key === 'r' && state.card) { speak(state.card.mt); return; }
  }
});

// Acquire the mic on the first gesture, so the first recording does not pay the
// getUserMedia cost mid-utterance.
window.addEventListener('pointerdown', prewarmMic, { once: true });
window.addEventListener('keydown', prewarmMic, { once: true });

boot();
