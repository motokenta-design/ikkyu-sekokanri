const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });
const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const multer = require('multer');
const session = require('express-session');

const app = express();
app.use(express.json());

// ─── Session ─────────────────────────────────────────────────────────────────
app.use(session({
  secret: process.env.SESSION_SECRET || 'sekokanri-beta-secret-2025',
  resave: false,
  saveUninitialized: false,
  cookie: { maxAge: 7 * 24 * 60 * 60 * 1000 },
}));

// ─── Auth middleware ──────────────────────────────────────────────────────────
const PUBLIC_PATHS = ['/login', '/lp.html', '/lp', '/api/auth/login', '/api/auth/logout', '/api/auth/me', '/api/health', '/api/categories'];
function requireAuth(req, res, next) {
  if (PUBLIC_PATHS.some(p => req.path === p || req.path.startsWith(p + '/'))) return next();
  if (req.session && req.session.authenticated) return next();
  if (req.path.startsWith('/api/')) return res.status(401).json({ error: 'Unauthorized' });
  res.redirect('/login');
}
app.use(requireAuth);

// ─── Auth routes ──────────────────────────────────────────────────────────────
app.get('/login', (req, res) => {
  if (req.session && req.session.authenticated) return res.redirect('/');
  res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

app.post('/api/auth/login', (req, res) => {
  const { code } = req.body || {};
  if (!code) return res.json({ ok: false });
  const validCodes = (process.env.BETA_CODES || '')
    .split(',').map(c => c.trim().toUpperCase()).filter(Boolean);
  if (validCodes.includes(code.trim().toUpperCase())) {
    req.session.authenticated = true;
    req.session.code = code.trim().toUpperCase();
    res.json({ ok: true });
  } else {
    res.json({ ok: false });
  }
});

app.post('/api/auth/logout', (req, res) => {
  req.session.destroy(() => res.json({ ok: true }));
});

app.get('/api/auth/me', (req, res) => {
  res.json({ authenticated: !!(req.session && req.session.authenticated) });
});

app.use(express.static(path.join(__dirname, 'public')));

// ─── Gemini setup ─────────────────────────────────────────────────────────────
const GEMINI_GENERATE_MODEL = 'gemini-2.5-flash';
const GEMINI_EXPLAIN_MODEL  = 'gemini-2.5-pro';

// ─── File paths ───────────────────────────────────────────────────────────────
const DATA_DIR   = path.join(__dirname, 'data');
const Q_FILE     = path.join(DATA_DIR, 'questions.json');
const UPLOAD_DIR = path.join(DATA_DIR, 'uploads');
const IMAGES_DIR = path.join(__dirname, 'public', 'images');
const USERS_DIR  = path.join(DATA_DIR, 'users');

function ensureDirs() {
  [DATA_DIR, UPLOAD_DIR, IMAGES_DIR, USERS_DIR].forEach(d => fs.mkdirSync(d, { recursive: true }));
}
ensureDirs();

const uploadImage = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (/^image\/(jpeg|png|gif|webp|svg\+xml)$/.test(file.mimetype)) cb(null, true);
    else cb(new Error('画像ファイルのみ可'));
  },
});

function loadQuestions() {
  try { return JSON.parse(fs.readFileSync(Q_FILE, 'utf8')); } catch { return []; }
}
function saveQuestions(qs) {
  fs.writeFileSync(Q_FILE, JSON.stringify(qs, null, 2), 'utf8');
}

// ─── Robust JSON parser ───────────────────────────────────────────────────────
function parseGeminiJson(raw) {
  let s = raw.replace(/^```(?:json)?\s*/m, '').replace(/\s*```\s*$/m, '').trim();
  try { return JSON.parse(s); } catch {}
  const s2 = s.replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '');
  try { return JSON.parse(s2); } catch {}
  const s3 = s2.replace(/"(?:[^"\\]|\\.)*"/gs, m => m.replace(/\n/g, '\\n').replace(/\r/g, '\\r'));
  try { return JSON.parse(s3); } catch {}
  const m = s3.match(/\[[\s\S]*\]/);
  if (m) try { return JSON.parse(m[0]); } catch {}
  const objects = [];
  let depth = 0, start = -1, inStr = false, escape = false;
  for (let i = 0; i < s3.length; i++) {
    const c = s3[i];
    if (escape) { escape = false; continue; }
    if (c === '\\' && inStr) { escape = true; continue; }
    if (c === '"') { inStr = !inStr; continue; }
    if (inStr) continue;
    if (c === '{') { if (depth++ === 0) start = i; }
    else if (c === '}' && depth > 0) {
      if (--depth === 0 && start !== -1) {
        try { objects.push(JSON.parse(s3.slice(start, i + 1))); } catch {}
        start = -1;
      }
    }
  }
  if (objects.length > 0) return objects;
  throw new Error('JSON解析失敗: ' + s.slice(0, 300));
}

// ─── Retry helper ─────────────────────────────────────────────────────────────
function withTimeout(promise, ms = 15 * 60 * 1000, label = '') {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`タイムアウト (${ms/1000}秒): ${label}`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

async function withRetry(fn, maxRetries = 6, label = '') {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await withTimeout(fn(), 15 * 60 * 1000, label);
    } catch (e) {
      const msg = e.message || '';
      const is429 = msg.includes('429') || msg.includes('quota') || msg.includes('RESOURCE_EXHAUSTED');
      if (!is429 || attempt === maxRetries) throw e;
      const delayMatch = msg.match(/"retryDelay"\s*:\s*"(\d+)s"/);
      const delaySec = delayMatch ? Math.max(parseInt(delayMatch[1]), 10) : Math.min(30 * Math.pow(2, attempt - 1), 300);
      console.log(`⏳ レート制限 – ${delaySec}秒後にリトライ (${attempt}/${maxRetries})`);
      await new Promise(r => setTimeout(r, delaySec * 1000));
    }
  }
}

// ─── Health ───────────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', questions: loadQuestions().length });
});

app.get('/api/categories', (req, res) => {
  const qs = loadQuestions();
  const cats = ['建築学', '共通工学', '施工管理法', '法規', '施工'];
  res.json({
    categories: cats.map(name => ({
      name,
      count: qs.filter(q => q.category === name).length,
    }))
  });
});

// ─── Questions API ────────────────────────────────────────────────────────────
app.get('/api/questions', (req, res) => {
  const { category, source } = req.query;
  let qs = loadQuestions();
  if (category && category !== 'all') qs = qs.filter(q => q.category === category);
  if (source === 'ai') qs = qs.filter(q => q.year === 0);
  res.json(qs);
});

// ─── Explain API ──────────────────────────────────────────────────────────────
app.post('/api/explain', async (req, res) => {
  const { question, choices, answer, category, year } = req.body;
  if (!question) return res.status(400).json({ error: 'questionが必要です' });
  try {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      model: GEMINI_EXPLAIN_MODEL,
      generationConfig: { temperature: 0.1 },
    });
    const prompt = `あなたは一級建築施工管理技士試験の専門家です。以下の問題について解説してください。

科目：${category}
問題文：${question}
選択肢：
  1. ${choices[0]}
  2. ${choices[1]}
  3. ${choices[2]}
  4. ${choices[3]}
正解：${answer}番

解説の厳守事項：
・確実に正しい知識のみ記述し、不確かな内容は「（※要確認）」と明記すること
・法規問題では根拠法令と条文番号を必ず記載すること（例：建設業法第26条第1項）
・正解の理由と、各誤り選択肢がなぜ誤りなのかを説明すること
・本問はAIが独自に創作したオリジナル問題である旨を末尾に付記すること
・200〜400字程度

解説のみを返してください（JSON不要）。`;
    const result = await withRetry(() => model.generateContent(prompt));
    res.json({ explanation: result.response.text() });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ─── User progress API ────────────────────────────────────────────────────────
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function loadUser(id) {
  try { return JSON.parse(fs.readFileSync(path.join(USERS_DIR, `${id}.json`), 'utf8')); } catch { return null; }
}
function saveUser(id, data) {
  fs.writeFileSync(path.join(USERS_DIR, `${id}.json`), JSON.stringify(data), 'utf8');
}

app.post('/api/user/new', (req, res) => {
  const userId = require('crypto').randomUUID();
  const now = new Date().toISOString();
  const data = { userId, createdAt: now, updatedAt: now, streak: 0, lastDate: null, sessions: [], totalCorrect: 0, totalAnswered: 0, weakIds: [] };
  saveUser(userId, data);
  res.json({ userId });
});

app.get('/api/user/:id', (req, res) => {
  const { id } = req.params;
  if (!UUID_RE.test(id)) return res.status(400).json({ error: '無効なID' });
  const data = loadUser(id);
  if (!data) return res.status(404).json({ error: 'ユーザーが見つかりません' });
  res.json(data);
});

app.post('/api/user/:id/sync', (req, res) => {
  const { id } = req.params;
  if (!UUID_RE.test(id)) return res.status(400).json({ error: '無効なID' });
  const existing = loadUser(id);
  if (!existing) return res.status(404).json({ error: 'ユーザーが見つかりません' });
  const { streak, lastDate, sessions, totalCorrect, totalAnswered, weakIds } = req.body;
  const updated = { ...existing, streak, lastDate, sessions: sessions || [], totalCorrect: totalCorrect || 0, totalAnswered: totalAnswered || 0, weakIds: weakIds || [], updatedAt: new Date().toISOString() };
  saveUser(id, updated);
  res.json({ ok: true, updatedAt: updated.updatedAt });
});

// ─── セッション紐付け進捗API（βコードで自動クロスデバイス同期）────────────────
function getProgressPath(code) {
  const hash = require('crypto').createHash('sha256').update(code || 'default').digest('hex').slice(0, 16);
  return path.join(USERS_DIR, `prog_${hash}.json`);
}

app.get('/api/me/progress', (req, res) => {
  if (!req.session?.authenticated) return res.status(401).json({ error: 'Unauthorized' });
  try {
    const data = JSON.parse(fs.readFileSync(getProgressPath(req.session.code || 'default'), 'utf8'));
    res.json(data);
  } catch {
    res.json({ streak: 0, lastDate: null, sessions: [], totalCorrect: 0, totalAnswered: 0, weakIds: [], updatedAt: null });
  }
});

app.post('/api/me/progress', (req, res) => {
  if (!req.session?.authenticated) return res.status(401).json({ error: 'Unauthorized' });
  const { streak, lastDate, sessions, totalCorrect, totalAnswered, weakIds } = req.body;
  const data = {
    streak: streak || 0, lastDate: lastDate || null,
    sessions: sessions || [], totalCorrect: totalCorrect || 0,
    totalAnswered: totalAnswered || 0, weakIds: weakIds || [],
    updatedAt: new Date().toISOString(),
  };
  try {
    fs.writeFileSync(getProgressPath(req.session.code || 'default'), JSON.stringify(data), 'utf8');
    res.json({ ok: true, updatedAt: data.updatedAt });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ─── Admin: generate single batch ────────────────────────────────────────────
app.post('/api/admin/generate', async (req, res) => {
  const { category, topic, count = 5 } = req.body;
  if (!category) return res.status(400).json({ error: 'categoryが必要です' });
  const numCount = Math.min(Math.max(parseInt(count, 10) || 5, 1), 20);
  try {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      model: GEMINI_GENERATE_MODEL,
      generationConfig: { temperature: 0.55, maxOutputTokens: 8192, thinkingConfig: { thinkingBudget: 0 } },
    });
    const topicText = topic ? `\nテーマ：${topic}` : '';
    const prompt = `あなたは一級建築施工管理技士試験の専門家です。以下の条件で${numCount}問のオリジナル練習問題を作成してください。

科目：${category}${topicText}

【著作権上の重要事項】
・一般財団法人建設業振興基金が実施する試験の過去問・既存の問題集・著作物を一切参照・引用・模倣しないでください
・建設業法・建築基準法・労働安全衛生法等の法令や、JIS等の規格・基準に記載された事実情報のみを根拠に、独立して問題を創作してください
・問題の「アイデア」（問う知識の領域）は試験範囲から選んで構いませんが、「表現」（問題文・選択肢の文章）は完全にオリジナルにすること

厳守事項：
・実際の第一次検定と同等レベルの4者択一問題
・確実に正しい知識のみ記述し、不確かな内容は「（※要確認）」と明記すること
・法規問題では根拠法令と条文番号を必ず記載すること
・解説は200〜350字（正解理由＋各誤り選択肢の解説）
・解説末尾に「※本問はAIが独自に創作したオリジナル問題です。法令・公式資料でご確認ください。」を付記

JSON配列のみ返してください：
[{"no":1,"year":0,"category":"${category}","question":"問題文","choices":["選択肢1","選択肢2","選択肢3","選択肢4"],"answer":1,"explanation":"解説（200〜350字）※本問はAIが独自に創作したオリジナル問題です。法令・公式資料でご確認ください。","source":"AIオリジナル（${category}${topic ? '・' + topic : ''}）"}]`;
    const result = await withRetry(() => model.generateContent(prompt));
    const raw = result.response.text();
    const parsed = parseGeminiJson(raw);
    const ts = Date.now();
    const newQs = parsed.map((q, i) => ({
      ...q, id: `ai_${ts}_${i}`, sourceKey: `ai_${category}`,
      source: `AIオリジナル（${category}${topic ? '・' + topic : ''}）`,
    }));
    const allQs = loadQuestions();
    allQs.push(...newQs);
    saveQuestions(allQs);
    res.json({ added: newQs.length, total: allQs.length, questions: newQs });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// ─── Admin: list / filter questions ──────────────────────────────────────────
app.get('/api/admin/questions', (req, res) => {
  const { category, source, search, page = 1, limit = 50 } = req.query;
  let qs = loadQuestions();
  if (category && category !== 'all') qs = qs.filter(q => q.category === category);
  if (source) qs = qs.filter(q => (q.source || q.sourceKey || '').includes(source));
  if (search) qs = qs.filter(q => q.question.includes(search));
  const total = qs.length;
  const start = (parseInt(page, 10) - 1) * parseInt(limit, 10);
  res.json({ total, page: parseInt(page, 10), items: qs.slice(start, start + parseInt(limit, 10)) });
});

app.delete('/api/admin/questions/:id', (req, res) => {
  const qs = loadQuestions();
  const filtered = qs.filter(q => q.id !== decodeURIComponent(req.params.id));
  if (filtered.length === qs.length) return res.status(404).json({ error: '問題が見つかりません' });
  saveQuestions(filtered);
  res.json({ deleted: 1, total: filtered.length });
});

app.post('/api/admin/questions/delete-bulk', (req, res) => {
  const { ids } = req.body;
  if (!Array.isArray(ids) || ids.length === 0) return res.status(400).json({ error: 'ids配列が必要です' });
  const qs = loadQuestions();
  const idSet = new Set(ids);
  const filtered = qs.filter(q => !idSet.has(q.id));
  saveQuestions(filtered);
  res.json({ deleted: qs.length - filtered.length, total: filtered.length });
});

// ─── Admin: image upload ──────────────────────────────────────────────────────
app.post('/api/admin/questions/:id/image', uploadImage.single('image'), (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const qs = loadQuestions();
  const idx = qs.findIndex(q => q.id === id);
  if (idx === -1) { if (req.file) try { fs.unlinkSync(req.file.path); } catch {} return res.status(404).json({ error: '問題が見つかりません' }); }
  if (!req.file) return res.status(400).json({ error: '画像ファイルが必要です' });
  if (qs[idx].image) try { fs.unlinkSync(path.join(__dirname, 'public', qs[idx].image)); } catch {}
  const ext = path.extname(req.file.originalname).toLowerCase() || '.jpg';
  const filename = `q_${Date.now()}${ext}`;
  const dest = path.join(IMAGES_DIR, filename);
  fs.renameSync(req.file.path, dest);
  qs[idx].image = `/images/${filename}`;
  delete qs[idx].image_svg;
  saveQuestions(qs);
  res.json({ image: qs[idx].image });
});

app.delete('/api/admin/questions/:id/image', (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const qs = loadQuestions();
  const idx = qs.findIndex(q => q.id === id);
  if (idx === -1) return res.status(404).json({ error: '問題が見つかりません' });
  if (qs[idx].image) { try { fs.unlinkSync(path.join(__dirname, 'public', qs[idx].image)); } catch {} delete qs[idx].image; }
  delete qs[idx].image_svg;
  saveQuestions(qs);
  res.json({ ok: true });
});

// ─── Admin: generate SVG ──────────────────────────────────────────────────────
app.post('/api/admin/questions/:id/generate-svg', async (req, res) => {
  const id = decodeURIComponent(req.params.id);
  const qs = loadQuestions();
  const idx = qs.findIndex(q => q.id === id);
  if (idx === -1) return res.status(404).json({ error: '問題が見つかりません' });
  const q = qs[idx];
  try {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({ model: GEMINI_EXPLAIN_MODEL, generationConfig: { temperature: 0.2 } });
    const prompt = `以下の一級建築施工管理技士試験問題（${q.category}）に含まれる図・グラフ・表・計算図をSVGコードで再現してください。

問題文：${q.question}
選択肢：
${(q.choices || []).map((c, i) => `${i + 1}. ${c}`).join('\n')}

要件：
・<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 350">...</svg> 形式
・背景は白（rect fill="white"）
・日本語テキストはfont-family="'Helvetica Neue', Arial, sans-serif"
・シンプルで受験生が理解しやすいデザイン
・SVGコードのみを返す（説明文不要）`;
    const result = await withRetry(() => model.generateContent(prompt));
    const raw = result.response.text();
    const svgMatch = raw.match(/<svg[\s\S]*?<\/svg>/i);
    if (!svgMatch) return res.status(500).json({ error: 'SVGが生成できませんでした。' });
    qs[idx].image_svg = svgMatch[0];
    saveQuestions(qs);
    res.json({ image_svg: qs[idx].image_svg });
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ─── Bulk AI question generation (batch job) ──────────────────────────────────
const CATEGORY_TOPICS = {
  '建築学'   : ['構造力学・応力計算', 'RC構造（鉄筋コンクリート）', '鉄骨構造・接合部設計',
                '木構造・防火構造', '地盤・基礎工事', '建築材料の性質・試験',
                '環境工学（熱・光・音）', '建築設備（空調・給排水・電気）',
                '仮設構造物の設計', '耐震・免震・制振構造'],
  '共通工学' : ['測量・施工測量', '契約・積算・見積もり', '建設機械・仮設設備',
                '設計図書・仕様書の読み方', '情報通信技術（ICT）活用'],
  '施工管理法': ['工程管理・ネットワーク工程表', '品質管理・統計的手法（QC）',
                '安全管理・労働安全衛生法', '施工計画・仮設計画',
                '建設業法総合', '環境管理・廃棄物処理', '原価管理・コスト管理',
                '現場組織・職長教育', '施工記録・竣工図', '危機管理・リスク対応'],
  '法規'     : ['建築基準法（構造・防火規定）', '建築基準法（用途地域・確認申請）',
                '労働安全衛生法・特定元方事業者', '建設業法・請負契約・許可',
                '建設リサイクル法・廃棄物処理法', '消防法・危険物', '騒音・振動規制法'],
  '施工'     : ['地盤改良・山留め工事', '土工事・掘削・残土処理', '鉄筋工事・加工組立',
                '型枠工事・支保工', 'コンクリート工事・品質管理', '鉄骨工事・建方・溶接',
                '防水工事・シーリング工事', '外装仕上工事（タイル・石工事）',
                '内装仕上工事（左官・塗装・クロス）', '建設設備工事（電気・給排水・空調）'],
};

const batchJob = {
  running: false, stopped: false,
  total: 0,
  category: '', topic: '', log: [],
  catProgress: { '建築学':0, '共通工学':0, '施工管理法':0, '法規':0, '施工':0 },
  catTargets:  { '建築学':1000, '共通工学':500, '施工管理法':1000, '法規':1000, '施工':1000 },
};

async function runBatchJob() {
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({
    model: GEMINI_GENERATE_MODEL,
    generationConfig: { temperature: 0.55, maxOutputTokens: 8192, thinkingConfig: { thinkingBudget: 0 } },
  });

  for (const [cat, target] of Object.entries(batchJob.catTargets)) {
    if (batchJob.stopped) break;
    const existing = loadQuestions().filter(q => q.category === cat).length;
    const needed = Math.max(0, target - existing);
    if (needed <= 0) {
      batchJob.catProgress[cat] = existing;
      batchJob.log.push(`📚 ${cat} 完了（${existing}問）`);
      continue;
    }
    batchJob.log.push(`▶ ${cat} 開始（残り${needed}問）`);
    const topics = CATEGORY_TOPICS[cat] || ['全般'];
    let generated = 0;
    let topicIdx = 0;

    while (generated < needed && !batchJob.stopped) {
      const topic = topics[topicIdx % topics.length];
      topicIdx++;
      batchJob.category = cat;
      batchJob.topic = topic;

      const prompt = `あなたは一級建築施工管理技士試験の専門家です。
科目「${cat}」テーマ「${topic}」で5問のオリジナル練習問題を作成してください。

【著作権上の最重要事項】
・一般財団法人建設業振興基金が実施する試験の過去問・既存問題集・著作物を一切参照・引用・模倣しないこと
・問題の「表現」（問題文・選択肢の文章）は完全にオリジナルとし、試験機関の著作物と「類似性・依拠性」の両方を満たさないよう注意すること
・建設業法・建築基準法・労働安全衛生法等の法令や、JIS等の公的規格に記載された事実情報のみを根拠に独立して創作すること

問題作成基準：
・実際の第一次検定と同等レベルの4者択一問題
・確実に正しい知識のみ記述（不確かな内容は「（※要確認）」と明記）
・法規問題は根拠法令と条文番号を必ず記載
・解説200〜350字（正解理由＋誤り選択肢の解説）
・解説末尾：「※本問はAIが独自に創作したオリジナル問題です。法令・公式資料でご確認ください。」

JSON配列のみ返してください：
[{"no":1,"year":0,"category":"${cat}","question":"問題文","choices":["選択肢1","選択肢2","選択肢3","選択肢4"],"answer":1,"explanation":"解説文※本問はAIが独自に創作したオリジナル問題です。法令・公式資料でご確認ください。","source":"AIオリジナル（${cat}・${topic}）"}]`;

      try {
        const result = await withRetry(() => model.generateContent(prompt), 4, `batch_${cat}_${topic}`);
        const parsed = parseGeminiJson(result.response.text());
        if (!Array.isArray(parsed) || !parsed.length) throw new Error('JSON解析失敗');
        const ts = Date.now();
        const newQs = parsed.map((q, i) => ({
          ...q, id: `ai_${ts}_${topicIdx}_${i}`, sourceKey: `ai_${cat}`,
          source: `AIオリジナル（${cat}・${topic}）`,
        }));
        const allQs = loadQuestions();
        allQs.push(...newQs);
        saveQuestions(allQs);
        generated += newQs.length;
        batchJob.total = allQs.length;
        batchJob.catProgress[cat] = allQs.filter(q => q.category === cat).length;
        batchJob.log.push(`✅ ${cat}「${topic}」${newQs.length}問`);
        if (batchJob.log.length > 20) batchJob.log = batchJob.log.slice(-20);
      } catch (e) {
        batchJob.log.push(`❌ ${cat}「${topic}」: ${e.message.slice(0, 60)}`);
        console.error(`バッチ生成エラー ${cat}/${topic}:`, e.message);
      }
      await new Promise(r => setTimeout(r, 800));
    }
    batchJob.catProgress[cat] = loadQuestions().filter(q => q.category === cat).length;
    batchJob.log.push(`📚 ${cat} 完了（${batchJob.catProgress[cat]}問）`);
  }

  batchJob.running = false;
  batchJob.total = loadQuestions().length;
  console.log(`🎉 バッチ生成完了: 合計${batchJob.total}問`);
}

app.post('/api/admin/generate-batch/start', (req, res) => {
  if (batchJob.running) return res.status(409).json({ error: '生成処理中です' });
  batchJob.running = true;
  batchJob.stopped = false;
  batchJob.log = [];
  // Reset progress based on current counts
  const qs = loadQuestions();
  for (const cat of Object.keys(batchJob.catProgress)) {
    batchJob.catProgress[cat] = qs.filter(q => q.category === cat).length;
  }
  batchJob.total = qs.length;
  runBatchJob().catch(e => { batchJob.running = false; console.error('バッチエラー:', e.message); });
  res.json({ ok: true, targets: batchJob.catTargets });
});

app.post('/api/admin/generate-batch/stop', (req, res) => {
  batchJob.stopped = true;
  res.json({ ok: true });
});

app.get('/api/admin/generate-batch/status', (req, res) => {
  res.json({
    running: batchJob.running,
    stopped: batchJob.stopped,
    total: batchJob.total || loadQuestions().length,
    category: batchJob.category,
    topic: batchJob.topic,
    log: batchJob.log.slice(-20),
    catProgress: batchJob.catProgress,
    catTargets: batchJob.catTargets,
  });
});

// ─── Start server ─────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3457;
app.listen(PORT, () => {
  console.log(`🏗️  一級建築施工管理技士 受験対策アプリ v2.0`);
  console.log(`🚀 Server: http://localhost:${PORT}`);
  console.log(`🔧 Admin:  http://localhost:${PORT}/admin.html`);
  // 起動時にカテゴリ別問題数を表示
  const qs = loadQuestions();
  const cats = {};
  qs.forEach(q => { cats[q.category] = (cats[q.category]||0)+1; });
  console.log(`📚 問題数: ${qs.length}問`, cats);
});
