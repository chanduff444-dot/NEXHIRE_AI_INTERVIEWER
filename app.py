from dotenv import load_dotenv
import os
load_dotenv()
print("MONGO_URI:", os.environ.get("MONGO_URI"))
"""MONGO_URI = os.environ.get("MONGO_URI")
NexHire AI — Interview Intelligence Platform
============================================
Key Novelties:
  01 · AI Interview Copilot         — live analysis, follow-ups, credibility flags
  02 · Simulation-Based Interviewing — real-world engineering scenarios
  03 · Integrity Monitoring          — tab-switch, delays, AI-language detection
  04 · GitHub Analysis System        — code quality, contributions, project relevance  ← ADDED
  05 · Gamified Simulation           — timers, boosters, adaptive difficulty
  06 · Interview Replay Intelligence — timeline, key moments, hire narrative

Architecture:
  Layer 1 · Input      : PDF upload, Browser STT, Audio file, Transcript paste
  Layer 2 · Processing : Flask REST, PyMuPDF, Groq Whisper STT
  Layer 3 · AI/RAG     : Groq LLaMA 3.3 70B, sentence-transformers, ChromaDB
  Layer 4 · Output     : Live copilot, scores, report, metrics

Run:     python app.py
Visit:   http://localhost:5000
Install: pip install flask groq pymupdf sentence-transformers chromadb requests
"""
import os, json, re, tempfile, requests
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, join_room, leave_room, emit
from groq import Groq
import fitz  # PyMuPDF

# -- Database ---------------------------------------------------------------
# SQLite is the PRIMARY store (always works, zero config).
# MongoDB Atlas is synced in the background if available.
# This means sessions ALWAYS work regardless of Atlas status.
# ----------------------------------------------------------------------------
import sqlite3, threading, time as _time_mod

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexhire_sessions.db")
_db_lock  = threading.Lock()

# Atlas URI - your cluster
import os

MONGO_URI = os.environ.get("MONGO_URI")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable not set")

_MONGO_OK     = False
_sessions_col = None

try:
    from pymongo import MongoClient
    # Support both old and new pymongo
    try:
        from pymongo.server_api import ServerApi
        _mc = MongoClient(MONGO_URI, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
    except ImportError:
        _mc = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    _mc.admin.command('ping')
    _sessions_col = _mc["nexhire"]["sessions"]
    _sessions_col.create_index("pin", unique=True)
    _MONGO_OK = True
    print("[DB] MongoDB Atlas connected")
except Exception as _me:
    print(f"[DB] MongoDB Atlas not available ({type(_me).__name__}: {_me})")
    print("[DB] Running on SQLite only - sessions will still work perfectly")

# Always init SQLite
def _db_init():
    with _db_lock, sqlite3.connect(_DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                pin        TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        con.commit()
    print(f"[DB] SQLite ready: {_DB_PATH}")

_db_init()

# -- Session helpers ----------------------------------------------------------
# READ:  SQLite first (guaranteed), Atlas as secondary
# WRITE: SQLite always, Atlas in background thread

def _session_get(pin: str):
    """Get session - SQLite is primary, Atlas is backup."""
    # Always try SQLite first - it's local and instant
    try:
        with _db_lock, sqlite3.connect(_DB_PATH) as con:
            row = con.execute("SELECT data FROM sessions WHERE pin=?", (pin,)).fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        print(f"[DB] SQLite read error: {e}")

    # Fallback to Atlas if SQLite missed it
    if _MONGO_OK and _sessions_col is not None:
        try:
            doc = _sessions_col.find_one({"pin": pin}, {"_id": 0})
            if doc:
                # Sync back to SQLite for next time
                _sqlite_write(pin, doc)
                return doc
        except Exception as e:
            print(f"[DB] Atlas read error: {e}")

    return None

def _sqlite_write(pin: str, data: dict):
    """Write to SQLite - always called synchronously."""
    try:
        with _db_lock, sqlite3.connect(_DB_PATH) as con:
            con.execute(
                "INSERT OR REPLACE INTO sessions (pin, data, updated_at) VALUES (?,?,?)",
                (pin, json.dumps(data), _time_mod.time())
            )
            con.commit()
    except Exception as e:
        print(f"[DB] SQLite write error: {e}")

def _atlas_write_bg(pin: str, data: dict):
    """Write to Atlas in background thread - fire and forget."""
    if not _MONGO_OK or _sessions_col is None:
        return
    def _do():
        try:
            _sessions_col.update_one({"pin": pin}, {"$set": data}, upsert=True)
            print(f"[DB] Atlas sync OK: pin={pin}")
        except Exception as e:
            print(f"[DB] Atlas sync error: {e}")
    threading.Thread(target=_do, daemon=True).start()

def _session_set(pin: str, data: dict) -> None:
    """Save session - SQLite immediately, Atlas in background."""
    _sqlite_write(pin, data)
    print(f"[DB] Session saved to SQLite: pin={pin}")
    _atlas_write_bg(pin, data)

def _session_update(pin: str, fields: dict) -> None:
    """Update session fields - SQLite immediately, Atlas in background."""
    # Read-modify-write in SQLite
    try:
        with _db_lock, sqlite3.connect(_DB_PATH) as con:
            row = con.execute("SELECT data FROM sessions WHERE pin=?", (pin,)).fetchone()
            if row:
                doc = json.loads(row[0])
                doc.update(fields)
                con.execute(
                    "UPDATE sessions SET data=?, updated_at=? WHERE pin=?",
                    (json.dumps(doc), _time_mod.time(), pin)
                )
                con.commit()
                _atlas_write_bg(pin, doc)
    except Exception as e:
        print(f"[DB] SQLite update error: {e}")

# ── Optional: sentence-transformers + ChromaDB (real RAG) ──────────────────────
try:
    from sentence_transformers import SentenceTransformer, util
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# ─── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'nexhire-secret-key-change-in-prod')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# In-memory map: pin -> role -> socket_id
_room_members = {}  # { pin: { 'interviewer': sid, 'candidate': sid } }

# Pending game/simulation events for candidates who haven't joined yet
# { pin: { 'event': str, 'data': dict } }
_pending_game_events = {}

# ─── Groq ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client       = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_WHISPER = "whisper-large-v3"

SYSTEM_PROMPT = """You are NexHire AI, an advanced Interview Intelligence Platform.
You power: AI Copilot, Simulation Engine, Integrity Monitor, GitHub Analyzer, and Replay Intelligence.
CRITICAL: Respond ONLY with valid JSON. No markdown, no preamble. Pure JSON parseable by json.loads()."""

# ─── Layer 3: Vector Store globals ────────────────────────────────────────────
_embed_model   = None
_chroma_client = None
_collection    = None

def get_embed_model():
    global _embed_model
    if _embed_model is None and _ST_AVAILABLE:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model

def get_collection():
    global _chroma_client, _collection
    if _collection is None and _CHROMA_AVAILABLE:
        _chroma_client = chromadb.Client()
        _collection = _chroma_client.get_or_create_collection(
            name="resume_chunks", metadata={"hnsw:space": "cosine"}
        )
    return _collection

def chunk_text(text: str, size: int = 300) -> list:
    words = text.split()
    step  = int(size * 0.8)
    return [" ".join(words[i:i+size]) for i in range(0, len(words), step) if words[i:i+size]]

def index_resume_vectors(resume_text: str, session_id: str) -> int:
    if not (_ST_AVAILABLE and _CHROMA_AVAILABLE):
        return 0
    col   = get_collection()
    model = get_embed_model()
    if not col or not model:
        return 0
    try:
        old = col.get(where={"session_id": session_id})
        if old['ids']:
            col.delete(ids=old['ids'])
    except Exception:
        pass
    chunks     = chunk_text(resume_text)
    if not chunks:
        return 0
    embeddings = model.encode(chunks).tolist()
    ids        = [f"{session_id}_{i}" for i in range(len(chunks))]
    metas      = [{"session_id": session_id, "chunk_index": i} for i in range(len(chunks))]
    col.add(embeddings=embeddings, documents=chunks, ids=ids, metadatas=metas)
    return len(chunks)

def vector_retrieve(query: str, session_id: str, top_k: int = 3) -> list:
    if not (_ST_AVAILABLE and _CHROMA_AVAILABLE):
        return []
    col   = get_collection()
    model = get_embed_model()
    if not col or not model:
        return []
    try:
        n = min(top_k, col.count())
        if n == 0:
            return []
        q_emb = model.encode([query]).tolist()
        res   = col.query(query_embeddings=q_emb, n_results=n, where={"session_id": session_id})
        return [{"text": doc, "score": round(1 - dist, 4)}
                for doc, dist in zip(res['documents'][0], res['distances'][0])]
    except Exception:
        return []

def compute_similarity(t1: str, t2: str) -> dict:
    if _ST_AVAILABLE:
        model = get_embed_model()
        e1    = model.encode(t1, convert_to_tensor=True)
        e2    = model.encode(t2, convert_to_tensor=True)
        score = float(util.cos_sim(e1, e2)[0][0])
        label = "HIGH MATCH" if score >= 0.75 else "PARTIAL MATCH" if score >= 0.45 else "LOW MATCH"
        return {"similarity_score": round(score, 4), "similarity_pct": round(score * 100, 1),
                "label": label, "method": "sentence-transformers (real vectors)", "engine": "vector"}
    schema = {"similarity_score": 0.5, "label": "PARTIAL MATCH", "explanation": "string"}
    try:
        r    = call_groq(f"Score semantic similarity 0-1:\nT1: {t1}\nT2: {t2}", schema)
        s    = r.get("similarity_score", 0.5)
        r["similarity_pct"] = round(s * 100, 1)
        r["method"] = "LLM-estimated (install sentence-transformers for real vectors)"
        r["engine"] = "llm"
        return r
    except Exception:
        return {"similarity_score": 0.5, "similarity_pct": 50, "label": "PARTIAL MATCH",
                "method": "fallback", "engine": "fallback"}

# ─── Layer 2: Groq Whisper STT ────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, filename: str, language: str = "en") -> dict:
    ext  = os.path.splitext(filename.lower())[1] or ".wav"
    mime = {".mp3":"audio/mpeg",".wav":"audio/wav",".m4a":"audio/mp4",
            ".mp4":"audio/mp4",".webm":"audio/webm",".ogg":"audio/ogg",".flac":"audio/flac"}
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            tx = client.audio.transcriptions.create(
                file=(filename, f, mime.get(ext, "audio/wav")),
                model=GROQ_WHISPER, language=language,
                response_format="verbose_json", temperature=0.0
            )
        text = tx.text if hasattr(tx, 'text') else str(tx)
        segs = []
        if hasattr(tx, 'segments') and tx.segments:
            segs = [{"start": round(getattr(s,'start',0),2),
                     "end":   round(getattr(s,'end',0),2),
                     "text":  getattr(s,'text','')} for s in tx.segments]
        duration = segs[-1]['end'] if segs else 0
        return {"transcript": text, "segments": segs, "duration_seconds": duration,
                "language": language, "model": GROQ_WHISPER, "word_count": len(text.split())}
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# ─── Layer 3: LLM helper ──────────────────────────────────────────────────────
def call_groq(message: str, schema: dict) -> dict:
    prompt = f"{message}\n\nRespond ONLY with valid JSON:\n{json.dumps(schema, indent=2)}\n\nNo markdown. Pure JSON only."
    comp   = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        temperature=0.3, max_tokens=2048
    )
    raw = re.sub(r'^```(?:json)?\s*', '', comp.choices[0].message.content.strip())
    raw = re.sub(r'\s*```$', '', raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"error": "parse_failed", "raw": raw[:300]}

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
#  LAYER 1 — INPUT ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/parse-pdf', methods=['POST'])
def parse_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF accepted"}), 400
    try:
        pdf_bytes = file.read()
        text      = extract_text_from_pdf(pdf_bytes)
        if not text:
            return jsonify({"error": "Could not extract text"}), 400
        doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = len(doc); doc.close()
        return jsonify({"text": text, "pages": pages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/whisper-transcribe', methods=['POST'])
def whisper_transcribe():
    """STT: Upload audio → Groq Whisper large-v3 → transcript."""
    if 'file' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    file = request.files['file']
    ext  = os.path.splitext(file.filename.lower())[1]
    if ext not in {'.mp3','.wav','.m4a','.mp4','.webm','.ogg','.flac'}:
        return jsonify({"error": f"Format '{ext}' not supported"}), 400
    try:
        result = transcribe_audio(file.read(), file.filename, request.form.get('language','en'))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/challenge/kilimanjaro")
def kilimanjaro_game():
    return render_template("kilimanjaro_game.html")   

@app.route('/api/transcribe-blob', methods=['POST'])
def transcribe_blob():
    """Receive base64 audio blob from browser MediaRecorder → Groq Whisper."""
    data = request.get_json()
    if not data or 'audio_b64' not in data:
        return jsonify({"error": "audio_b64 required"}), 400
    try:
        import base64
        audio_bytes = base64.b64decode(data['audio_b64'])
        result      = transcribe_audio(audio_bytes, data.get('filename','recording.webm'), data.get('language','en'))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/transcribe-chunk', methods=['POST'])
def transcribe_chunk():
    """Live capture STT: Receives WebM audio chunk → Groq Whisper → returns text."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio chunk"}), 400
    file = request.files['audio']
    try:
        audio_bytes = file.read()
        if len(audio_bytes) < 1000:
            return jsonify({"text": "", "skipped": True, "reason": "chunk too small"})
        result = transcribe_audio(audio_bytes, file.filename or 'chunk.webm', 'en')
        return jsonify({"text": result.get("transcript",""), "duration": result.get("duration_seconds",0), "model": GROQ_WHISPER})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-transcript', methods=['POST'])
def process_transcript():
    """Accept paste/webhook transcript from Zoom, Meet, Webex → parse Q&A."""
    data = request.get_json()
    raw  = data.get('transcript','').strip()
    if not raw:
        return jsonify({"error": "Transcript text required"}), 400
    schema = {
        "qa_pairs": [{"question":"string","answer":"string","timestamp":"string"}],
        "total_turns": 0, "candidate_speaking_pct": 0,
        "topics_covered": ["string"], "speakers_detected": ["string"]
    }
    try:
        return jsonify(call_groq(f"Parse interview transcript into Q&A pairs. Detect speakers.\nTRANSCRIPT:\n{raw[:3000]}", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  LAYER 2+3 — PROCESSING + AI/RAG ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume():
    """LLM resume parse + index chunks into ChromaDB vector store."""
    data        = request.get_json()
    resume_text = data.get('resume_text','').strip()
    role_type   = data.get('role_type','technical')
    session_id  = data.get('session_id','default')
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    schema = {
        "name":"string","email":"string","phone":"string",
        "years_experience":0,"current_role":"string","education":"string","gpa":"string",
        "skills":["string"],"languages":["string"],"frameworks":["string"],"tools":["string"],"cloud":["string"],
        "companies":[{"name":"string","role":"string","duration":"string"}],
        "projects":[{"name":"string","description":"string"}],
        "major_claims":[{"claim":"string","priority":"high|medium|low","risk_level":"Low|Medium|High","validation_question":"string"}],
        "competencies":{"technical":["string"],"behavioral":["string"],"leadership":["string"]},
        "inflation_signals":["string"],"strengths_from_resume":["string"],
        "summary":"string","overall_resume_quality":"Poor|Fair|Good|Excellent"
    }
    try:
        result        = call_groq(f"Analyze this {role_type} role resume:\n\n{resume_text}", schema)
        chunks        = index_resume_vectors(resume_text, session_id)
        result["vector_indexed"] = chunks
        result["vector_engine"]  = f"sentence-transformers + ChromaDB ({chunks} chunks)" if chunks else "LLM-only (install sentence-transformers chromadb)"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-response', methods=['POST'])
def analyze_response():
    """
    Key Novelty 01 — AI Copilot core:
    RAG-augmented response analysis with contradiction detection,
    follow-up generation, credibility scoring, and coaching tips.
    """
    data        = request.get_json()
    question    = data.get('question','').strip()
    response    = data.get('response','').strip()
    resume_text = data.get('resume_text','')
    history     = data.get('history',[])
    session_id  = data.get('session_id','default')
    if not question or not response:
        return jsonify({"error": "Question and response required"}), 400

    # RAG Retrieval
    chunks  = vector_retrieve(f"{question} {response}", session_id, top_k=3)
    rag_ctx = ""
    if chunks:
        rag_ctx = "\n\nRetrieved resume evidence (vector search):\n" + \
                  "\n".join(f"[sim={c['score']}] {c['text']}" for c in chunks)

    schema = {
        "resume_consistency":"Aligned|Partial|Contradiction|Not Applicable",
        "consistency_reason":"string",
        "rag_evidence_used":["string"],
        "scores":{"technical_depth":0,"clarity":0,"specificity":0},
        "red_flags":[{"type":"string","description":"string","severity":"low|medium|high"}],
        "strengths":[{"description":"string"}],
        "follow_up_questions":[{"question":"string","category":"Technical|Behavioral|Leadership|Clarification","probe_type":"string"}],
        "coaching_tip":"string",
        "overall_assessment":"string",
        "answer_completeness":"Complete|Partial|Vague|Evasive",
        "confidence_signals":"string"
    }
    resume_ctx = f"Candidate resume:\n{resume_text[:1500]}\n" if resume_text else ""
    hist_ctx   = "Prior exchanges:\n" + "\n\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]]) + "\n\n" if history else ""
    prompt     = f"{resume_ctx}{rag_ctx}\n{hist_ctx}Question: {question}\nCandidate: {response}\n\nAnalyze. Use retrieved evidence to ground your verdict."
    try:
        result = call_groq(prompt, schema)
        if resume_text and response:
            sem = compute_similarity(response, resume_text[:500])
            result["semantic_match_score"] = sem["similarity_pct"]
            result["semantic_method"]      = sem["method"]
        result["rag_chunks_used"] = len(chunks)
        result["rag_chunks"]      = chunks
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rag-factcheck', methods=['POST'])
def rag_factcheck():
    """RAG fact-check: embed claim → retrieve resume chunks → LLM verdict grounded in evidence."""
    data        = request.get_json()
    claim       = data.get('claim','').strip()
    session_id  = data.get('session_id','default')
    resume_text = data.get('resume_text','')
    if not claim:
        return jsonify({"error": "claim required"}), 400
    chunks       = vector_retrieve(claim, session_id, top_k=4)
    evidence_ctx = "\n".join(f"[sim={c['score']}] {c['text']}" for c in chunks) if chunks else resume_text[:800]
    schema = {
        "verdict":"VERIFIED|LIKELY TRUE|UNVERIFIABLE|SUSPICIOUS|FALSE",
        "confidence":0,"explanation":"string",
        "supporting_evidence":["string"],"contradicting_evidence":["string"],
        "probe_question":"string"
    }
    try:
        result = call_groq(
            f"Fact-check claim using retrieved resume evidence:\nCLAIM: \"{claim}\"\nEVIDENCE:\n{evidence_ctx}\nBase verdict strictly on evidence.",
            schema
        )
        result["retrieved_chunks"] = chunks
        result["rag_method"]       = f"sentence-transformers + ChromaDB ({len(chunks)} chunks)" if chunks else "resume text only"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate full post-interview evaluation report."""
    data        = request.get_json()
    transcript  = data.get('transcript','').strip()
    resume_text = data.get('resume_text','')
    if not transcript:
        return jsonify({"error": "Transcript required"}), 400
    schema = {
        "candidate_name":"string","candidate_summary":"string",
        "strengths":["string"],"concerns":["string"],
        "validated_skills":["string"],"potential_resume_risks":["string"],
        "scores":{"communication":0,"technical_depth":0,"problem_solving":0,"leadership":0,"cultural_fit":0},
        "confidence_level":"Low|Medium|High","confidence_percent":0,
        "red_flags_summary":["string"],"hire_signal":"Strong Yes|Lean Yes|Neutral|Lean No|Strong No",
        "final_insight":"string","recommended_next_steps":["string"],"key_questions_unanswered":["string"]
    }
    resume_ctx = f"Resume:\n{resume_text}\n\n" if resume_text else ""
    try:
        return jsonify(call_groq(f"{resume_ctx}Transcript:\n{transcript}\n\nGenerate full post-interview evaluation.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """Generate structured interview question bank from resume."""
    data        = request.get_json()
    resume_text = data.get('resume_text','').strip()
    role_type   = data.get('role_type','technical')
    difficulty  = data.get('difficulty','mixed')
    if not resume_text:
        return jsonify({"error": "Resume text required"}), 400
    schema = {
        "opening_questions":[{"question":"string","purpose":"string"}],
        "technical_questions":[{"question":"string","skill_tested":"string","difficulty":"easy|medium|hard"}],
        "behavioral_questions":[{"question":"string","competency":"string","star_method":True}],
        "deep_dive_questions":[{"question":"string","claim_being_validated":"string"}],
        "closing_questions":[{"question":"string"}]
    }
    try:
        return jsonify(call_groq(f"Generate {difficulty} {role_type} interview questions from:\n\n{resume_text}", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-trap-questions', methods=['POST'])
def generate_trap_questions():
    """Generate expert trap questions to expose fake experience claims."""
    data   = request.get_json()
    claim  = data.get('claim','').strip()
    skill  = data.get('skill','').strip()
    resume = data.get('resume_text','')
    if not claim and not skill:
        return jsonify({"error": "claim or skill required"}), 400
    schema = {
        "skill_detected":"string","claimed_level":"Beginner|Intermediate|Advanced|Expert",
        "trap_questions":[{"question":"string","difficulty":"Medium|Hard|Expert","trap_type":"string","what_a_fake_says":"string","what_an_expert_says":"string","credibility_weight":0}],
        "instant_credibility_checks":["string"],"deflection_signals":["string"],"scoring_rubric":"string"
    }
    ctx = f"Resume:\n{resume[:800]}\n\n" if resume else ""
    try:
        return jsonify(call_groq(f"{ctx}Claim:\"{claim}\" Skill:{skill}\nGenerate 4-5 trap questions exposing fake experience.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/credibility-score', methods=['POST'])
def credibility_score():
    """Comprehensive credibility analysis across all interview dimensions."""
    data    = request.get_json()
    history = data.get('history',[])
    resume  = data.get('resume_text','')
    if not history:
        return jsonify({"error": "Session history required"}), 400
    schema = {
        "overall_credibility_score":0,
        "credibility_level":"Very Low|Low|Moderate|High|Very High",
        "credibility_breakdown":{"resume_accuracy":0,"technical_depth":0,"consistency_across_answers":0,"specificity_of_examples":0,"handling_of_failure_scenarios":0},
        "verified_claims":["string"],
        "disputed_claims":[{"claim":"string","reason":"string","severity":"Low|Medium|High"}],
        "fabrication_risk":"Low|Medium|High|Critical",
        "fabrication_signals":["string"],
        "genuine_strengths_detected":["string"],
        "credibility_verdict":"string",
        "recommended_verification":["string"]
    }
    history_text = "\n\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history])
    ctx          = f"Resume:\n{resume[:1000]}\n\n" if resume else ""
    try:
        return jsonify(call_groq(f"{ctx}Interview:\n{history_text}\n\nComprehensive credibility analysis.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/multi-interviewer-sync', methods=['POST'])
def multi_interviewer_sync():
    """Synthesize panel observations into unified hiring intelligence."""
    data         = request.get_json()
    interviewers = data.get('interviewers',[])
    candidate    = data.get('candidate_name','Candidate')
    role         = data.get('role','')
    if not interviewers:
        return jsonify({"error": "Interviewer data required"}), 400
    schema = {
        "consensus_hire_signal":"Strong Yes|Lean Yes|Neutral|Lean No|Strong No",
        "agreement_level":"Full Consensus|Majority|Split|Divided","agreement_percent":0,
        "panel_scores":{"technical":0,"communication":0,"cultural_fit":0,"leadership":0,"problem_solving":0},
        "shared_strengths":["string"],"shared_concerns":["string"],
        "conflicting_observations":[{"topic":"string","interviewer_a_view":"string","interviewer_b_view":"string","resolution":"string"}],
        "unique_insights_per_interviewer":[{"interviewer":"string","unique_finding":"string","importance":"Low|Medium|High"}],
        "areas_needing_more_probing":["string"],"panel_recommendation":"string","debrief_talking_points":["string"]
    }
    panel_text = "\n\n".join([
        f"Interviewer:{iv.get('name','Unknown')} ({iv.get('role','Panel')})\nFocus:{iv.get('focus','General')}\nNotes:{iv.get('notes','')}\nScore:{iv.get('score','N/A')}/10\nHire:{iv.get('hire_signal','Neutral')}"
        for iv in interviewers
    ])
    try:
        return jsonify(call_groq(f"Candidate:{candidate} Role:{role}\n\nPanel from {len(interviewers)} interviewers:\n{panel_text}\n\nSynthesize unified intelligence.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    """Fact-check a candidate claim (Intelligence+ tab)."""
    data    = request.get_json()
    claim   = data.get('claim','').strip()
    context = data.get('context','')
    if not claim:
        return jsonify({"error": "Claim required"}), 400
    schema = {
        "claim_extracted":"string","verdict":"VERIFIED|LIKELY TRUE|UNVERIFIABLE|SUSPICIOUS|FALSE",
        "verdict_color":"green|yellow|gray|orange|red","confidence":0,
        "explanation":"string","known_facts":["string"],"red_flags":["string"],"probe_question":"string"
    }
    try:
        return jsonify(call_groq(f"Fact-check: CLAIM: {claim}\nCONTEXT: {context}", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/semantic-match', methods=['POST'])
def semantic_match():
    """Cosine similarity (Platform+ tab)."""
    data = request.get_json()
    t1   = data.get('text1','').strip()
    t2   = data.get('text2','').strip()
    if not t1 or not t2:
        return jsonify({"error": "text1 and text2 required"}), 400
    return jsonify(compute_similarity(t1, t2))

@app.route('/api/legal-scan', methods=['POST'])
def legal_scan():
    """Employment law question scanner (Platform+ tab)."""
    data         = request.get_json()
    question     = data.get('question','').strip()
    jurisdiction = data.get('jurisdiction','IN')
    if not question:
        return jsonify({"error": "Question required"}), 400
    jnames = {'IN':'India','US':'United States','UK':'United Kingdom','EU':'European Union'}
    jname  = jnames.get(jurisdiction,'India')
    schema = {
        "verdict":"SAFE|CAUTION|ILLEGAL","risk_score":0,"legal_category":"string",
        "law_reference":"string","reason":"string","protected_attribute":"string",
        "safe_alternatives":["string"],"reformulated_question":"string","explanation":"string"
    }
    try:
        return jsonify(call_groq(f"Employment law expert for {jname}. Assess: \"{question}\"\nReturn SAFE/CAUTION/ILLEGAL and 3 safe alternatives.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cheat-detect', methods=['POST'])
def cheat_detect():
    """Detect AI-assistance patterns, language shifts, and fabrication signals."""
    data      = request.get_json()
    responses = data.get('responses',[])
    single    = data.get('response','')
    resume    = data.get('resume_text','')
    schema = {
        "cheat_risk_index":0,"risk_level":"Low|Medium|High|Critical","risk_verdict":"string",
        "signals":{"language_shift_score":0,"answer_perfection_score":0,"consistency_drop_score":0,"specificity_flip_score":0},
        "anomalies":[{"type":"string","severity":"low|medium|high","description":"string","evidence":"string"}],
        "probe_questions":["string"],"verdict_explanation":"string","fabrication_likelihood":"Low|Moderate|High|Very High"
    }
    text = f"Single response AI-assistance check:\n\n{single}" if single else \
           "Full session:\n\n" + "\n\n".join([f"Q:{r.get('q','')}\nA:{r.get('a','')}" for r in responses[-8:]])
    ctx = f"Resume:\n{resume[:400]}\n\n" if resume else ""
    try:
        return jsonify(call_groq(f"{ctx}{text}\n\nDetect AI-assistance patterns: language shifts, perfection without anecdotes, inconsistency.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  KEY NOVELTY 02 + 05 — SIMULATION ENGINE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/generate-simulation', methods=['POST'])
def generate_simulation():
    
    """
    Key Novelty 02 + 05: Generate a gamified simulation scenario.
    Types: incident, design, debug, arch, oncall
    Difficulty: easy, medium, hard, expert
    Adaptive: adjusts based on candidate performance history
    """
      
    data       = request.get_json()
    sim_type   = data.get('type', 'incident')
    difficulty = data.get('difficulty', 'medium')
    resume     = data.get('resume_text', '')
    history    = data.get('history', [])
    type_map={
        'incident': 'Production Incident Response',
        'design':   'System Design Under Pressure',
        'debug':    'Live Debugging Challenge',
        'arch':     'Architecture Decision',
        'oncall':   'On-Call Escalation Scenario'
    }
    type_name  = type_map.get(sim_type, 'Production Incident')
    time_limit = {'easy': 90, 'medium': 120, 'hard': 90, 'expert': 60}.get(difficulty, 120)

    past_correct = sum(1 for h in history if h.get('correct'))
    past_total   = len(history)
    adapt_ctx    = f"Candidate history: {past_correct}/{past_total} correct. " if past_total > 0 else ""
    ctx          = f"Resume context: {resume[:400]}\n" if resume else ""

    schema = {
        "title": "string",
        "type": sim_type,
        "difficulty": difficulty,
        "description": "2-3 sentence urgent scenario description",
        "context": {"system": "string", "error": "string", "stakes": "string"},
        "time_limit": time_limit,
        "options": [
            {"label": "A", "text": "option text max 120 chars", "correct": False, "explanation": "why wrong/right"},
            {"label": "B", "text": "option text max 120 chars", "correct": True,  "explanation": "why this is best"},
            {"label": "C", "text": "option text max 120 chars", "correct": False, "explanation": "why wrong"},
            {"label": "D", "text": "option text max 120 chars", "correct": False, "explanation": "why suboptimal"}
        ],
        "expert_insight": "1-2 sentences what senior engineer thinks first",
        "hint": "1 sentence hint without revealing answer",
        "tags": ["string"],
        "scoring": {"base_points": 20, "time_bonus_max": 50, "streak_multiplier": 1.5}
    }
    try:
        return jsonify(call_groq(
            f"{ctx}{adapt_ctx}Generate {difficulty} {type_name} simulation for a software engineering interview. "
            f"Make it realistic, urgent, and technically accurate. Only ONE option should be correct.",
            schema
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
      

@app.route('/api/simulation-evaluate', methods=['POST'])
def simulation_evaluate():
    """Evaluate candidate's full simulation performance with coaching feedback."""
    data    = request.get_json()
    history = data.get('history', [])
    resume  = data.get('resume_text', '')
    if not history:
        return jsonify({"error": "Simulation history required"}), 400
    schema = {
        "overall_score": 0,
        "performance_level": "Novice|Developing|Competent|Proficient|Expert",
        "decision_quality": 0, "speed_under_pressure": 0,
        "improvement_areas": ["string"], "strengths_shown": ["string"],
        "coaching_feedback": "string",
        "recommended_next_difficulty": "easy|medium|hard|expert",
        "technical_gaps": ["string"]
    }
    history_text = "\n".join([
        f"Scenario:{h.get('title','?')} Diff:{h.get('difficulty','?')} Correct:{h.get('correct','?')} Time:{h.get('time_used','?')}s"
        for h in history
    ])
    try:
        return jsonify(call_groq(
            f"Resume:{resume[:300] if resume else 'N/A'}\nSimulation history:\n{history_text}\nEvaluate engineering decision-making.",
            schema
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  KEY NOVELTY 03 — INTEGRITY MONITOR ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/integrity-analyze', methods=['POST'])
def integrity_analyze():
    """
    Key Novelty 03: Comprehensive integrity analysis combining:
    - Behavioral signals (tab switches, copy-paste, response timing)
    - AI language pattern detection
    - Claim consistency across session history
    """
    data         = request.get_json()
    tab_switches = data.get('tab_switches', 0)
    paste_events = data.get('paste_events', 0)
    long_pauses  = data.get('long_pauses', 0)
    responses    = data.get('responses', [])
    resume       = data.get('resume_text', '')

    behavioral_risk = min(100, tab_switches * 18 + paste_events * 10 + long_pauses * 12)

    schema = {
        "integrity_score": 0,
        "risk_level": "Low|Medium|High|Critical",
        "behavioral_risk": behavioral_risk,
        "ai_language_risk": 0,
        "claim_consistency_score": 0,
        "overall_verdict": "string",
        "risk_breakdown": {
            "tab_switch_risk": 0, "paste_risk": 0,
            "response_timing_risk": 0, "language_pattern_risk": 0
        },
        "anomalies": [{"type": "string", "severity": "low|medium|high", "description": "string"}],
        "probe_questions": ["string"],
        "recommendation": "string"
    }
    ctx            = f"Resume: {resume[:300]}\n" if resume else ""
    responses_text = "\n".join([f"Q:{r.get('q','')} A:{r.get('a','')[:200]}" for r in responses[-6:]]) if responses else "No responses"
    try:
        result = call_groq(
            f"{ctx}Behavioral signals: tab_switches={tab_switches}, paste_events={paste_events}, long_pauses={long_pauses}\n"
            f"Responses:\n{responses_text}\n"
            "Analyze interview integrity. Detect AI assistance patterns and behavioral anomalies.",
            schema
        )
        result['behavioral_risk'] = behavioral_risk
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/response-timing-analyze', methods=['POST'])
def response_timing_analyze():
    """Analyze response timing patterns for anomaly detection."""
    data      = request.get_json()
    timings   = data.get('timings', [])
    questions = data.get('questions', [])
    if not timings:
        return jsonify({"error": "Timing data required"}), 400
    avg_time  = sum(timings) / len(timings)
    anomalies = [i for i, t in enumerate(timings) if t > avg_time * 2 or t > 20]
    schema = {
        "avg_response_time": round(avg_time, 1),
        "anomalous_responses": anomalies,
        "timing_pattern": "Consistent|Variable|Suspicious|Highly Suspicious",
        "long_pause_indices": anomalies,
        "verdict": "string", "explanation": "string"
    }
    timing_text = "\n".join([f"Q{i+1}: {t}s - {questions[i] if i < len(questions) else 'unknown'}" for i, t in enumerate(timings)])
    try:
        return jsonify(call_groq(f"Response timings:\n{timing_text}\nIdentify suspicious delays (>20s) and patterns.", schema))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  KEY NOVELTY 04 — GITHUB ANALYSIS SYSTEM  ← NEW
# ═══════════════════════════════════════════════════════════════════

def fetch_github_data(username: str) -> dict:
    """Fetch public GitHub profile data via GitHub REST API (no auth needed for public data)."""
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "NexHire-AI"}
    base    = f"https://api.github.com/users/{username}"

    user_res  = requests.get(base, headers=headers, timeout=10)
    if user_res.status_code == 404:
        return {"error": f"GitHub user '{username}' not found"}
    if user_res.status_code != 200:
        return {"error": f"GitHub API error: {user_res.status_code}"}

    user = user_res.json()

    repos_res = requests.get(f"{base}/repos?per_page=30&sort=updated", headers=headers, timeout=10)
    repos     = repos_res.json() if repos_res.status_code == 200 else []

    # Build compact repo summaries
    repo_summaries = []
    for r in repos[:15]:
        if isinstance(r, dict):
            repo_summaries.append({
                "name":        r.get("name",""),
                "description": (r.get("description") or "")[:120],
                "language":    r.get("language",""),
                "stars":       r.get("stargazers_count", 0),
                "forks":       r.get("forks_count", 0),
                "updated":     (r.get("updated_at") or "")[:10],
                "topics":      r.get("topics", [])[:5]
            })

    languages = {}
    for r in repos:
        if isinstance(r, dict) and r.get("language"):
            languages[r["language"]] = languages.get(r["language"], 0) + 1

    return {
        "username":      user.get("login",""),
        "name":          user.get("name",""),
        "bio":           user.get("bio",""),
        "public_repos":  user.get("public_repos", 0),
        "followers":     user.get("followers", 0),
        "following":     user.get("following", 0),
        "created_at":    (user.get("created_at") or "")[:10],
        "updated_at":    (user.get("updated_at") or "")[:10],
        "company":       user.get("company",""),
        "location":      user.get("location",""),
        "blog":          user.get("blog",""),
        "top_languages": languages,
        "repos":         repo_summaries,
        "total_stars":   sum(r.get("stargazers_count",0) for r in repos if isinstance(r,dict)),
        "total_forks":   sum(r.get("forks_count",0)     for r in repos if isinstance(r,dict)),
    }

@app.route('/api/analyze-github', methods=['POST'])
def analyze_github():
    """
    Key Novelty 04: GitHub Analysis System.
    Fetches public GitHub profile → AI analysis of:
    - Code quality signals, contribution frequency, project relevance
    - Language consistency with resume claims
    - Technical depth evidence
    - Red flags and credibility boosters
    """
    data       = request.get_json()
    username   = (data.get('username') or '').strip().lstrip('@').split('/')[-1]
    resume     = data.get('resume_text', '')
    role_type  = data.get('role_type', 'software engineer')

    if not username:
        return jsonify({"error": "GitHub username required"}), 400

    # Fetch real GitHub data
    gh = fetch_github_data(username)
    if "error" in gh:
        return jsonify(gh), 404

    schema = {
        "github_score":            0,
        "activity_level":          "Inactive|Low|Moderate|Active|Highly Active",
        "technical_depth_score":   0,
        "code_quality_signals":    ["string"],
        "contribution_assessment": "string",
        "language_consistency":    "Consistent|Partial|Inconsistent|Cannot Verify",
        "language_consistency_reason": "string",
        "top_skills_evidenced":    ["string"],
        "project_highlights": [
            {"name":"string","relevance":"High|Medium|Low","why":"string","stars":0}
        ],
        "resume_vs_github": [
            {"claim":"string","github_evidence":"string","verdict":"Verified|Partial|Unverified|Contradicted"}
        ],
        "red_flags":    ["string"],
        "credibility_boosters": ["string"],
        "hire_signal":  "Strong Signal|Positive Signal|Neutral|Weak Signal|Concern",
        "hire_reason":  "string",
        "interview_probes": ["string"],
        "summary":      "string"
    }

    repo_text    = "\n".join([
        f"- {r['name']} ({r['language'] or 'unknown'}): {r['description'] or 'no description'} | Stars:{r['stars']} Forks:{r['forks']} Updated:{r['updated']}"
        for r in gh['repos']
    ])
    lang_text    = ", ".join([f"{k}:{v} repos" for k, v in sorted(gh['top_languages'].items(), key=lambda x:-x[1])[:8]])
    resume_ctx   = f"\nCandidate Resume:\n{resume[:800]}\n" if resume else ""

    prompt = f"""GitHub Profile Analysis for hiring decision:

Username: {gh['username']}
Name: {gh['name']}
Bio: {gh['bio']}
Account age: since {gh['created_at']}
Public repos: {gh['public_repos']} | Followers: {gh['followers']}
Total stars earned: {gh['total_stars']}
Languages: {lang_text}

Top Repositories:
{repo_text}
{resume_ctx}
Role being hired for: {role_type}

Analyze this GitHub profile for: code quality signals, contribution frequency, 
project relevance, language consistency with resume claims, and provide a hire signal.
Flag any red flags (empty repos, no commits, irrelevant projects) and credibility boosters."""

    try:
        result = call_groq(prompt, schema)
        result["raw_github"] = {
            "username":     gh["username"],
            "name":         gh["name"],
            "bio":          gh["bio"],
            "public_repos": gh["public_repos"],
            "followers":    gh["followers"],
            "created_at":   gh["created_at"],
            "total_stars":  gh["total_stars"],
            "top_languages":gh["top_languages"],
            "repos_fetched":len(gh["repos"])
        }
        result["profile_url"] = f"https://github.com/{gh['username']}"
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  KEY NOVELTY 06 — REPLAY INTELLIGENCE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/generate-replay', methods=['POST'])
def generate_replay():
    """
    Key Novelty 06: Build interview replay timeline with AI narrative.
    Generates timestamped timeline, flagged events, strength moments, hire narrative.
    """
    data    = request.get_json()
    history = data.get('history', [])
    resume  = data.get('resume_text', '')
    if not history:
        return jsonify({"error": "Session history required"}), 400

    schema = {
        "session_summary":          "string",
        "total_duration_estimate":  "string",
        "candidate_arc":            "string",
        "timeline": [
            {
                "timestamp":   "M:SS",
                "type":        "flag|strength|neutral|milestone",
                "event":       "string",
                "detail":      "string",
                "significance":"high|medium|low",
                "badge":       "string"
            }
        ],
        "key_moments": [
            {"moment":"string","type":"positive|negative|neutral","impact":"string"}
        ],
        "hire_narrative":           "string",
        "hire_signal":              "Strong Yes|Lean Yes|Neutral|Lean No|Strong No",
        "confidence":               0,
        "debrief_talking_points":   ["string"],
        "areas_to_probe_next_round":["string"]
    }
    history_text = "\n\n".join([
        f"[Q{i+1}] Q: {h['q']}\nA: {h['a'][:300]}\n"
        f"Consistency: {h.get('result',{}).get('resume_consistency','?') if h.get('result') else '?'}\n"
        f"Flags: {len(h.get('result',{}).get('red_flags',[]) if h.get('result') else [])}"
        for i, h in enumerate(history)
    ])
    ctx = f"Resume:\n{resume[:500]}\n\n" if resume else ""
    try:
        return jsonify(call_groq(
            f"{ctx}Full interview session:\n{history_text}\n\n"
            "Generate complete replay intelligence: timeline, key moments, hire narrative for debrief meeting.",
            schema
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  LAYER 4 — OUTPUT / METRICS
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/accuracy-metrics', methods=['POST'])
def accuracy_metrics():
    """Real precision/recall/F1 from actual session history."""
    data    = request.get_json()
    history = data.get('history', [])
    if not history:
        return jsonify({"error": "Session history required"}), 400

    tp=fp=fn=tn=0
    scores_td, scores_cl, scores_sp = [], [], []
    flags_total=flags_high=flags_med=flags_low=0

    for h in history:
        r   = h.get('result',{}) or {}
        c   = (r.get('resume_consistency','') or '').lower()
        flags = r.get('red_flags',[]) or []
        sc  = r.get('scores',{}) or {}
        if 'contradiction' in c:  tp += 1
        elif 'partial' in c:      fp += 1; fn += 1
        elif 'aligned' in c:      tn += 1
        else:                     fn += 1
        flags_total += len(flags)
        for f in flags:
            sev = (f.get('severity') or '').lower()
            if sev=='high':       flags_high+=1
            elif sev=='medium':   flags_med+=1
            else:                 flags_low+=1
        if sc.get('technical_depth'): scores_td.append(sc['technical_depth'])
        if sc.get('clarity'):         scores_cl.append(sc['clarity'])
        if sc.get('specificity'):     scores_sp.append(sc['specificity'])

    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    accuracy  = (tp+tn)/len(history) if history else 0.0

    return jsonify({
        "total_responses": len(history),
        "confusion_matrix": {"tp":tp,"fp":fp,"fn":fn,"tn":tn},
        "contradiction_detection": {
            "precision":round(precision,3),"recall":round(recall,3),
            "f1_score":round(f1,3),"accuracy":round(accuracy,3)
        },
        "answer_quality": {
            "avg_technical_depth": round(sum(scores_td)/len(scores_td),2) if scores_td else 0,
            "avg_clarity":         round(sum(scores_cl)/len(scores_cl),2) if scores_cl else 0,
            "avg_specificity":     round(sum(scores_sp)/len(scores_sp),2) if scores_sp else 0
        },
        "red_flags": {"total":flags_total,"high":flags_high,"medium":flags_med,"low":flags_low},
        "vector_engine": "sentence-transformers + ChromaDB" if (_ST_AVAILABLE and _CHROMA_AVAILABLE) else "LLM-only",
        "stt_engine":    f"Groq {GROQ_WHISPER}"
    })


# ─── In-memory PIN / session store (use Redis/DB in production) ───────────────
@app.route('/api/create-session', methods=['POST'])
def create_session():

    data = request.get_json()

    name = data.get("name", "Interviewer")
    role = data.get("role", "host")
    position = data.get("position", "")
    pin = str(data.get("pin", "")).strip()

    import random, time as _time, secrets

    # Generate PIN if invalid
    if not pin or len(pin) != 6 or not pin.isdigit():
        for _ in range(20):
            pin = str(random.randint(100000, 999999))
            if _session_get(pin) is None:
                break

    room_name = f"nexhire-{pin}"
    session_id = f"SID-{int(_time.time()*1000)}"

    session_doc = {
        "pin": pin,
        "room_name": room_name,
        "session_id": session_id,
        "interviewer": name,
        "role": role,
        "position": position,
        "created_at": _time.time(),
        "candidate_joined": False
    }

    candidate_token = secrets.token_urlsafe(16)
    candidate_link = f"http://localhost:5000/join/{candidate_token}"

    _session_set(pin, session_doc)

    return jsonify({
        "pin": pin,
        "room_name": room_name,
        "session_id": session_id,
        "candidate_link": candidate_link
    })
import random, time as _time

# Val



@app.route('/api/debug-pin', methods=['GET'])
def debug_pin():
    """Debug endpoint - shows what's stored for a PIN. Remove in production."""
    pin = request.args.get('pin', '').strip()
    if not pin:
        return jsonify({'error': 'pass ?pin=XXXXXX'})
    # Check Atlas
    atlas_doc = None
    atlas_err = None
    if _MONGO_OK and _sessions_col is not None:
        try:
            atlas_doc = _sessions_col.find_one({'pin': pin}, {'_id': 0})
        except Exception as e:
            atlas_err = str(e)
    # Check SQLite
    sqlite_doc = None
    try:
        with sqlite3.connect(_DB_PATH) as con:
            row = con.execute('SELECT data FROM sessions WHERE pin=?', (pin,)).fetchone()
            sqlite_doc = json.loads(row[0]) if row else None
    except Exception as e:
        sqlite_doc = {'error': str(e)}
    # List all SQLite pins
    all_pins = []
    try:
        with sqlite3.connect(_DB_PATH) as con:
            rows = con.execute('SELECT pin, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 20').fetchall()
            all_pins = [{'pin': r[0], 'updated_at': r[1]} for r in rows]
    except: pass
    return jsonify({
        'mongo_ok': _MONGO_OK,
        'queried_pin': pin,
        'atlas_doc': atlas_doc,
        'atlas_error': atlas_err,
        'sqlite_doc': sqlite_doc,
        'all_sqlite_pins': all_pins
    })
@app.route("/join/<token>")
def join_candidate(token):
    # For now just open the main page
    return render_template("index.html", role="candidate", token=token)
@app.route('/api/validate-pin', methods=['POST'])
def validate_pin():
    """
    Called when candidate submits their PIN.
    Returns valid:true ONLY if an interviewer has already created
    a real session with that PIN.  No demo-mode fallback.
    """
    data           = request.get_json()
    pin            = str(data.get('pin', '')).strip()
    candidate_name = data.get('candidate_name', 'Candidate')

    if not pin or len(pin) != 6 or not pin.isdigit():
        return jsonify({"valid": False, "error": "PIN must be exactly 6 digits"}), 200

    session = _session_get(pin)
    print(f"[validate-pin] pin={pin} -> session={session}")
    if session is None:
        # Last resort: check SQLite directly in case Atlas write succeeded but read failed
        import sqlite3 as _sq
        import json as _js
        try:
            with sqlite3.connect(_DB_PATH) as _con:
                _row = _con.execute("SELECT data FROM sessions WHERE pin=?", (pin,)).fetchone()
                if _row:
                    session = json.loads(_row[0])
                    print(f"[validate-pin] found in SQLite fallback: {session}")
        except Exception as _e:
            print(f"[validate-pin] SQLite fallback error: {_e}")
    if session is None:
        return jsonify({"valid": False, "error": "No active session found for this PIN. Ask your interviewer to share the correct PIN."}), 200

    # Mark candidate as joined
    _session_update(pin, {"candidate_joined": True, "candidate_name": candidate_name})
    session["candidate_joined"] = True
    session["candidate_name"]   = candidate_name

    return jsonify({"valid": True, "session": session})


@app.route('/api/session-status', methods=['GET'])
def session_status():
    """Lightweight endpoint: check if a PIN exists (used by frontend health check)."""
    pin = str(request.args.get('pin', '')).strip()
    if not pin:
        return jsonify({"exists": False}), 200
    session = _session_get(pin)
    if session:
        return jsonify({"exists": True, "interviewer": session.get("interviewer",""), "position": session.get("position","")})
    return jsonify({"exists": False}), 200


@app.route('/api/candidate-resume', methods=['POST'])
def candidate_resume():
    """Candidate uploads their resume PDF; text is stored in session for interviewer."""
    pin  = request.form.get('pin', '').strip()
    file = request.files.get('resume')

    if not pin or len(pin) != 6 or not pin.isdigit():
        return jsonify({"error": "Invalid PIN"}), 400

    session = _session_get(pin)
    if session is None:
        return jsonify({"error": "Session not found for this PIN"}), 400

    resume_text = ""
    if file and file.filename:
        try:
            pdf_bytes   = file.read()
            resume_text = extract_text_from_pdf(pdf_bytes)
        except Exception as e:
            return jsonify({"error": f"Could not parse PDF: {e}"}), 400

    _session_update(pin, {"candidate_resume": resume_text})
    return jsonify({"ok": True, "chars": len(resume_text),
                    "preview": resume_text[:200] if resume_text else ""})


@app.route('/api/get-candidate-resume', methods=['POST'])
def get_candidate_resume():
    """Interviewer fetches the candidate's uploaded resume text."""
    data = request.get_json()
    pin  = str(data.get('pin', '')).strip()
    session = _session_get(pin)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    text = session.get('candidate_resume', '')
    return jsonify({"resume_text": text, "chars": len(text)})


#  Replaces Zoom / Google Meet / Webex with self-hosted WebRTC SFU
#  Install LiveKit server: docker run -p 7880:7880 livekit/livekit-server --dev
#  Install Python SDK:     pip install livekit-api
# ═══════════════════════════════════════════════════════════════════

# ── LiveKit credentials ──────────────────────────────────────────────────────
# Set these three env vars before running Flask:
#   export LIVEKIT_API_KEY=APIxxxxxxxxxxxxxxxxx
#   export LIVEKIT_API_SECRET=your_livekit_secret_here
#   export LIVEKIT_URL=wss://your-project.livekit.cloud
# For LiveKit Cloud free tier: https://cloud.livekit.io
# ─────────────────────────────────────────────────────────────────────────────
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise RuntimeError("LiveKit credentials not set in environment variables")
def _lk_make_jwt(identity: str, display_name: str, room: str) -> str:
    """
    Pure-stdlib HS256 JWT that satisfies LiveKit's token format.

    Fixes vs the previous implementation
    ─────────────────────────────────────
    BUG-A  Missing `nbf` (not-before) claim — LiveKit Cloud enforces it
           and rejects tokens without it with a 401.
    BUG-B  Missing `canPublishData: true` — required for LiveKit data
           channel messages (used by the candidate integrity monitor).
    BUG-C  json.dumps() without separators=(',',':') produced JSON with
           spaces, making the base64 payload differ from what the SDK
           produces and occasionally failing strict parsers.
    BUG-D  Token TTL was 86400 s (24 h). Changed to 21600 s (6 h) —
           enough for any interview; shorter = better security.
    """
    import base64 as _b, hmac as _h, hashlib as _hs, json as _j, time as _t

    def _b64u(data: bytes) -> str:
        return _b.urlsafe_b64encode(data).rstrip(b'=').decode()

    now = int(_t.time())
    hdr = _b64u(_j.dumps({"alg": "HS256", "typ": "JWT"}, separators=(',', ':')).encode())
    pld = _b64u(_j.dumps({
        "iss":  LIVEKIT_API_KEY,
        "sub":  identity,
        "name": display_name,
        "iat":  now,
        "nbf":  now,           # ← was missing; required by LiveKit Cloud
        "exp":  now + 21600,
        "video": {
            "roomJoin":       True,
            "room":           room,
            "canPublish":     True,
            "canSubscribe":   True,
            "canPublishData": True,   # ← was missing
        },
    }, separators=(',', ':')).encode())

    sig = _b64u(
        _h.new(LIVEKIT_API_SECRET.encode(), f"{hdr}.{pld}".encode(), _hs.sha256).digest()
    )
    return f"{hdr}.{pld}.{sig}"


@app.route('/api/livekit-token', methods=['POST'])
def livekit_token():
    """
    Generate and return a LiveKit access token.

    Called by the browser (Video Room tab + Candidate portal) immediately
    before room.connect().  Supports two code paths:

      livekit-sdk  Uses the official livekit-api Python package.
                   Recommended once `pip install livekit-api` is done.
      stdlib-jwt   Pure-stdlib HS256 JWT — zero extra dependencies.
                   Works identically with real LiveKit Cloud credentials.

    Request JSON
    ─────────────
      room      room name, e.g. "nexhire-111111"
      identity  unique participant id, e.g. "arjun-host"
      role      "host" | "technical" | "devops" | "hr" | "candidate"
      name      (optional) display-name override

    Response JSON
    ─────────────
      token       signed JWT passed directly to room.connect()
      server_url  LiveKit WSS URL (from LIVEKIT_URL env var)
      mode        "livekit-sdk" or "stdlib-jwt"
    """
    body     = request.get_json(silent=True) or {}
    room     = (body.get('room')     or 'nexhire-interview').strip().replace(' ', '-')
    identity = (body.get('identity') or 'participant').strip()
    role     = (body.get('role')     or 'interviewer').strip()
    name     = (body.get('name')     or identity).strip()
    display_name = f"{name} ({role})" if role.lower() not in name.lower() else name

    # ── Path 1: official livekit-api SDK ─────────────────────────────────────
    try:
        from livekit.api import AccessToken, VideoGrants          # type: ignore
        grants = VideoGrants(
            room_join=True,
            room=room,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        token = (
            AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(display_name)
            .with_grants(grants)
            .to_jwt()
        )
        print(f"[LK] SDK token  room={room}  identity={identity}")
        return jsonify({"token": token, "server_url": LIVEKIT_URL,
                        "mode": "livekit-sdk", "room": room, "identity": identity})

    except ImportError:
        pass  # fall through to stdlib path

    # ── Path 2: stdlib JWT (no extra dependencies needed) ────────────────────
    try:
        token = _lk_make_jwt(identity=identity, display_name=display_name, room=room)
        print(f"[LK] stdlib JWT  room={room}  identity={identity}")
        return jsonify({"token": token, "server_url": LIVEKIT_URL,
                        "mode": "stdlib-jwt", "room": room, "identity": identity})

    except Exception as exc:
        print(f"[LK] token error: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route('/api/groq-json', methods=['POST'])
def groq_json():
    """Proxy: frontend calls this instead of Groq directly. Returns parsed JSON."""
    data       = request.get_json()
    system_msg = data.get('system', '')
    user_msg   = data.get('user', '')
    max_tokens = int(data.get('max_tokens', 2000))
    try:
        comp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.75,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        raw = comp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        return jsonify({"result": json.loads(raw)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/groq-direct', methods=['POST'])
def groq_direct():
    """Proxy: frontend calls this for plain-text Groq responses."""
    data       = request.get_json()
    system_msg = data.get('system', '')
    user_msg   = data.get('user', '')
    max_tokens = int(data.get('max_tokens', 400))
    try:
        comp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.6,
            max_tokens=max_tokens
        )
        return jsonify({"result": comp.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        import livekit.api
        livekit_status = "livekit-api SDK installed ✅"
    except ImportError:
        livekit_status = "dev-jwt mode (pip install livekit-api for production)"

    return jsonify({
        "status":              "ok",
        "llm_model":           GROQ_MODEL,
        "stt_model":           GROQ_WHISPER,
        "vector_engine":       "sentence-transformers + ChromaDB" if (_ST_AVAILABLE and _CHROMA_AVAILABLE) else "LLM-only",
        "sentence_transformers":_ST_AVAILABLE,
        "chromadb":            _CHROMA_AVAILABLE,
        "github_analysis":     "active",
        "livekit":             livekit_status,
        "video_platform":      "LiveKit WebRTC SFU (self-hosted, replaces Zoom/Meet)",
        "novelties_implemented": [
            "01 - AI Interview Copilot",
            "02 - Simulation-Based Interviewing",
            "03 - Integrity Monitoring",
            "04 - GitHub Analysis System",
            "05 - Gamified Simulation",
            "06 - Replay Intelligence",
            "07 - LiveKit Self-Hosted Video (replaces Zoom)"
        ]
    })




# ═══════════════════════════════════════════════════════════════════
#  REAL-TIME LAYER — Flask-SocketIO Event Handlers
#  Room identifier = session PIN (6-digit string)
# ═══════════════════════════════════════════════════════════════════

@socketio.on('connect')
def on_connect():
    print(f"[WS] Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    # Remove from _room_members on disconnect
    for pin, roles in list(_room_members.items()):
        for role, stored_sid in list(roles.items()):
            if stored_sid == sid:
                del _room_members[pin][role]
                print(f"[WS] {role} disconnected from PIN {pin}")
                # Notify remaining participant
                emit('peer_disconnected', {'role': role}, room=pin)
    print(f"[WS] Client disconnected: {sid}")

def _handle_join(data):
    """
    Core join logic shared by join_session and join_room.
    Both interviewer and candidate call this after PIN validation.
    data: { pin, role, name? }   role = 'interviewer' | 'candidate'
    """
    pin  = str(data.get('pin', '')).strip()
    role = data.get('role', 'candidate')
    name = data.get('name', '')
    sid  = request.sid

    if not pin or len(pin) != 6:
        emit('error', {'message': 'Invalid PIN'})
        return

    # Add to SocketIO room (pin is the room name)
    join_room(pin)

    # Track in memory
    if pin not in _room_members:
        _room_members[pin] = {}
    _room_members[pin][role] = sid

    # Update session metadata in DB
    update_fields = {
        f"{role}_socket_id": sid,
        f"{role}_online":    True
    }
    if role == 'candidate' and name:
        update_fields['candidate_name'] = name
    _session_update(pin, update_fields)

    members = list(_room_members[pin].keys())
    print(f"[WS] {role} joined PIN={pin} sid={sid}  room_members={members}")

    # Notify the other participant that a peer has joined
    emit('peer_joined', {'role': role, 'name': name}, room=pin, include_self=False)

    # Acknowledge to the joiner — include full member list so frontend can
    # decide immediately whether to initiate WebRTC (both sides present)
    emit('session_joined', {
        'pin':     pin,
        'role':    role,
        'members': members,
        'both_present': ('interviewer' in members and 'candidate' in members)
    })

    # For candidates: also emit candidate_joined directly to their socket
    # so the frontend can reliably trigger the portal-open transition
    if role == 'candidate':
        session = _session_get(pin) or {}
        emit('candidate_joined', {
            'pin':        pin,
            'name':       name or session.get('candidate_name', 'Candidate'),
            'interviewer': session.get('interviewer', ''),
            'position':   session.get('position', ''),
            'room_name':  session.get('roomName', f'nexhire-{pin}')
        })
        print(f"[WS] candidate_joined emitted → {sid}")

        # ── Deliver any pending game/simulation event that fired before
        #    the candidate connected (race condition fix) ──────────────
        pending = _pending_game_events.pop(pin, None)
        if pending:
            socketio.emit(pending['event'], pending['data'], room=sid)
            print(f"[WS] Delivered pending '{pending['event']}' to late-joining candidate PIN={pin}")

    # If both sides are now present, tell the interviewer to start WebRTC offer
    # This handles the case where the interviewer joined first and is waiting.
    if 'interviewer' in members and 'candidate' in members:
        interviewer_sid = _room_members[pin].get('interviewer')
        if interviewer_sid:
            socketio.emit('both_participants_ready', {
                'pin': pin,
                'members': members
            }, room=interviewer_sid)
            print(f"[WS] both_participants_ready → interviewer {interviewer_sid} PIN={pin}")


@socketio.on('join_session')
def on_join_session(data):
    """Alias: primary join event name used by the frontend."""
    _handle_join(data)


@socketio.on('join_room')
def on_join_room(data):
    """
    Spec-required alias for join_session.
    Some frontend paths emit 'join_room' instead of 'join_session'.
    data: { pin, role, name? }
    """
    _handle_join(data)


# ─── 1. Live Copilot Q&A Loop ────────────────────────────────────────────────

@socketio.on('send_question')
def on_send_question(data):
    """
    Interviewer → candidate.
    data: { pin, question, question_id }
    """
    pin      = str(data.get('pin', '')).strip()
    question = data.get('question', '').strip()
    q_id     = data.get('question_id', '')

    if not pin or not question:
        emit('error', {'message': 'pin and question required'})
        return

    # Persist current_question in session
    _session_update(pin, {'current_question': question, 'question_id': q_id})

    print(f"[WS] send_question PIN={pin}: {question[:60]}")

    # Broadcast to everyone in the room (interviewer sees their own sent message too)
    emit('receive_question', {
        'question':    question,
        'question_id': q_id,
        'from':        'interviewer',
        'pin':         pin
    }, room=pin)


@socketio.on('candidate_answer')
def on_candidate_answer(data):
    """
    Candidate → interviewer (+ optional copilot analysis trigger).
    data: { pin, answer, question, question_id, resume_text, history }
    """
    pin         = str(data.get('pin', '')).strip()
    answer      = data.get('answer', '').strip()
    question    = data.get('question', '').strip()
    q_id        = data.get('question_id', '')
    resume_text = data.get('resume_text', '')
    history     = data.get('history', [])

    if not pin or not answer:
        emit('error', {'message': 'pin and answer required'})
        return

    # Persist answer in live_transcript
    session = _session_get(pin) or {}
    transcript = session.get('live_transcript', [])
    transcript.append({'q': question, 'a': answer, 'q_id': q_id})
    _session_update(pin, {'live_transcript': transcript})

    print(f"[WS] candidate_answer PIN={pin}: {answer[:60]}")

    # Immediately forward the answer to everyone in the room
    emit('receive_answer', {
        'answer':      answer,
        'question':    question,
        'question_id': q_id,
        'from':        'candidate',
        'pin':         pin
    }, room=pin)

    # Trigger async copilot analysis if resume context is available
    if question and answer:
        def _analyze():
            try:
                schema = {
                    "resume_consistency": "Aligned|Partial|Contradiction|Not Applicable",
                    "scores": {"technical_depth": 0, "clarity": 0, "specificity": 0},
                    "red_flags": [{"type": "string", "description": "string", "severity": "low|medium|high"}],
                    "follow_up_questions": [{"question": "string", "category": "Technical|Behavioral|Clarification"}],
                    "coaching_tip": "string",
                    "overall_assessment": "string",
                    "answer_completeness": "Complete|Partial|Vague|Evasive"
                }
                ctx     = (f"Resume:\n{resume_text[:1000]}\n\n") if resume_text else ""
                hist_ctx = ("Prior exchanges:\n" + "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]]) + "\n\n") if history else ""
                prompt  = f"{ctx}{hist_ctx}Question: {question}\nCandidate Answer: {answer}\n\nProvide live copilot analysis."
                result  = call_groq(prompt, schema)
                # Emit copilot analysis update to interviewer only
                interviewer_sid = _room_members.get(pin, {}).get('interviewer')
                if interviewer_sid:
                    socketio.emit('copilot_analysis_update', {
                        'question_id': q_id,
                        'question':    question,
                        'answer':      answer,
                        'analysis':    result,
                        'pin':         pin
                    }, room=interviewer_sid)
                    print(f"[WS] copilot_analysis_update sent to interviewer PIN={pin}")
            except Exception as e:
                print(f"[WS] Copilot analysis error: {e}")

        import threading
        threading.Thread(target=_analyze, daemon=True).start()


# ─── 2. Simulation Sync ───────────────────────────────────────────────────────

@socketio.on('start_simulation')
def on_start_simulation(data):
    """
    Interviewer launches simulation.
    Routes two DIFFERENT events to two DIFFERENT socket IDs:
      - 'simulation_started'  → candidate only  (runs simulation engine)
      - 'simulation_monitor'  → interviewer only (shows monitor dashboard)

    This replaces the previous `emit(..., room=pin)` broadcast which sent
    the same event to everyone in the room, causing both UIs to run the
    simulation engine simultaneously and corrupting the candidate's score.

    data: { pin, sim_type, difficulty }
    """
    pin        = str(data.get('pin', '')).strip()
    sim_type   = data.get('sim_type', 'incident')
    difficulty = data.get('difficulty', 'medium')

    if not pin:
        emit('error', {'message': 'pin required'})
        return

    # Persist simulation state so late-joining clients can recover it
    _session_update(pin, {
        'simulation_state': 'active',
        'sim_type':         sim_type,
        'sim_difficulty':   difficulty
    })

    members = _room_members.get(pin, {})
    candidate_sid   = members.get('candidate')
    interviewer_sid = members.get('interviewer')

    print(f"[WS] start_simulation PIN={pin} type={sim_type} diff={difficulty} "
          f"candidate_sid={candidate_sid} interviewer_sid={interviewer_sid}")

    # ── Emit to CANDIDATE only ───────────────────────────────────────────────
    # The candidate receives 'simulation_started' and runs simGenerateNew().
    # If the candidate is not yet connected, log a warning — the event is lost
    # (the client should call sync_session_state on reconnect to recover).
    if candidate_sid:
        socketio.emit('simulation_started', {
            'pin':        pin,
            'sim_type':   sim_type,
            'difficulty': difficulty,
            'message':    'Interviewer has started a simulation. Your screen will now show the challenge.'
        }, room=candidate_sid)
        print(f"[WS] simulation_started → candidate {candidate_sid}")
    else:
        # Queue for late-joining candidate
        _pending_game_events[pin] = {'event': 'simulation_started', 'data': {
            'pin': pin, 'sim_type': sim_type, 'difficulty': difficulty,
            'message': 'Interviewer has started a simulation. Your screen will now show the challenge.'
        }}
        print(f"[WS] WARNING: no candidate connected for PIN={pin}, simulation_started queued")

    # ── Emit to INTERVIEWER only ─────────────────────────────────────────────
    # The interviewer receives 'simulation_monitor' and shows only the
    # monitoring dashboard. They NEVER receive 'simulation_started' so their
    # browser never triggers simGenerateNew() or simSelectOption().
    if interviewer_sid:
        socketio.emit('simulation_monitor', {
            'pin':        pin,
            'sim_type':   sim_type,
            'difficulty': difficulty,
            'message':    'Simulation sent to candidate. Monitoring dashboard active.'
        }, room=interviewer_sid)
        print(f"[WS] simulation_monitor → interviewer {interviewer_sid}")
    else:
        print(f"[WS] WARNING: no interviewer connected for PIN={pin}, simulation_monitor not delivered")


@socketio.on('start_game')
def on_start_game(data):
    """
    Interviewer launches Kilimanjaro Challenge game.
    Routes targeted events — game_started → candidate, game_monitor → interviewer.
    Frontend emits: { pin, difficulty }
    """
    pin        = str(data.get('pin', '')).strip()
    difficulty = data.get('difficulty', 'medium')

    if not pin:
        emit('error', {'message': 'pin required'})
        return

    _session_update(pin, {
        'game_state':      'active',
        'game_difficulty': difficulty,
        'game_score':      0,
        'game_correct':    0,
        'game_altitude':   0,
        'game_streak':     0,
    })

    members         = _room_members.get(pin, {})
    candidate_sid   = members.get('candidate')
    interviewer_sid = members.get('interviewer')

    print(f"[WS] start_game PIN={pin} diff={difficulty} "
          f"candidate={candidate_sid} interviewer={interviewer_sid}")

    if candidate_sid:
        socketio.emit('game_started', {
            'pin':        pin,
            'difficulty': difficulty,
            'message':    'Kilimanjaro Challenge launched — good luck!'
        }, room=candidate_sid)
        print(f"[WS] game_started → candidate {candidate_sid}")
    else:
        # ── RACE-CONDITION FIX: store the event so it fires when candidate joins ──
        game_payload = {
            'pin':        pin,
            'difficulty': difficulty,
            'message':    'Kilimanjaro Challenge launched — good luck!'
        }
        _pending_game_events[pin] = {'event': 'game_started', 'data': game_payload}
        print(f"[WS] WARNING: no candidate for PIN={pin} — game_started queued for delivery on join")

    if interviewer_sid:
        socketio.emit('game_monitor', {
            'pin':        pin,
            'difficulty': difficulty,
            'message':    'Challenge sent to candidate. Monitor dashboard active.'
        }, room=interviewer_sid)
        print(f"[WS] game_monitor → interviewer {interviewer_sid}")


@socketio.on('game_answer')
def on_game_answer(data):
    """
    Candidate submits a game answer → game_update forwarded to interviewer.
    data: { pin?, option_idx, correct, score, altitude, streak, correct_count, question_id, ts }
    """
    pin = str(data.get('pin', '')).strip()
    if not pin:
        sid = request.sid
        pin = next((p for p, m in _room_members.items() if m.get('candidate') == sid), '')
    if not pin:
        print(f"[WS] game_answer: no pin found, dropping")
        return

    correct       = bool(data.get('correct', False))
    option_idx    = data.get('option_idx', -1)
    question_id   = data.get('question_id', '')
    ts            = data.get('ts', '')

    # Trust client-sent scores (candidate engine tracks them precisely).
    # Fall back to DB-incremented values only if client didn't send them.
    if data.get('score') is not None:
        score         = int(data.get('score', 0))
        correct_count = int(data.get('correct_count', 0))
        altitude      = int(data.get('altitude', 0))
        streak        = int(data.get('streak', 0))
    else:
        session       = _session_get(pin) or {}
        score         = session.get('game_score', 0)    + (10 if correct else 0)
        correct_count = session.get('game_correct', 0)  + (1  if correct else 0)
        altitude      = session.get('game_altitude', 0) + (50 if correct else 0)
        streak        = session.get('game_streak', 0)
        streak        = streak + 1 if correct else 0

    _session_update(pin, {
        'game_score':    score,
        'game_correct':  correct_count,
        'game_altitude': altitude,
        'game_streak':   streak,
    })

    interviewer_sid = _room_members.get(pin, {}).get('interviewer')
    monitor_payload = {
        'pin':             pin,
        'correct':         correct,
        'option_idx':      option_idx,
        'selected_option': option_idx,
        'question_id':     question_id,
        'score':           score,
        'correct_count':   correct_count,
        'altitude':        altitude,
        'streak':          streak,
        'ts':              ts,
    }
    if interviewer_sid:
        # Targeted emit to interviewer SID (legacy game_update path)
        socketio.emit('game_update', monitor_payload, room=interviewer_sid)
        print(f"[WS] game_update → interviewer {interviewer_sid} PIN={pin} score={score}")
    else:
        print(f"[WS] WARNING: game_answer received but no interviewer in room PIN={pin}")

    # Also broadcast monitor_update to the whole room — belt-and-suspenders:
    # ensures the interviewer receives data even if their SID lookup is stale.
    socketio.emit('monitor_update', monitor_payload, room=pin)
    print(f"[WS] monitor_update → room {pin} score={score}")


# ─── candidate_progress → monitor_update ────────────────────────────────────
# This is the canonical real-time pipeline required by the spec.
# candidate_progress  : emitted by candidate after every answered question
# monitor_update      : broadcast to the entire PIN room so interviewer receives it
# game_answer         : legacy alias — kept for backward compatibility, also triggers
#                       monitor_update so the interviewer always gets data regardless
#                       of which event the candidate front-end sends.

@socketio.on('candidate_progress')
def on_candidate_progress(data):
    """
    Canonical event: candidate sends progress after every answer.
    Broadcasts monitor_update to the whole PIN room so the interviewer
    monitor always receives it regardless of which SID is registered.

    data: { pin, score, altitude, streak, correct_count, question_id,
            correct, option_idx, ts }
    """
    pin = str(data.get('pin', '')).strip()
    if not pin:
        sid = request.sid
        pin = next((p for p, m in _room_members.items()
                    if m.get('candidate') == sid), '')
    if not pin:
        print("[WS] candidate_progress: no pin, dropping")
        return

    score         = int(data.get('score', 0))
    altitude      = int(data.get('altitude', 0))
    streak        = int(data.get('streak', 0))
    correct_count = int(data.get('correct_count', 0))
    correct       = bool(data.get('correct', False))
    question_id   = data.get('question_id', '')
    option_idx    = data.get('option_idx', -1)

    _session_update(pin, {
        'game_score':    score,
        'game_correct':  correct_count,
        'game_altitude': altitude,
        'game_streak':   streak,
    })

    payload = {
        'pin':           pin,
        'score':         score,
        'altitude':      altitude,
        'streak':        streak,
        'correct_count': correct_count,
        'correct':       correct,
        'selected_option': option_idx,
        'question_id':   question_id,
        'ts':            data.get('ts', ''),
    }

    # Broadcast to the whole room — interviewer receives monitor_update,
    # candidate receives it too but the frontend ignores it based on role.
    emit('monitor_update', payload, room=pin)
    print(f"[WS] monitor_update → room {pin} score={score} alt={altitude} streak={streak}")


@socketio.on('launch_kilimanjaro')
def on_launch_kilimanjaro(data):
    """
    Spec-required alias for start_game.
    Interviewer launches the challenge; routes game_started → candidate,
    game_monitor → interviewer, and persists state.

    data: { pin, difficulty }
    """
    # Delegate entirely to on_start_game so logic stays in one place
    on_start_game(data)
    print(f"[WS] launch_kilimanjaro delegated to start_game PIN={data.get('pin','')}")


@socketio.on('game_finished')
def on_game_finished(data):
    """
    Candidate signals game completion → both sides get final results.
    data: { pin?, final_score, correct_count, total_questions, altitude_reached }
    """
    pin = str(data.get('pin', '')).strip()
    if not pin:
        sid = request.sid
        pin = next((p for p, m in _room_members.items() if m.get('candidate') == sid), '')
    if not pin:
        return

    session = _session_get(pin) or {}
    payload = {
        'pin':              pin,
        'final_score':      data.get('final_score',      session.get('game_score', 0)),
        'correct_count':    data.get('correct_count',    session.get('game_correct', 0)),
        'total_questions':  data.get('total_questions',  10),
        'altitude_reached': data.get('altitude_reached', session.get('game_altitude', 0)),
    }
    _session_update(pin, {'game_state': 'finished', **payload})

    members         = _room_members.get(pin, {})
    candidate_sid   = members.get('candidate')
    interviewer_sid = members.get('interviewer')

    if candidate_sid:   socketio.emit('game_finished', payload, room=candidate_sid)
    if interviewer_sid: socketio.emit('game_finished', payload, room=interviewer_sid)
    print(f"[WS] game_finished PIN={pin} score={payload['final_score']}")


@socketio.on('simulation_answer')
def on_simulation_answer(data):
    """
    Candidate submits a simulation answer → interviewer monitor updates.
    Routes to interviewer socket ONLY (not room-broadcast).
    data: { pin, selected_option, question_id, time_used, correct, score, streak }
    """
    pin = str(data.get('pin', '')).strip()
    if not pin:
        return

    _session_update(pin, {'last_sim_answer': data})

    interviewer_sid = _room_members.get(pin, {}).get('interviewer')
    if interviewer_sid:
        socketio.emit('simulation_answer_received', data, room=interviewer_sid)
        print(f"[WS] simulation_answer_received → interviewer {interviewer_sid} PIN={pin}")
    else:
        print(f"[WS] WARNING: no interviewer connected for PIN={pin}, answer update not delivered")


@socketio.on('end_simulation')
def on_end_simulation(data):
    """Interviewer ends simulation — both sides notified via targeted emits."""
    pin = str(data.get('pin', '')).strip()
    if not pin:
        return

    _session_update(pin, {'simulation_state': 'ended'})

    members         = _room_members.get(pin, {})
    candidate_sid   = members.get('candidate')
    interviewer_sid = members.get('interviewer')

    payload = {'pin': pin}
    if candidate_sid:
        socketio.emit('simulation_ended', payload, room=candidate_sid)
    if interviewer_sid:
        socketio.emit('simulation_ended', payload, room=interviewer_sid)

    print(f"[WS] simulation_ended PIN={pin}")


# ─── 3. WebRTC Video Signaling ────────────────────────────────────────────────
# Architecture: pure SocketIO relay — no LiveKit dependency.
#   Interviewer browser → webrtc_offer  → Flask → candidate browser
#   Candidate browser  → webrtc_answer → Flask → interviewer browser
#   Both sides         → ice_candidate → Flask → other side
#
# The server is a dumb relay: it reads the pin, finds the target SID
# from _room_members, and emits directly to that SID (not room-broadcast)
# to avoid echo back to the sender which can corrupt the signaling state.

@socketio.on('webrtc_offer')
def on_webrtc_offer(data):
    """
    Interviewer sends SDP offer → relay to candidate only.
    data: { pin, offer: RTCSessionDescriptionInit }
    """
    pin = str(data.get('pin', '')).strip()
    candidate_sid = _room_members.get(pin, {}).get('candidate')
    print(f"[WS] webrtc_offer  PIN={pin}  candidate_sid={candidate_sid}")
    if candidate_sid:
        socketio.emit('webrtc_offer', {'offer': data.get('offer'), 'pin': pin},
                      room=candidate_sid)
    else:
        print(f"[WS] webrtc_offer: no candidate in room PIN={pin} — queuing not supported")


@socketio.on('webrtc_answer')
def on_webrtc_answer(data):
    """
    Candidate sends SDP answer → relay to interviewer only.
    data: { pin, answer: RTCSessionDescriptionInit }
    """
    pin = str(data.get('pin', '')).strip()
    interviewer_sid = _room_members.get(pin, {}).get('interviewer')
    print(f"[WS] webrtc_answer PIN={pin}  interviewer_sid={interviewer_sid}")
    if interviewer_sid:
        socketio.emit('webrtc_answer', {'answer': data.get('answer'), 'pin': pin},
                      room=interviewer_sid)
    else:
        print(f"[WS] webrtc_answer: no interviewer in room PIN={pin}")


@socketio.on('ice_candidate')
def on_ice_candidate(data):
    """
    Exchange ICE candidates — relay to the other side only.
    data: { pin, candidate: RTCIceCandidateInit, role: 'interviewer'|'candidate' }
    """
    pin      = str(data.get('pin', '')).strip()
    role     = data.get('role', '')
    members  = _room_members.get(pin, {})

    # Send to whoever is NOT the sender
    target_role = 'candidate' if role == 'interviewer' else 'interviewer'
    target_sid  = members.get(target_role)

    if target_sid:
        socketio.emit('ice_candidate', {
            'candidate': data.get('candidate'),
            'role':      role,
            'pin':       pin
        }, room=target_sid)
    else:
        print(f"[WS] ice_candidate: no {target_role} in room PIN={pin}")


# ─── 4. Session State Sync ────────────────────────────────────────────────────

@socketio.on('sync_session_state')
def on_sync_session_state(data):
    """
    Any participant can request a full session state snapshot.
    data: { pin }
    """
    pin = str(data.get('pin', '')).strip()
    session = _session_get(pin)
    if session:
        emit('session_state', session)
    else:
        emit('error', {'message': 'Session not found'})


@socketio.on('transcript_update')
def on_transcript_update(data):
    """
    Broadcast live transcript text to all room participants.
    data: { pin, text, speaker }
    """
    pin = str(data.get('pin', '')).strip()
    emit('transcript_update', data, room=pin)


# ─── 5. Candidate Integrity Events ────────────────────────────────────────────

@socketio.on('integrity_event')
def on_integrity_event(data):
    """
    Candidate emits behavioural integrity signals → relay to interviewer.
    Also update the session record so the final report captures them.

    data: {
      pin          : str   – 6-digit session PIN
      event_type   : str   – 'tab_switch' | 'copy_attempt' | 'paste_attempt'
                             | 'long_pause' | 'screen_blur' | 'devtools_open'
                             | 'right_click' | 'session_start'
      payload      : dict  – event-specific details
      score        : int   – candidate's current integrity score (0-100)
    }
    """
    pin        = str(data.get('pin', '')).strip()
    event_type = data.get('event_type', 'unknown')
    payload    = data.get('payload', {})
    score      = data.get('score', 100)

    if not pin:
        return

    print(f"[WS] integrity_event PIN={pin} type={event_type} score={score}")

    # Persist running counters in session
    session = _session_get(pin) or {}
    counters = session.get('integrity_counters', {})
    key_map = {
        'tab_switch':    'tab_switches',
        'copy_attempt':  'copy_attempts',
        'paste_attempt': 'paste_events',
        'long_pause':    'long_pauses',
        'screen_blur':   'screen_blurs',
        'devtools_open': 'devtools_opens',
        'right_click':   'right_clicks',
    }
    field = key_map.get(event_type)
    if field:
        counters[field] = counters.get(field, 0) + 1
    counters['integrity_score'] = score
    _session_update(pin, {'integrity_counters': counters,
                           'last_integrity_event': event_type})

    # Relay to interviewer
    interviewer_sid = _room_members.get(pin, {}).get('interviewer')
    if interviewer_sid:
        socketio.emit('integrity_event', {
            'event_type': event_type,
            'payload':    payload,
            'score':      score,
            'pin':        pin
        }, room=interviewer_sid)

    # Echo acknowledgement back to candidate
    emit('integrity_ack', {'event_type': event_type, 'score': score})

if __name__ == '__main__':
    print("=" * 65)
    print("  NexHire AI — Interview Intelligence Platform")
    print(f"  LLM     : Groq {GROQ_MODEL}")
    print(f"  STT     : Groq {GROQ_WHISPER}")
    print(f"  RAG     : {'sentence-transformers + ChromaDB ACTIVE' if (_ST_AVAILABLE and _CHROMA_AVAILABLE) else 'pip install sentence-transformers chromadb'}")
    print(f"  DB      : {'MongoDB ACTIVE ✅  — nexhire.sessions' if _MONGO_OK else 'In-memory fallback ⚠️  (MongoDB not found)'}")
    print(f"  GitHub  : Active (public profiles, no auth required)")
    print(f"  Video   : LiveKit WebRTC (self-hosted — replaces Zoom/Meet)")
    print(f"  Visit   : http://localhost:5000")
    print("=" * 65)
    print("\nKey Novelties:")
    print("  01 ✅  AI Interview Copilot")
    print("  02 ✅  Simulation-Based Interviewing")
    print("  03 ✅  Integrity Monitoring")
    print("  04 ✅  GitHub Analysis System")
    print("  05 ✅  Gamified Simulation")
    print("  06 ✅  Interview Replay Intelligence")
    print("  07 ✅  LiveKit Self-Hosted Video  ← replaces Zoom/Meet")
    print()
    print("MongoDB:")
    print("  Install : pip install pymongo")
    print("  Run     : mongod  (or  docker run -p 27017:27017 mongo)")
    print("  URI     : set MONGO_URI env var  (default: mongodb://localhost:27017)")
    print()
    print("LiveKit server (run in separate terminal):")
    print("  docker run --rm -p 7880:7880 livekit/livekit-server --dev")
    print()
    socketio.run(app, debug=True, port=5000, host='0.0.0.0', allow_unsafe_werkzeug=True)