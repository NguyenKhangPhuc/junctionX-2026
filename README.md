# Kaleva OuluLens — Multilingual Agentic Newsroom

> **Junction X 2026 Hackathon — Kaleva Media Challenge**  
> Transforming literal news translation into a two-way **Explanatory Journalism** and **Audio/Video Dubbing** workflow governed by Finland's **JSN (Journalist's Guidelines)**.

---

## 1. The Problem & Target Community Justification

### Why Literal Translation Fails International Residents
When a Finnish newspaper runs a story about changes to *varhaiskasvatus*, *talvikunnossapito*, or *YTHS* queues, a 1:1 machine translation into English leaves international residents just as confused as before. They do not lack English words—they lack **institutional and cultural context** (how *Wilma* works, when *Kela* or *eVaka* deadlines apply, and how municipal decisions affect their family).

### Target Communities in Oulu (`TODO §1`)
1. **Primary Segment — International Tech Workers & Families (Option A):**
   * **Size & Urgency:** Oulu is home to over **15,000 international residents**, anchored by global ICT employers (Nokia's Oulu campus, Bittium, Polar, OuluHealth) and R&D hubs.
   * **The Retention Gap:** While Finland invests heavily in attracting international engineers, regional surveys show high family turnover within 24–36 months due to spouses and parents feeling disconnected from municipal life, schooling, and local decision-making.
2. **Secondary Segment — International Students & University Researchers (Option B):**
   * **Size & Lifecycle:** Over **2,000+ international degree students and researchers** at the University of Oulu and Oamk in Linnanmaa. They experience rapid onboarding needs every autumn (housing, winter cycling safety, student healthcare) and transition into long-term residents when integrated early.

---

## 2. Agentic Newsroom Architecture & Workflow (`TODO §2 & §3`)

```text
Kaleva Free RSS (maksuttomat) + Uploaded Video Clips
   │
   ├─► [1. Ingestion & Municipal Enrichment Agent] (app/services/ingestion_service.py)
   │      • Deduplicates Kaleva RSS stories
   │      • Enriches articles with live City of Oulu Open Data (api.ouka.fi/v1 PostgREST)
   │
   ├─► [2. Explanatory Journalism & Guardrail Agent] (app/services/translation_service.py)
   │      • Classifies into Kaleva Media's 6 User Needs:
   │        (Information, Understanding, Usefulness, Emotion, Break, Community)
   │      • Filters out hyper-local items irrelevant to international residents
   │      • Generates plain-English "Expat Explainer Boxes" (Wilma, Kela, Varhaiskasvatus, etc.)
   │      • Flags High-Stakes Topics (Healthcare, Tax, Permits, Housing) for mandatory senior review
   │      • Attaches JSN AI-Disclosure metadata
   │
   ├─► [3. Human-in-the-Loop (HITL) Editorial Gate] (app/api/newsroom.py)
   │      • Status lifecycle: draft ──► pending_review ──► processing_media ──► completed
   │      • Duty Editor inspects side-by-side Finnish source, English script, and explainer cards
   │      • Editor can APPROVE (with edits) or REJECT (returning notes to draft)
   │
   └─► [4. Edge-TTS & FFmpeg Broadcast Renderer] (app/services/video_pipeline_service.py)
          • Synthesizes natural neural speech per time-coded segment (Edge-TTS / Coqui XTTS-v2)
          • Fits audio to exact timeline slots via FFmpeg atempo & silence padding
          • Generates a 720p broadcast slate MP4 if the source article is text-only
```

---

## 3. Two-Way Community Co-Creation & Reverse-Bridge (`TODO §4`)

International residents are treated as **active newsroom co-creators** rather than passive consumers:

1. **Ask & Vote (`/api/community/questions`, `/api/community/topics/vote`):**
   * Residents submit everyday navigation questions (e.g., *"How do I apply for an English daycare in Linnanmaa mid-semester?"*) and upvote community priorities.
2. **Automated RAG Briefing at $\ge 3$ Upvotes (`/api/community/questions/{id}/upvote`):**
   * Crossing the upvote threshold triggers `CommunityService`, which performs cross-lingual semantic vector search (`intfloat/multilingual-e5-large` + `pgvector` HNSW cosine similarity) over Kaleva's Finnish archives plus `api.ouka.fi`, drafting a 60-second audio/video script straight into the editor's `pending_review` queue.
3. **Reverse-Bridge Pitch to Kaleva's Finnish Newsdesk (`/api/community/questions/{id}/pitch-to-kaleva`):**
   * Translates high-interest international community issues **back into Finnish investigative story pitches** (`otsikko_fi`, `juttuidea_fi`, `haastateltavat_fi`) so Kaleva's mainstream Finnish reporters can cover systemic municipal bottlenecks.
4. **Community Error Flagging (`/api/community/videos/{dub_id}/flag-error`):**
   * In accordance with JSN self-regulation, any published asset flagged by a community member immediately transitions to `flagged_for_review` and re-enters the human editor queue.

---

## 4. Business Model, Editorial Firewall & Unit Economics (`TODO §5`)

### Revenue Streams
* **B2B "Talent Retention" Enterprise Licenses:** Bulk seat subscriptions sold to Oulu tech employers (Nokia, Bittium, Polar) and the University of Oulu at **€5–€8 / international employee / month** as part of their relocation and family-retention package.
* **B2G Municipal Integration Co-Funding:** Service partnership with **International House Oulu / City of Oulu** to sponsor the public Community Q&A video/audio briefings, deflecting repetitive support queries at municipal service desks.
* **Consumer Freemium:** Free weekly municipal safety/utility briefings; Kaleva+ bundle for full daily explanatory video digests.

### JSN Editorial Firewall
* B2B corporate sponsors purchase **distribution access tokens** for their staff; they have **zero visibility or influence** over story selection, explanatory framing, or the HITL queue.
* Every published item carries an explicit `ai_disclosure` statement and logs the approving Kaleva duty editor (`approved_by`).

### Unit Economics
* **Compute Cost per Asset:**
  * LLM Contextualization & Translation (OpenRouter Gemini 2.5 Flash / Llama 3.3 70B or local Ollama): **~$0.002 / article**
  * Embeddings (`fastembed` local CPU) + Speech Synthesis (`Edge-TTS`) + Video Assembly (`FFmpeg`): **$0.00 marginal API cost**
  * **Total AI Compute Cost:** **$< \$0.01$ per published video/audio briefing**
* **Human Editor Overhead:** Reduced from ~45 minutes (manual translation, institutional explanation, voiceover recording) to **~90 seconds** of verification in the HITL review queue.
* **Breakeven Threshold:** **2 enterprise B2B agreements** (~500 seats at €6/mo = **€3,000 MRR**) fully covers cloud infrastructure and 15 hours/month of a Kaleva sub-editor's review time.

---

## 5. Quickstart & Demo Commands

### 1. Environment & Database Setup
```bash
cp .env.example .env
# Ensure DATABASE_URL matches the docker-compose host port (5433):
# DATABASE_URL=postgresql://postgres:postgres@localhost:5433/vectordb
# OPENROUTER_API_KEY=your_key_here

docker compose up -d
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --port 5605
```

### 2. Live Pitch Demo Sequence (cURL)
```bash
# 1. Seed 3 realistic Oulu expat community questions (Wilma, Baana winter cycling, Linnanmaa daycare)
curl -X POST http://localhost:5605/api/community/seed-demo

# 2. Ingest live Finnish news from Kaleva's free RSS feed + enrich with Oulu Open Data
curl -X POST "http://localhost:5605/api/newsroom/ingest-kaleva?limit=3"

# 3. Inspect the Human-in-the-Loop (HITL) Editor Queue (includes high_stakes_warning & expat_explainer_box)
curl http://localhost:5605/api/newsroom/review-queue

# 4. Upvote Question #3 (Wilma) from 2 -> 3 upvotes to trigger live RAG + Oulu API auto-drafting
curl -X POST http://localhost:5605/api/community/questions/3/upvote

# 5. Generate a Reverse-Bridge Finnish investigative pitch for Kaleva's newsdesk
curl -X POST http://localhost:5605/api/community/questions/1/pitch-to-kaleva

# 6. Approve Draft #1 in the HITL gate to trigger background Edge-TTS + FFmpeg video rendering
curl -X POST http://localhost:5605/api/newsroom/review/1/approve \
  -H "Content-Type: application/json" \
  -d '{"edited_transcript_text": "Welcome to Oulu! Here is your verified Kaleva briefing on Linnanmaa daycare registration.", "editor_name": "Kaleva Senior Editor"}'
```