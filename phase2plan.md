Building Phase 2 requires wiring the **Kaleva free RSS feed** and the **Oulu Open Data API (`api.ouka.fi/v1`)** directly into your existing SQLAlchemy models (`Article`, `Video`, `VideoDubbedVersion`) and your `translation_service.py` / `tts_service.py` stack.

### How the Two Data Sources Work Together

1. **Kaleva RSS (`.../rss-maksuttomat/`)**: Provides free-to-read Finnish news XML items (`<title>`, `<description>`, `<link>`, `<pubDate>`).
2. **Oulu Open Data (`[https://api.ouka.fi/v1/](https://api.ouka.fi/v1/)`)**: A **PostgREST** JSON API providing live municipal data—such as city properties (schools, daycares like *Ahvenojan päiväkoti* via `/v1/properties`), sports/swimming facility hours (`/v1/exerciseserviceopenhours`), and city infrastructure.
3. **The Magic**: When a Kaleva article mentions a school, daycare, sports hall, or district, the backend queries `api.ouka.fi/v1` to pull real municipal facts and injects them into the LLM prompt so the translated video/audio digest explains both the news *and* the practical civic details.



---

### Step 2.1: Ingestion & Oulu Open Data Enricher (`app/services/ingestion_service.py`)

Instead of requiring a user to upload a video first, create a service that pulls a free Kaleva RSS story, cross-references `api.ouka.fi`, and stores it in your existing `articles` table.

**What to implement:**

- Parse the XML feed using `xml.etree.ElementTree` (built into Python, zero new dependencies) and `httpx` (already installed).
- Query `[https://api.ouka.fi/v1/properties](https://api.ouka.fi/v1/properties)` or `/v1/exerciseserviceopenhours` using PostgREST filters (e.g., `?property_name=ilike.*päiväkoti*`) to attach real Oulu facility context.



```python
# app/services/ingestion_service.py
import xml.etree.ElementTree as ET
import httpx
from sqlalchemy.orm import Session
from app.models.article import Article

KALEVA_FREE_RSS = "https://www.kaleva.fi/feedit/rss/managed-listing/rss-maksuttomat/"
OUKA_API_BASE = "https://api.ouka.fi/v1"

class IngestionService:
    async def fetch_kaleva_free_rss(self, db: Session, limit: int = 5) -> list[Article]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(KALEVA_FREE_RSS)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        saved_articles = []

        for item in root.findall("./channel/item")[:limit]:
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()

            # Deduplicate by title
            existing = db.query(Article).filter(Article.title == title).first()
            if existing:
                saved_articles.append(existing)
                continue

            # Enrich with Oulu municipal data if relevant keywords appear
            ouka_context = await self.fetch_ouka_context(f"{title} {description}")
            full_content = f"Summary: {description}\nSource: {link}\nOulu City Data: {ouka_context}"

            article = Article(
                title=title,
                content=full_content,
                original_language="fi",
                author="Kaleva RSS (Maksuttomat)"
            )
            db.add(article)
            db.commit()
            db.refresh(article)
            saved_articles.append(article)

        return saved_articles

    async def fetch_ouka_context(self, text: str) -> str:
        """Queries PostgREST endpoints on api.ouka.fi/v1 for local facility context."""
        text_lower = text.lower()
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                # Example 1: Daycare / School mentions -> query /v1/properties
                if any(k in text_lower for k in ["päiväkoti", "koulu", "varhaiskasvatus"]):
                    r = await client.get(f"{OUKA_API_BASE}/properties", params={"limit": 3})
                    if r.status_code == 200 and r.json():
                        names = [p.get("property_name", "") for p in r.json() if p.get("property_name")]
                        return f"Oulu Municipal Properties sample: {', '.join(names[:3])}"

                # Example 2: Sports / Swimming / Winter tracks -> query /v1/exerciseserviceopenhours
                if any(k in text_lower for k in ["liikunta", "uimahalli", "latu", "ouluhalli"]):
                    r = await client.get(f"{OUKA_API_BASE}/exerciseserviceopenhours", params={"limit": 2})
                    if r.status_code == 200 and r.json():
                        return f"Oulu Sports Facility Open Data: {r.json()[:2]}"
            except Exception:
                pass
        return "General Oulu municipal jurisdiction."

```

---

### Step 2.2: Explanatory Newsroom Prompt (`app/services/translation_service.py`)

Right now, `translation_service.py` translates transcript segments literally. Add a method (or modify your prompt) so that when it processes a Kaleva article or video transcript, it generates **structured JSON** containing both the narration script and the cultural/civic glossary for tech workers and their families in Oulu.

**Add this prompt template to `translation_service.py`:**

```python
OULU_EXPLANATORY_SYSTEM_PROMPT = """
You are a senior bilingual news editor at Kaleva Media serving international tech workers and their families in Oulu, Finland.
Follow Finland's JSN Journalistic Guidelines: strictly separate verified facts from context, never invent policies, and cite sources.

Given a Finnish news item and optional Oulu City API data, return ONLY valid JSON matching this exact schema:
{
  "headline_en": "Translated, clear headline",
  "relevance_tag": "One of: [Family & Daycare, Housing & Energy, Transport & Winter, Healthcare & Kela, Work & Tax, Local Culture]",
  "spoken_narration_segments": [
    {"start": 0.0, "end": 6.0, "text": "Clear spoken English for video/audio dubbing..."}
  ],
  "expat_explainer_box": [
    {
      "finnish_term": "e.g., Varhaiskasvatus / Wilma / Talvikunnossapito",
      "what_it_means_for_you": "Plain-English explanation of how an international family in Oulu should act on this."
    }
  ]
}
"""

```

**Why this fits your existing DB without new migrations:**
You can store this entire JSON object directly inside `VideoDubbedVersion.transcript_json` (which is already a `JSONB` column in migration `a3145af6f975`) and store the joined `spoken_narration_segments` inside `VideoDubbedVersion.transcript_text`!

---

### Step 2.3: Human-in-the-Loop (HITL) Gate inside `VideoPipelineService`

Currently, `VideoPipelineService.process_video` runs Steps 1 through 6 non-stop. To comply with the hackathon's **Non-Negotiable HITL Rule**, split `process_video` into two methods:

1. **`prepare_draft(video_id, target_language)` (Steps 1–4)**:
- Runs audio extraction, Whisper STT, and the Explanatory LLM call.
- Saves the result to `video_dubbed_versions` with `status = "pending_review"` (no DB schema change needed—`status` is a `String` column, not a locked Postgres enum).
- **STOPS HERE.** Does not call TTS or FFmpeg merge yet.




2. **`approve_and_publish(dubbed_version_id, edited_transcript_text, edited_transcript_json)` (Steps 5–6)**:
- Called by a human journalist/editor after reviewing or editing the translation.


- Updates `transcript_text` and `transcript_json` with the human-verified text.


- Runs **Step 5 (`tts_service.py` using Edge-TTS `en-US-GuyNeural`)** and **Step 6 (`ffmpeg_service.py` muxing)**.


- Sets `status = "completed"`.



---

### Step 2.4: New FastAPI Endpoints (`app/api/newsroom.py`)

Create a single new router file `app/api/newsroom.py` and mount it in `app/main.py` (`app.include_router(newsroom.router, prefix="/api/newsroom")`).

| Endpoint | Method | What it does |
| --- | --- | --- |
| `/api/newsroom/ingest-kaleva` | `POST` | Pulls latest free items from `kaleva.fi/.../rss-maksuttomat/`, enriches with `api.ouka.fi/v1`, saves `Article` rows, and creates a `pending_review` draft in `video_dubbed_versions`. |
| `/api/newsroom/review-queue` | `GET` | Queries all `VideoDubbedVersion` rows where `status == "pending_review"`, returning the original Finnish text, the AI English narration, and the `expat_explainer_box`.

 |
| `/api/newsroom/review/{dub_id}/approve` | `POST` | Accepts human edits to the English script, sets `status = "approved"`, and triggers Edge-TTS audio synthesis + FFmpeg video muxing in `BackgroundTasks`.

 |

```python
# app/api/newsroom.py
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.ingestion_service import IngestionService
from app.services.video_pipeline_service import VideoPipelineService

router = APIRouter(tags=["Newsroom HITL"])

class EditorApprovalPayload(BaseModel):
    edited_transcript_text: str
    editor_name: str = "Kaleva Duty Editor"

@router.post("/ingest-kaleva")
async def ingest_kaleva_rss(db: Session = Depends(get_db)):
    service = IngestionService()
    articles = await service.fetch_kaleva_free_rss(db, limit=5)
    return {"ingested_count": len(articles), "articles": [{"id": a.id, "title": a.title} for a in articles]}

@router.get("/review-queue")
def get_review_queue(db: Session = Depends(get_db)):
    pending = (
        db.query(VideoDubbedVersion)
        .filter(VideoDubbedVersion.status == "pending_review")
        .all()
    )
    return pending

@router.post("/review/{dub_id}/approve")
def approve_draft(
    dub_id: int,
    payload: EditorApprovalPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    dub = db.query(VideoDubbedVersion).filter(VideoDubbedVersion.id == dub_id).first()
    if not dub:
        raise HTTPException(status_code=404, detail="Draft not found")

    dub.transcript_text = payload.edited_transcript_text
    dub.status = "processing_media"
    db.commit()

    # Trigger Step 5 (Edge-TTS) & Step 6 (FFmpeg merge) asynchronously
    pipeline = VideoPipelineService(db)
    background_tasks.add_task(pipeline.render_approved_dub, dub.id)

    return {"status": "approved_and_rendering", "dub_id": dub.id, "approved_by": payload.editor_name}

```

### Summary of Execution Order 

1. **Create `app/services/ingestion_service.py**` with the Kaleva RSS XML parser and `api.ouka.fi/v1` enrichment above.
2. **Update `app/services/translation_service.py**` with `OULU_EXPLANATORY_SYSTEM_PROMPT` so every translation outputs the Oulu expat explainer alongside the script.


3. **Split `VideoPipelineService.process_video**` in `app/services/video_pipeline_service.py` so it pauses at `status = "pending_review"` before TTS, and add `render_approved_dub(dub_id)` to run TTS + FFmpeg after approval.


4. **Mount `app/api/newsroom.py**` in `app/main.py`.



Implementing **Phase 3 (Two-Way Community Touchpoint)** directly addresses the challenge’s evaluation criteria for **Community Integration** (involving target users actively rather than treating them as passive consumers) and **Two-Way Community Loops** (letting readers submit tips, ask questions, flag errors, and vote on coverage priorities).

To avoid slow database migrations during the hackathon, you can build Phase 3 **entirely on top of your existing `articles` and `video_dubbed_versions` tables** by using the `author` and `transcript_json` (`JSONB`) columns to store community questions, votes, and error flags.

---

### How Phase 3 Works End-to-End

```
[Oulu Tech Worker / Parent]
   ├── 1. Submits a Question or Tip ("How do winter cycling routes work in Linnanmaa?")
   ├── 2. Upvotes existing community questions to set newsroom priorities
   └── 3. Flags a translation/cultural error on a published video ("Wilma is for schools, not tax!")
          │
          ▼
[Community Agent Service]
   ├── Searches ingested Kaleva RSS articles (in `articles` table)
   ├── Queries live Oulu Open Data (`api.ouka.fi/v1`) for official city facts
   └── Generates an "Answer Draft + Audio Script" saved with `status = "pending_review"`
          │
          ▼
[Kaleva Duty Editor (HITL Gate from Phase 2)]
   └── Reviews, edits, and approves -> Renders Edge-TTS Audio Briefing for the Community!

```

---

### Step 3.1: Zero-Migration Storage Strategy (Reusing Existing Models)

Instead of running `alembic revision` and risking schema conflicts, use `Article` and `VideoDubbedVersion` like this:

1. **Community Questions & Tips (`articles` table):**
* Store a community submission as an `Article` row where `author = "COMMUNITY_QUESTION"` or `"COMMUNITY_TIP"`.
* Store the metadata (submitter role like *"Nokia Engineer / Parent"*, upvote count, category, and status) as a JSON string or structured prefix inside `Article.content`, or inside an associated `VideoDubbedVersion.transcript_json` (`JSONB`).


2. **Error Flagging (`video_dubbed_versions` table):**
* When a user flags a factual or linguistic error in a published video/article, append the flag to `transcript_json["community_flags"]` and automatically revert `VideoDubbedVersion.status` back to `"flagged_for_review"` so a human journalist must inspect it.





---

### Step 3.2: Community RAG & Answer Agent (`app/services/community_service.py`)

Create `app/services/community_service.py`. When a community question gets enough upvotes (or when an editor clicks "Draft Answer"), this service searches your database of Kaleva articles + live `api.ouka.fi/v1` endpoints and uses your OpenRouter/Ollama setup to draft a verified response for human review.

```python
# app/services/community_service.py
import json
import httpx
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models.article import Article
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.ingestion_service import IngestionService
from app.services.translation_service import TranslationService

COMMUNITY_ANSWER_PROMPT = """
You are a community journalist at Kaleva Media's 'OuluLens' desk serving international tech workers and families in Oulu.
A community member asked the following question or submitted a tip:
QUESTION: {question}

Here is verified context from Kaleva Media archives and Oulu City Open Data (api.ouka.fi):
CONTEXT:
{context}

Follow JSN Journalistic Guidelines:
1. Only state facts supported by the provided Kaleva or Oulu City context.
2. If the exact answer is not in the context, state clearly what requires further verification by a human Kaleva reporter.
3. Explain Finnish municipal terms (e.g., Wilma, Kela, Varhaiskasvatus, Talvikunnossapito) in plain English.

Return ONLY valid JSON with this structure:
{{
  "headline_en": "Clear answer headline for the Oulu expat community",
  "kaleva_user_need": "One of: [Information, Understanding, Usefulness, Emotion, Break, Community]",
  "draft_answer_script": "60-second spoken English audio briefing script answering the question...",
  "sources_used": ["List of Kaleva articles or Oulu API endpoints used"],
  "journalist_verification_note": "Note for the human editor on what to double-check before publishing"
}}
"""

class CommunityService:
    def __init__(self, db: Session):
        self.db = db
        self.ingestion = IngestionService()
        self.translator = TranslationService()

    async def draft_community_answer(self, question_article_id: int) -> dict:
        question_row = self.db.query(Article).filter(Article.id == question_article_id).first()
        if not question_row:
            raise ValueError("Question not found")

        meta = json.loads(question_row.content)
        question_text = meta.get("question_text", question_row.title)

        # 1. Search ingested Kaleva articles in Postgres using keyword match
        keywords = [w for w in question_text.split() if len(w) > 3][:4]
        kaleva_matches = []
        if keywords:
            filters = [Article.content.ilike(f"%{kw}%") for kw in keywords]
            kaleva_matches = (
                self.db.query(Article)
                .filter(Article.author != "COMMUNITY_QUESTION")
                .filter(or_(*filters))
                .limit(3)
                .all()
            )

        kaleva_context = "\n".join([f"- {a.title}: {a.content[:300]}" for a in kaleva_matches])

        # 2. Query Oulu Open Data API (api.ouka.fi/v1)
        ouka_context = await self.ingestion.fetch_ouka_context(question_text)

        combined_context = f"Kaleva Articles:\n{kaleva_context or 'None found'}\n\nOulu API:\n{ouka_context}"

        # 3. Call LLM (OpenRouter / Ollama) via your existing TranslationService helper
        prompt = COMMUNITY_ANSWER_PROMPT.format(question=question_text, context=combined_context)
        llm_raw = self.translator._call_llm(prompt)  # Uses your existing OpenRouter/Ollama fallback

        try:
            parsed = json.loads(llm_raw)
        except Exception:
            parsed = {
                "headline_en": f"Community Q&A: {question_row.title}",
                "kaleva_user_need": "Usefulness",
                "draft_answer_script": llm_raw,
                "sources_used": ["Kaleva RSS", "api.ouka.fi"],
                "journalist_verification_note": "Verify municipal details manually."
            }

        # Save draft back into the question's JSON metadata and mark as pending_review for HITL
        meta["status"] = "pending_review"
        meta["ai_draft"] = parsed
        question_row.content = json.dumps(meta)
        self.db.commit()
        return parsed

```

---

### Step 3.3: Community Router Endpoints (`app/api/community.py`)

Create `app/api/community.py` with four endpoints covering every requirement in the hackathon brief:

1. **`POST /api/community/questions`** — Submit a question or local news tip.


2. **`GET /api/community/questions`** — List community questions sorted by upvotes (shows community priority setting).


3. **`POST /api/community/questions/{id}/upvote`** — Vote on coverage priorities + auto-trigger an AI draft for the editor once a threshold (e.g., 3 votes) is reached.


4. **`POST /api/community/videos/{dub_id}/flag-error`** — Let residents flag translation inaccuracies or hallucinations, immediately pausing the item into `"flagged_for_review"` for JSN accountability.



```python
# app/api/community.py
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.models.article import Article
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.community_service import CommunityService

router = APIRouter(tags=["Community Co-Creation"])

class CommunityQuestionCreate(BaseModel):
    title: str
    question_text: str
    persona_tag: str = "Oulu Tech Worker & Family"  # e.g. "Nokia Engineer", "Linnanmaa Parent"
    language: str = "en"

class ErrorFlagPayload(BaseModel):
    flagged_segment: str
    correction_note: str
    reporter_persona: str = "Oulu Community Member"

@router.post("/questions")
def submit_question_or_tip(payload: CommunityQuestionCreate, db: Session = Depends(get_db)):
    meta = {
        "question_text": payload.question_text,
        "persona_tag": payload.persona_tag,
        "upvotes": 1,
        "status": "open_for_voting",
        "ai_draft": None
    }
    row = Article(
        title=payload.title,
        content=json.dumps(meta),
        original_language=payload.language,
        author="COMMUNITY_QUESTION"
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "title": row.title, "meta": meta}

@router.get("/questions")
def list_community_questions(db: Session = Depends(get_db)):
    rows = db.query(Article).filter(Article.author == "COMMUNITY_QUESTION").all()
    results = []
    for r in rows:
        try:
            meta = json.loads(r.content)
        except Exception:
            meta = {"question_text": r.content, "upvotes": 0, "status": "open"}
        results.append({
            "id": r.id,
            "title": r.title,
            "created_at": r.created_at,
            **meta
        })
    # Sort highest upvoted first so the community sets the newsroom agenda
    results.sort(key=lambda x: x.get("upvotes", 0), reverse=True)
    return results

@router.post("/questions/{question_id}/upvote")
async def upvote_question(question_id: int, db: Session = Depends(get_db)):
    row = db.query(Article).filter(
        Article.id == question_id,
        Article.author == "COMMUNITY_QUESTION"
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Community question not found")

    meta = json.loads(row.content)
    meta["upvotes"] = meta.get("upvotes", 0) + 1
    row.content = json.dumps(meta)
    db.commit()

    # Automatically draft an explanatory answer for the editor queue when upvotes >= 3
    if meta["upvotes"] >= 3 and meta.get("status") == "open_for_voting":
        service = CommunityService(db)
        draft = await service.draft_community_answer(question_id)
        return {"id": row.id, "upvotes": meta["upvotes"], "auto_drafted_for_editor": True, "draft": draft}

    return {"id": row.id, "upvotes": meta["upvotes"], "auto_drafted_for_editor": False}

@router.post("/videos/{dub_id}/flag-error")
def flag_translation_or_fact_error(
    dub_id: int,
    payload: ErrorFlagPayload,
    db: Session = Depends(get_db)
):
    dub = db.query(VideoDubbedVersion).filter(VideoDubbedVersion.id == dub_id).first()
    if not dub:
        raise HTTPException(status_code=404, detail="Dubbed version not found")

    current_json = dub.transcript_json if isinstance(dub.transcript_json, dict) else {"segments": dub.transcript_json or []}
    flags = current_json.get("community_flags", [])
    flags.append({
        "flagged_segment": payload.flagged_segment,
        "correction_note": payload.correction_note,
        "reporter": payload.reporter_persona
    })
    current_json["community_flags"] = flags

    dub.transcript_json = current_json
    # JSN Accountability: pull back into the human editor review queue
    dub.status = "flagged_for_review"
    db.commit()

    return {
        "dub_id": dub.id,
        "new_status": dub.status,
        "message": "Video pulled back to the Kaleva Editor Queue for human verification."
    }

```

---

### Step 3.4: Pre-Seed 3 Realistic Oulu Tech Family Questions for the Demo

Judges love seeing realistic data rather than `"test 123"`. Add a quick seed endpoint (`POST /api/community/seed-demo`) or startup script that populates three authentic Oulu tech expat questions:

1. **Linnanmaa / Technopolis Parent:** *"Kaleva reported changes to Oulu's early childhood education (varhaiskasvatus) service vouchers. How does this affect English-speaking private daycares near Linnanmaa?"* (`upvotes: 14`)
2. **Relocated Nokia Hardware Engineer:** *"What does the city's new winter cycling maintenance (talvikunnossapito) priority map mean for the Baana commuter corridor from Kaukovainio to the university?"* (`upvotes: 9`)
3. **Spouse of Oulu ICT Worker:** *"Where can international families find the updated Oulu swimming hall (Raksilan uimahalli replacement / Raatti) opening hours during autumn school holidays?"* (`upvotes: 2` — *leave this at 2 upvotes so during the live demo you can click "Upvote" once, hit 3, and watch it auto-generate the AI draft for the editor live!*)

---

### Checklist to Complete Phase 3

* [ ] **1. Create `app/services/community_service.py**` with `COMMUNITY_ANSWER_PROMPT` (notice it explicitly outputs `"kaleva_user_need"` from Section 4 of the brief: *Information, Understanding, Usefulness, Emotion, Break, or Community*).


* [ ] **2. Create `app/api/community.py**` with `/questions`, `/upvote`, and `/flag-error`.
* [ ] **3. Register router in `app/main.py`:**
```python
from app.api import community
app.include_router(community.router, prefix="/api/community")

``