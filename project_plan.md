# Kaleva OuluLens: Project Implementation Plan

---

## Phase 1: Clean Up & Technical Stabilization (Immediate)

done
---

## Phase 2: "Explanatory Journalism" Agent Workflow

Transform the raw literal translation pipeline into a structured newsroom workflow adhering to JSN guidelines.

```
Kaleva / Oulu News (RSS/Article) 
    ──> [News Ingestion Agent] 
    ──> [Context & Cultural Explainer Agent] 
    ──> [Human-in-the-Loop Review Gate (HITL)] 
    ──> [Edge-TTS + Video/Audio Generator]
```

- [ ] **Create Oulu News Ingestion Service (`app/services/rss_service.py`)**
  - Ingest mock Kaleva Media articles or RSS feeds relating to municipal Oulu decisions (e.g., school enrollment, heating subsidies, local traffic/cycling, health services).
- [ ] **Implement Contextualization Prompt (`app/services/context_service.py`)**
  - Replace the 1:1 literal translation system prompt with an Explanatory Journalism prompt:
    - Detect implicit Finnish terminology: *Wilma, Kela, TE-palvelut, verokortti, YTHS, talvikunnossapito* (winter road maintenance).
    - Add a concise sidebar or spoken segment: *"What this means for international residents and their families in Oulu"*.
    - Ensure tone complies with JSN factual fairness guidelines.
- [ ] **Implement the Editorial Review Gate (HITL)**
  - Update `video_dubbed_versions.status` enum in the database to support editorial states: `draft`, `pending_review`, `approved`, `published`.
  - Create review API endpoints:
    - `GET /api/editor/queue`: List articles and translated scripts waiting for approval.
    - `PATCH /api/editor/{id}/approve`: Mark script as approved and trigger Edge-TTS audio/video rendering.
    - `PATCH /api/editor/{id}/reject`: Send back with editorial notes.

---

## Phase 3: Two-Way Community Touchpoint

Empower international families to participate directly rather than acting as passive consumers.

- [ ] **Create Community Interaction Endpoints (`app/api/community.py`)**
  - `POST /api/community/questions`: Allow international workers/parents to submit everyday navigation questions (e.g., *"How do I register my child for an English-language daycare in Linnanmaa?"*, *"What are the stud tire rules in Northern Ostrobothnia?"*).
  - `GET /api/community/topics/vote`: Allow community members to upvote issues they want Kaleva journalists to cover next.
- [ ] **Implement the Community FAQ Generator**
  - Create a quick workflow where upvoted questions retrieve context from ingested municipal documents/news using simple semantic search or keyword matching, drafting an audio/video answer for editor approval.

---

## Phase 4: Demo UI & Hackathon Pitch Polish

- [ ] **Minimal Web Dashboard** (Streamlit, React, or simple FastAPI Jinja HTML templates)
  - **Tab 1: Editor Desk (HITL)** — View incoming Finnish Kaleva stories, read the AI-suggested English explanatory breakdown, edit the text, and hit *"Approve & Broadcast"*.
  - **Tab 2: Community Feed** — Mobile-friendly preview showing the generated bilingual/English video digest with captions and the Oulu survival explainer box.
  - **Tab 3: Community Desk** — Interface for submitting and upvoting local questions.
- [ ] **Prepare Pitch Deck / Narrative**
  - **Problem:** Oulu has hundreds of foreign software engineers and researchers whose families struggle with local bureaucracy and school communications because news is locked behind Finnish language barriers or unverified social media groups.
  - **Solution:** Kaleva OuluLens uses agents to monitor and contextualize regional news, keeps journalists firmly in control with a JSN-compliant approval gate, and distributes short, accessible audio/video digests.
  - **Business Model:** B2B integration packages sold to regional tech employers (Nokia, Oulu ICT cluster) and regional relocation services as an employee retention and integration perk.