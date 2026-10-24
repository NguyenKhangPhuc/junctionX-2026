# Project TODO: Multilingual Agentic Newsroom (Kaleva Media)

> Status verified against the codebase (`app/`, `tests/`, HEAD `e3e4395`).
> `[x]` = implemented in code. `**partial**` = code exists but scope/gap remains.


## 1. Target Community & Scope
- [x] Select primary target community/communities:
  - [x] Option A: Work-based immigrants & international professionals (e.g., Oulu tech hub) — implemented: `COMMUNITY_SYSTEM_PROMPT` targets "international tech workers and families in Oulu"; personas `"Oulu Tech Worker & Family"`, `"Nokia Engineer & Parent"` (`app/services/community_service.py`, `app/api/community.py`)
  - [x] Option B: International students and university researchers — persona `"University of Oulu Researcher"` (`app/api/community.py:seed_demo_questions`)
  - [ ] Option C: Seasonal tourism workers in Lapland (winter season, rapid onboarding)
  - [ ] Option D: Permanent residents with refugee/humanitarian backgrounds (Arabic, Ukrainian, Russian, Somali, etc.)
  - [ ] Option E: Domestic language minorities (Sámi languages: North, Inari, Skolt; Swedish speakers)
- [ ] Document justification for community selection (information gap, urgency, addressable size). — **partial**: selection lives only in code prompts; `README.md` is empty, no written rationale
- [ ] Profile community information lifecycles (e.g., short-term seasonal turnover vs. long-term integration).

---

## 2. Agentic Newsroom Architecture & Newsroom Workflow
- [x] Integrate Kaleva Media's RSS feed (`RSS | Kaleva`) as an input source. — `KALEVA_FREE_RSS` fetched + deduped in `app/services/ingestion_service.py`, exposed via `POST /api/newsroom/ingest-kaleva`
- [ ] Build source monitoring agents:
  - [ ] City council decisions and municipal agendas — **partial**: keyword-driven Oulu Open Data enrichment (`fetch_ouka_context`, `api.ouka.fi/v1` properties / exercise open hours); no council agenda or seat/decision ingestion
  - [ ] Official public authority announcements (Kela, Migri, TE-toimisto, police, healthcare)
  - [x] Local news and events feeds — Kaleva free RSS (`maksuttomat`) ingested; note: only that single feed is wired
- [ ] Implement relevance filter based on the target community's daily needs. — **partial**: `relevance_tag` taxonomy (Family & Daycare, Housing & Energy, Transport & Winter, Healthcare & Kela, Work & Tax, Local Culture) classifies items, but nothing drops irrelevant ones
- [x] Implement explanatory journalism engine:
  - [x] Add cultural context injection (bridge implied Finnish institutional knowledge). — `OULU_EXPLANATORY_SYSTEM_PROMPT` names Varhaiskasvatus / Wilma / Talvikunnossapito; `expat_explainer_box` emitted
  - [x] Focus on "What this means for you" rather than verbatim translation. — `expat_explainer_box[].what_it_means_for_you`; JSN-style fact/context separation in the prompt
- [ ] Align agent pipeline with Kaleva Media's 6 User Needs. — **partial**: 4 of 6 covered
  - [x] *Information* (What happened) — `headline_en` + spoken narration segments
  - [x] *Understanding* (Why, context, impact) — `expat_explainer_box`
  - [x] *Usefulness* (Everyday utility: permits, transit, housing, rights) — community Q&A drafts default to `kaleva_user_need: "Usefulness"`
  - [ ] *Emotion* (Human interest, inspiration)
  - [ ] *Break* (Leisure, entertainment)
  - [x] *Community* (Connection, platforming local voices) — community questions/upvotes feed the newsroom agenda
- [ ] Design reverse-bridge agent (translating community issues/stories into Finnish for mainstream coverage).

---

## 3. Human-in-the-Loop (HITL) & Journalistic Integrity
- [ ] Map newsroom roles and agent hierarchy (Editor-in-Chief -> Section Editor -> Agent -> Human QA). — **partial**: stage order documented in `project_plan.md` Phase 2 and personas implied in prompts; no written role/agent map
- [x] Implement review checkpoints ensuring adherence to JSN Guidelines (Journalist’s Guidelines of Finland). — JSN rules in both LLM prompts; `pending_review` -> `processing_media` -> `completed` gate (`app/api/newsroom.py`, `app/services/video_pipeline_service.py:approve_and_publish`). Note: `PATCH /api/editor/{id}/reject` from the plan is still missing — community `flag-error` is the only pullback path
- [ ] Build safety guards for high-stakes topics (legal status, healthcare, labor rights, housing contracts). — **partial**: `journalist_verification_note` asks the human editor to verify; no topic-based guardrail or blocking rule
- [ ] Create specialized QA workflows for under-resourced languages (e.g., human-verified Sámi translations).
- [ ] Establish transparency and AI disclosure statements on all published assets.

---

## 4. Community Integration & Distribution
- [ ] Map existing community communication hubs (WhatsApp channels, Telegram, Facebook groups, local hubs).
- [x] Select distribution channel tailored to community habits (chat apps, lightweight web, audio/podcast, short video). — short dubbed video/audio chosen and rendered (Edge-TTS + FFmpeg timeline in `app/services/tts_service.py`, `ffmpeg_service.py`); chat/web delivery not wired
- [x] Design bidirectional feedback loops:
  - [x] Story tip submission — `POST /api/community/questions` (`author="COMMUNITY_QUESTION"`, `persona_tag`)
  - [x] Q&A / information request interface — `POST/GET /api/community/questions`, auto-draft at >=3 upvotes
  - [x] Fact-checking and error-flagging mechanism by community members — `POST /api/community/videos/{dub_id}/flag-error` sets status `flagged_for_review`
- [ ] Identify trusted community ambassadors/liaisons to validate content tone and accuracy.

---

## 5. Business Model & Sustainability
- [ ] Define revenue streams:
  - [ ] B2B employer/institution licenses (tech employers, tourism operators, universities)
  - [ ] B2G/Municipal integration support or regional development grants
  - [ ] Foundation funding (particularly for indigenous/minority languages like Sámi)
  - [ ] Targeted recruitment and local service ads
  - [ ] Consumer subscriptions / freemium tiers
- [ ] Formulate firewall strategy to prevent conflicts of interest between sponsors/employers and reporting.
- [ ] Estimate unit economics:
  - [ ] Agent operational compute costs per language/community
  - [ ] Human editor overhead costs
  - [ ] Breakeven subscriber/client threshold

---

## 6. Prototyping, Demo & Evaluation
- [x] Set up pipeline prototype ingesting sample RSS/official announcements. — Kaleva RSS + Oulu Open Data enrichment; full ingest -> draft -> review -> render chain works
- [ ] Implement sample agent pipeline with human approval dashboard. — **partial**: queue API only (`GET /api/newsroom/review-queue`, `POST /api/newsroom/review/{dub_id}/approve`); no editor UI yet
- [ ] Generate sample localized outputs demonstrating the User Needs model. — **partial**: `POST /api/community/seed-demo` ships 3 EN explainer drafts covering 2 of 6 needs; English-only, no rendered demo asset committed
- [ ] Prepare pitch deck and demo addressing:
  - [ ] Community centrality
  - [ ] HITL and ethics compliance
  - [ ] Viability and error resilience
  - [ ] Business sustainability and synergy with Kaleva Media
