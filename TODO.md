# Project TODO: Multilingual Agentic Newsroom (Kaleva Media)

## 1. Target Community & Scope
- [ ] Select primary target community/communities:
  - [ ] Option A: Work-based immigrants & international professionals (e.g., Oulu tech hub)
  - [ ] Option B: International students and university researchers
  - [ ] Option C: Seasonal tourism workers in Lapland (winter season, rapid onboarding)
  - [ ] Option D: Permanent residents with refugee/humanitarian backgrounds (Arabic, Ukrainian, Russian, Somali, etc.)
  - [ ] Option E: Domestic language minorities (Sámi languages: North, Inari, Skolt; Swedish speakers)
- [ ] Document justification for community selection (information gap, urgency, addressable size).
- [ ] Profile community information lifecycles (e.g., short-term seasonal turnover vs. long-term integration).

---

## 2. Agentic Newsroom Architecture & Newsroom Workflow
- [ ] Integrate Kaleva Media's RSS feed (`RSS | Kaleva`) as an input source.
- [ ] Build source monitoring agents:
  - [ ] City council decisions and municipal agendas
  - [ ] Official public authority announcements (Kela, Migri, TE-toimisto, police, healthcare)
  - [ ] Local news and events feeds
- [ ] Implement relevance filter based on the target community's daily needs.
- [ ] Implement explanatory journalism engine:
  - [ ] Add cultural context injection (bridge implied Finnish institutional knowledge).
  - [ ] Focus on "What this means for you" rather than verbatim translation.
- [ ] Align agent pipeline with Kaleva Media's 6 User Needs:
  - [ ] *Information* (What happened)
  - [ ] *Understanding* (Why, context, impact)
  - [ ] *Usefulness* (Everyday utility: permits, transit, housing, rights)
  - [ ] *Emotion* (Human interest, inspiration)
  - [ ] *Break* (Leisure, entertainment)
  - [ ] *Community* (Connection, platforming local voices)
- [ ] Design reverse-bridge agent (translating community issues/stories into Finnish for mainstream coverage).

---

## 3. Human-in-the-Loop (HITL) & Journalistic Integrity
- [ ] Map newsroom roles and agent hierarchy (Editor-in-Chief -> Section Editor -> Agent -> Human QA).
- [ ] Implement review checkpoints ensuring adherence to JSN Guidelines (Journalist’s Guidelines of Finland).
- [ ] Build safety guards for high-stakes topics (legal status, healthcare, labor rights, housing contracts).
- [ ] Create specialized QA workflows for under-resourced languages (e.g., human-verified Sámi translations).
- [ ] Establish transparency and AI disclosure statements on all published assets.

---

## 4. Community Integration & Distribution
- [ ] Map existing community communication hubs (WhatsApp channels, Telegram, Facebook groups, local hubs).
- [ ] Select distribution channel tailored to community habits (chat apps, lightweight web, audio/podcast, short video).
- [ ] Design bidirectional feedback loops:
  - [ ] Story tip submission
  - [ ] Q&A / information request interface
  - [ ] Fact-checking and error-flagging mechanism by community members
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
- [ ] Set up pipeline prototype ingesting sample RSS/official announcements.
- [ ] Implement sample agent pipeline with human approval dashboard.
- [ ] Generate sample localized outputs demonstrating the User Needs model.
- [ ] Prepare pitch deck and demo addressing:
  - [ ] Community centrality
  - [ ] HITL and ethics compliance
  - [ ] Viability and error resilience
  - [ ] Business sustainability and synergy with Kaleva Media