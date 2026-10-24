# app/services/community_service.py
import json
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.ingestion_service import IngestionService
from app.services.translation_service import TranslationService

COMMUNITY_SYSTEM_PROMPT = (
    "You are a community journalist at Kaleva Media's 'OuluLens' desk serving "
    "international tech workers and families in Oulu. Always respond with valid JSON only."
)

COMMUNITY_ANSWER_PROMPT = """
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

        try:
            meta = json.loads(question_row.content)
        except Exception:
            meta = {"question_text": question_row.content, "upvotes": 3}

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

        # 3. Call LLM (OpenRouter / Ollama) with 2-argument signature (system_prompt, user_prompt)
        user_prompt = COMMUNITY_ANSWER_PROMPT.format(question=question_text, context=combined_context)
        try:
            llm_raw = self.translator._call_llm(COMMUNITY_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(llm_raw)
        except Exception:
            parsed = {
                "headline_en": f"Community Q&A: {question_row.title}",
                "kaleva_user_need": "Usefulness",
                "draft_answer_script": (
                    f"Here is what Oulu residents need to know regarding '{question_row.title}'. "
                    f"Based on Oulu municipal data and Kaleva reporting: {question_text}"
                ),
                "sources_used": ["Kaleva RSS", "api.ouka.fi"],
                "journalist_verification_note": "Verify municipal details manually before publishing."
            }

        # 4. Create or update Video + VideoDubbedVersion so it enters the Editor Review Queue
        video = self.db.query(Video).filter(Video.article_id == question_row.id).first()
        if not video:
            video = Video(
                file_name=f"community_qa_{question_row.id}.mp4",
                file_path=f"/data/videos/raw/community_qa_{question_row.id}.mp4",
                duration=60.0,
                original_language="fi",
                article_id=question_row.id,
                is_default_file=False,
            )
            self.db.add(video)
            self.db.flush()

        target_lang = question_row.original_language or "en"
        dub = (
            self.db.query(VideoDubbedVersion)
            .filter(
                VideoDubbedVersion.video_id == video.id,
                VideoDubbedVersion.target_language == target_lang,
            )
            .first()
        )
        narration_script = parsed.get("draft_answer_script", "")
        if not dub:
            dub = VideoDubbedVersion(
                video_id=video.id,
                target_language=target_lang,
                status="pending_review",
                transcript_text=narration_script,
                transcript_json=parsed,
            )
            self.db.add(dub)
            self.db.flush()
        else:
            dub.status = "pending_review"
            dub.transcript_text = narration_script
            dub.transcript_json = dict(parsed)

        # Save draft back into the question's JSON metadata and mark as pending_review for HITL
        meta["status"] = "pending_review"
        meta["ai_draft"] = parsed
        meta["dub_id"] = dub.id
        question_row.content = json.dumps(meta)
        self.db.commit()
        return parsed