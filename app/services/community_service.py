"""Community co-creation service: semantic retrieval, FAQ drafting, and Reverse-Bridge pitches."""

import json
import logging
from typing import List

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.ingestion_service import IngestionService
from app.services.translation_service import (
    AI_DISCLOSURE_STATEMENT,
    TranslationService,
)

logger = logging.getLogger(__name__)

COMMUNITY_SYSTEM_PROMPT = (
    "You are a community journalist at Kaleva Media's 'OuluLens' desk serving "
    "international tech workers, researchers, and families in Oulu. Always respond with valid JSON only."
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
3. Explain Finnish municipal terms (e.g., Wilma, Kela, Varhaiskasvatus, Talvikunnossapito, YTHS) in plain English.

Return ONLY valid JSON with this structure:
{{
  "headline_en": "Clear answer headline for the Oulu expat community",
  "relevance_tag": "One of: [Family & Daycare, Housing & Energy, Transport & Winter, Healthcare & Kela, Work & Tax, Local Culture]",
  "kaleva_user_need": "One of: [Information, Understanding, Usefulness, Emotion, Break, Community]",
  "draft_answer_script": "60-second spoken English audio briefing script answering the question...",
  "expat_explainer_box": [
    {{
      "finnish_term": "Relevant Finnish term",
      "what_it_means_for_you": "Actionable explanation for international residents in Oulu"
    }}
  ],
  "sources_used": ["List of Kaleva articles or Oulu API endpoints used"],
  "journalist_verification_note": "Note for the human editor on what to double-check before publishing"
}}
"""

REVERSE_BRIDGE_SYSTEM_PROMPT = (
    "Olet Kalevan uutispäällikkö Oulussa. Tehtäväsi on kääntää Oulun kansainvälisen "
    "yhteisön esiin nostamat arjen ongelmat ja kysymykset suomenkielisiksi juttuvinkeiksi "
    "Kalevan toimitukselle JSN:n Journalistin ohjeiden mukaisesti. Vastaa VAIN kelvollisella JSON-muodolla."
)

REVERSE_BRIDGE_PROMPT = """
Oulussa asuva kansainvälinen yhteisö on äänestänyt seuraavan aiheen tärkeäksi:
OTSikko: {title}
KYSYMYS / HAVAINTO: {question_text}
KOHDERYHMÄ (Persona): {persona_tag}
ÄÄNET (Upvotes): {upvotes}

Laadi tästä suomenkielinen juttuvinkki (Reverse-Bridge Pitch) Kalevan uutistoimitukselle, jotta suomenkielinen valtamedia voi tutkia asiaa:
{{
  "otsikko_fi": "Napakka suomenkielinen juttuotsikko Kalevan toimitukselle",
  "kaleva_user_need": "Community",
  "juttuidea_fi": "Miksi tämä ilmiö on uutisarvoinen koko Oululle ja mitä toimittajan tulisi selvittää?",
  "haastateltavat_fi": ["Ehdotetut haastateltavat, esim. Oulun kaupungin sivistys- ja kulttuuripalvelut, International House Oulu, yhteisön edustaja"],
  "taustakonteksti_fi": "Miten aihe liittyy Oulun elinvoimaan ja kansainvälisten osaajien pitovoimaan?"
}}
"""


class CommunityService:
    def __init__(self, db: Session):
        self.db = db
        self.ingestion = IngestionService()
        self.translator = TranslationService()

    def _retrieve_relevant_kaleva_articles(
        self, query_text: str, limit: int = 3
    ) -> List[Article]:
        """Search ingested Kaleva articles using pgvector cosine distance first, with keyword ilike fallback."""
        base_query = self.db.query(Article).filter(
            Article.author != "COMMUNITY_QUESTION"
        )

        # 1. Try cross-lingual semantic vector search (multilingual-e5-large) if available
        try:
            from app.services.embedding_service import embed_query

            query_vec = embed_query(query_text)
            if query_vec is not None:
                semantic_matches = (
                    base_query.filter(Article.embedding.is_not(None))
                    .order_by(Article.embedding.cosine_distance(query_vec))
                    .limit(limit)
                    .all()
                )
                if semantic_matches:
                    return semantic_matches
        except Exception as e:
            logger.debug(
                f"Vector search unavailable or skipped ({e}); falling back to keyword search."
            )

        # 2. Fallback to SQL keyword ilike matching
        keywords = [w for w in query_text.split() if len(w) > 3][:4]
        if keywords:
            filters = [Article.content.ilike(f"%{kw}%") for kw in keywords]
            return base_query.filter(or_(*filters)).limit(limit).all()

        return base_query.order_by(Article.id.desc()).limit(limit).all()

    async def draft_community_answer(self, question_article_id: int) -> dict:
        question_row = (
            self.db.query(Article).filter(Article.id == question_article_id).first()
        )
        if not question_row:
            raise ValueError("Question not found")

        try:
            meta = json.loads(question_row.content)
        except Exception:
            meta = {"question_text": question_row.content, "upvotes": 3}

        question_text = meta.get("question_text", question_row.title)

        # 1. Retrieve matching Kaleva articles via semantic / keyword search
        kaleva_matches = self._retrieve_relevant_kaleva_articles(question_text, limit=3)
        kaleva_context = "\n".join(
            [f"- {a.title}: {a.content[:300]}" for a in kaleva_matches]
        )

        # 2. Query Oulu Open Data API (api.ouka.fi/v1)
        ouka_context = await self.ingestion.fetch_ouka_context(question_text)

        combined_context = (
            f"Kaleva Articles:\n{kaleva_context or 'None found'}\n\n"
            f"Oulu API:\n{ouka_context}"
        )

        # 3. Call LLM (OpenRouter / Ollama)
        user_prompt = COMMUNITY_ANSWER_PROMPT.format(
            question=question_text, context=combined_context
        )
        try:
            llm_raw = self.translator._call_llm(COMMUNITY_SYSTEM_PROMPT, user_prompt)
            cleaned = self.translator._strip_code_fences(llm_raw)
            parsed = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Community LLM draft fallback triggered: {e}")
            parsed = {
                "headline_en": f"Community Q&A: {question_row.title}",
                "relevance_tag": "Family & Daycare",
                "kaleva_user_need": "Usefulness",
                "draft_answer_script": (
                    f"Here is what Oulu residents need to know regarding '{question_row.title}'. "
                    f"Based on Oulu municipal data and Kaleva reporting: {question_text}"
                ),
                "expat_explainer_box": [
                    {
                        "finnish_term": "Oulun kaupunki (City of Oulu)",
                        "what_it_means_for_you": "Official municipal guidance verified with api.ouka.fi.",
                    }
                ],
                "sources_used": ["Kaleva RSS", "api.ouka.fi"],
                "journalist_verification_note": "Verify municipal details manually before publishing.",
            }

        narration_script = parsed.get("draft_answer_script", "")
        parsed.setdefault(
            "spoken_narration_segments",
            [{"start": 0.0, "end": 15.0, "text": narration_script}],
        )
        parsed["ai_disclosure"] = AI_DISCLOSURE_STATEMENT
        parsed = self.translator._apply_editorial_guardrails(parsed, question_text)

        # 4. Create or update Video + VideoDubbedVersion so it enters the Editor Review Queue
        video = (
            self.db.query(Video).filter(Video.article_id == question_row.id).first()
        )
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

    def generate_finnish_newsroom_pitch(self, question_article_id: int) -> dict:
        """Reverse-Bridge Agent: Turn an international community issue into a Finnish story pitch for Kaleva reporters."""
        question_row = (
            self.db.query(Article).filter(Article.id == question_article_id).first()
        )
        if not question_row:
            raise ValueError("Question not found")

        try:
            meta = json.loads(question_row.content)
        except Exception:
            meta = {"question_text": question_row.content, "upvotes": 1}

        question_text = meta.get("question_text", question_row.title)
        persona_tag = meta.get("persona_tag", "Oulu International Resident")
        upvotes = meta.get("upvotes", 1)

        user_prompt = REVERSE_BRIDGE_PROMPT.format(
            title=question_row.title,
            question_text=question_text,
            persona_tag=persona_tag,
            upvotes=upvotes,
        )

        try:
            llm_raw = self.translator._call_llm(
                REVERSE_BRIDGE_SYSTEM_PROMPT, user_prompt
            )
            cleaned = self.translator._strip_code_fences(llm_raw)
            pitch = json.loads(cleaned)
        except Exception as e:
            logger.warning(f"Reverse-bridge LLM fallback triggered: {e}")
            pitch = {
                "otsikko_fi": f"Juttuvinkki kansainväliseltä yhteisöltä: {question_row.title}",
                "kaleva_user_need": "Community",
                "juttuidea_fi": (
                    f"Oulun kansainvälinen yhteisö ({persona_tag}, {upvotes} ääntä) nosti esiin arjen "
                    f"haasteen: '{question_text}'. Selvitetään Oulun kaupungin viranhaltijoilta, miten "
                    f"palvelupolkua ja viestintää voidaan selkeyttää."
                ),
                "haastateltavat_fi": [
                    "Oulun kaupungin asiantuntija / International House Oulu",
                    f"Kysymyksen esittänyt yhteisön jäsen ({persona_tag})",
                ],
                "taustakonteksti_fi": "Kansainvälisten osaajien ja perheiden sitouttaminen Oulun seudulle.",
            }

        pitch["ai_disclosure"] = AI_DISCLOSURE_STATEMENT
        meta["finnish_newsroom_pitch"] = pitch
        question_row.content = json.dumps(meta)
        self.db.commit()
        return pitch