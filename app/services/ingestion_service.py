"""Service for ingesting Kaleva free RSS feeds and enriching with Oulu Open Data API."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

import httpx
from sqlalchemy.orm import Session

from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.translation_service import translation_service

logger = logging.getLogger(__name__)

KALEVA_FREE_RSS = "https://www.kaleva.fi/feedit/rss/managed-listing/rss-maksuttomat/"
OUKA_API_BASE = "https://api.ouka.fi/v1"


class IngestionService:
    """Ingests Finnish local news from Kaleva RSS and enriches with Oulu municipal data."""

    async def fetch_kaleva_free_rss(
        self,
        db: Session,
        limit: int = 5,
        create_hitl_drafts: bool = True,
    ) -> List[Article]:
        """Fetch latest free Kaleva RSS items, enrich with Oulu Open Data, and store in DB."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(KALEVA_FREE_RSS)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)
        saved_articles: List[Article] = []

        for item in root.findall("./channel/item")[:limit]:
            title = (item.findtext("title") or "").strip()
            description = (item.findtext("description") or "").strip()
            link = (item.findtext("link") or "").strip()

            if not title:
                continue

            # Deduplicate by title
            existing = db.query(Article).filter(Article.title == title).first()
            if existing:
                saved_articles.append(existing)
                if create_hitl_drafts:
                    self._ensure_hitl_draft_for_article(db, existing)
                continue

            # Enrich with Oulu municipal data if relevant keywords appear
            ouka_context = await self.fetch_ouka_context(f"{title} {description}")
            full_content = (
                f"Summary: {description}\n"
                f"Source: {link}\n"
                f"Oulu City Data: {ouka_context}"
            )

            article = Article(
                title=title,
                content=full_content,
                original_language="fi",
                author="Kaleva RSS (Maksuttomat)",
            )
            db.add(article)
            db.commit()
            db.refresh(article)
            saved_articles.append(article)

            if create_hitl_drafts:
                self._ensure_hitl_draft_for_article(
                    db=db,
                    article=article,
                    ouka_context=ouka_context,
                )

        return saved_articles

    async def fetch_ouka_context(self, text: str) -> str:
        """Query PostgREST endpoints on api.ouka.fi/v1 for local facility context."""
        text_lower = text.lower()
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                # 1. Daycare / School mentions -> query /v1/properties with PostgREST ilike filter
                matched_edu_kw = next(
                    (
                        k
                        for k in ["päiväkoti", "koulu", "lukio", "varhaiskasvatus", "opetus"]
                        if k in text_lower
                    ),
                    None,
                )
                if matched_edu_kw:
                    filter_term = (
                        "päiväkoti"
                        if matched_edu_kw in ("päiväkoti", "varhaiskasvatus")
                        else ("koulu" if matched_edu_kw in ("koulu", "opetus") else matched_edu_kw)
                    )
                    r = await client.get(
                        f"{OUKA_API_BASE}/properties",
                        params={
                            "property_name": f"ilike.*{filter_term}*",
                            "limit": 3,
                        },
                    )
                    data = r.json() if r.status_code == 200 else []
                    if not data:
                        r = await client.get(
                            f"{OUKA_API_BASE}/properties", params={"limit": 3}
                        )
                        data = r.json() if r.status_code == 200 else []

                    if data:
                        names = [
                            p.get("property_name", "")
                            for p in data
                            if isinstance(p, dict) and p.get("property_name")
                        ]
                        if names:
                            return f"Oulu Municipal Properties ({filter_term}): {', '.join(names[:3])}"

                # 2. Sports / Swimming / Winter tracks -> query /v1/exerciseserviceopenhours
                if any(
                    k in text_lower
                    for k in ["liikunta", "uimahalli", "latu", "ouluhalli", "urheilu", "jää"]
                ):
                    r = await client.get(
                        f"{OUKA_API_BASE}/exerciseserviceopenhours", params={"limit": 2}
                    )
                    if r.status_code == 200 and r.json():
                        return f"Oulu Sports Facility Open Data: {r.json()[:2]}"
            except Exception as e:
                logger.warning(f"Oulu Open Data enrichment lookup skipped: {e}")

        return "General Oulu municipal jurisdiction."

    def _ensure_hitl_draft_for_article(
        self,
        db: Session,
        article: Article,
        ouka_context: Optional[str] = None,
        target_language: str = "en",
    ) -> VideoDubbedVersion:
        """Ensure an ingested Article has a Video container and a pending_review DubbedVersion."""
        video = db.query(Video).filter(Video.article_id == article.id).first()
        if not video:
            raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "videos" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Attach to default background/anchor video if available
            default_video = (
                db.query(Video).filter(Video.is_default_file.is_(True)).first()
            )
            existing_mp4s = list(raw_dir.glob("*.mp4"))

            if default_video and default_video.file_path:
                default_file_path = default_video.file_path
                default_duration = default_video.duration or 20.0
            elif existing_mp4s:
                default_file_path = f"/data/videos/raw/{existing_mp4s[0].name}"
                default_duration = 20.0
            else:
                default_file_path = f"/data/videos/raw/kaleva_article_{article.id}.mp4"
                default_duration = 20.0

            video = Video(
                file_name=f"kaleva_article_{article.id}.mp4",
                file_path=default_file_path,
                duration=default_duration,
                article_id=article.id,
                original_language="fi",
                original_transcript_text=f"{article.title}\n\n{article.content}",
                original_transcript_json=[
                    {"start": 0.0, "end": default_duration, "text": article.title}
                ],
                is_default_file=False,
            )
            db.add(video)
            db.commit()
            db.refresh(video)

        dub = (
            db.query(VideoDubbedVersion)
            .filter(
                VideoDubbedVersion.video_id == video.id,
                VideoDubbedVersion.target_language == target_language,
            )
            .first()
        )
        if dub:
            return dub

        logger.info(
            f"Generating explanatory newsroom draft for Article ID {article.id} ('{article.title}')..."
        )
        narration_text, explanatory_payload = (
            translation_service.generate_oulu_explanatory_digest(
                finnish_text=f"Title: {article.title}\n{article.content}",
                ouka_context=ouka_context or "",
                target_lang=target_language,
            )
        )

        dub = VideoDubbedVersion(
            video_id=video.id,
            target_language=target_language,
            transcript_text=narration_text,
            transcript_json=explanatory_payload,
            status="pending_review",
            is_default_file=False,
        )
        db.add(dub)
        db.commit()
        db.refresh(dub)
        return dub


ingestion_service = IngestionService()