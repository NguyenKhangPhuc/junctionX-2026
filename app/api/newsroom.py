"""FastAPI router for Kaleva RSS ingestion and Human-in-the-Loop (HITL) editorial review."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.services.ingestion_service import IngestionService
from app.services.video_pipeline_service import VideoPipelineService
from app.utils.database import get_db

router = APIRouter(tags=["Newsroom HITL"])


class EditorApprovalPayload(BaseModel):
    edited_transcript_text: str
    edited_transcript_json: Optional[Dict[str, Any]] = None
    editor_name: str = "Kaleva Duty Editor"


@router.post("/ingest-kaleva")
def ingest_kaleva_rss(limit: int = 5, db: Session = Depends(get_db)):
    """Pull latest free items from Kaleva RSS, enrich with api.ouka.fi/v1, and create pending_review drafts."""
    service = IngestionService()
    articles = asyncio.run(
        service.fetch_kaleva_free_rss(db, limit=limit, create_hitl_drafts=True)
    )
    return {
        "ingested_count": len(articles),
        "articles": [{"id": a.id, "title": a.title} for a in articles],
    }


@router.get("/review-queue")
def get_review_queue(db: Session = Depends(get_db)):
    """Return all drafts awaiting human review or flagged by the community."""
    order_col = getattr(VideoDubbedVersion, "updated_at", VideoDubbedVersion.id)
    pending = (
        db.query(VideoDubbedVersion)
        .filter(VideoDubbedVersion.status.in_(["pending_review", "flagged_for_review"]))
        .order_by(order_col.desc())
        .all()
    )

    results = []
    for dub in pending:
        video = db.query(Video).filter(Video.id == dub.video_id).first()
        article = (
            db.query(Article).filter(Article.id == video.article_id).first()
            if video and video.article_id
            else None
        )
        t_json = dub.transcript_json if isinstance(dub.transcript_json, dict) else {}

        results.append(
            {
                "id": dub.id,
                "video_id": dub.video_id,
                "article_id": article.id if article else None,
                "status": dub.status,
                "target_language": dub.target_language,
                "original_finnish_title": (
                    article.title if article else (video.file_name if video else "")
                ),
                "original_finnish_text": (
                    article.content
                    if article
                    else (video.original_transcript_text if video else "")
                ),
                "headline_en": t_json.get("headline_en"),
                "relevance_tag": t_json.get("relevance_tag"),
                "transcript_text": dub.transcript_text,
                "expat_explainer_box": t_json.get("expat_explainer_box", []),
                "community_flags": t_json.get("community_flags", []),
                "transcript_json": dub.transcript_json,
            }
        )
    return results


@router.post("/review/{dub_id}/approve")
def approve_draft(
    dub_id: int,
    payload: EditorApprovalPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Accept human edits to the English script and trigger Edge-TTS + FFmpeg in BackgroundTasks."""
    dub = (
        db.query(VideoDubbedVersion)
        .filter(VideoDubbedVersion.id == dub_id)
        .first()
    )
    if not dub:
        raise HTTPException(status_code=404, detail="Draft not found")

    dub.transcript_text = payload.edited_transcript_text
    if payload.edited_transcript_json is not None:
        dub.transcript_json = dict(payload.edited_transcript_json)
        flag_modified(dub, "transcript_json")
    elif isinstance(dub.transcript_json, dict):
        updated_json = dict(dub.transcript_json)
        updated_json["spoken_narration_segments"] = [
            {"start": 0.0, "end": 15.0, "text": payload.edited_transcript_text}
        ]
        dub.transcript_json = updated_json
        flag_modified(dub, "transcript_json")

    dub.status = "processing_media"
    db.commit()

    # Trigger Step 5 (Edge-TTS) & Step 6 (FFmpeg merge) asynchronously
    pipeline = VideoPipelineService()
    background_tasks.add_task(pipeline.render_approved_dub, dub.id)

    return {
        "status": "approved_and_rendering",
        "dub_id": dub.id,
        "approved_by": payload.editor_name,
    }