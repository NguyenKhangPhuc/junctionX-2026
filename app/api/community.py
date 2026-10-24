# app/api/community.py
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
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
            meta = {"question_text": r.content, "upvotes": 0, "status": "open_for_voting"}
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
def upvote_question(question_id: int, db: Session = Depends(get_db)):
    row = db.query(Article).filter(
        Article.id == question_id,
        Article.author == "COMMUNITY_QUESTION"
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Community question not found")

    try:
        meta = json.loads(row.content)
    except Exception:
        meta = {"question_text": row.content, "upvotes": 0, "status": "open_for_voting"}

    meta["upvotes"] = meta.get("upvotes", 0) + 1
    row.content = json.dumps(meta)
    db.commit()

    # Automatically draft an explanatory answer for the editor queue when upvotes >= 3
    if meta["upvotes"] >= 3 and meta.get("status") == "open_for_voting":
        service = CommunityService(db)
        draft = asyncio.run(service.draft_community_answer(question_id))
        return {
            "id": row.id,
            "upvotes": meta["upvotes"],
            "auto_drafted_for_editor": True,
            "draft": draft,
        }

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

    if isinstance(dub.transcript_json, dict):
        current_json = dict(dub.transcript_json)
    else:
        current_json = {"segments": list(dub.transcript_json or [])}

    flags = list(current_json.get("community_flags", []))
    flags.append({
        "flagged_segment": payload.flagged_segment,
        "correction_note": payload.correction_note,
        "reporter": payload.reporter_persona
    })
    current_json["community_flags"] = flags

    dub.transcript_json = current_json
    flag_modified(dub, "transcript_json")

    # JSN Accountability: pull back into the human editor review queue
    dub.status = "flagged_for_review"
    db.commit()

    return {
        "dub_id": dub.id,
        "new_status": dub.status,
        "message": "Video pulled back to the Kaleva Editor Queue for human verification."
    }


@router.post("/seed-demo")
def seed_demo_questions(db: Session = Depends(get_db)):
    """Seed 3 realistic Oulu community questions for live demo (idempotent)."""
    existing = db.query(Article).filter(Article.author == "COMMUNITY_QUESTION").count()
    if existing > 0:
        return {"seeded": 0, "message": "Demo questions already exist."}

    demo_questions = [
        {
            "title": "English-speaking daycare (päiväkoti) availability in Linnanmaa?",
            "meta": {
                "question_text": (
                    "We just relocated to Oulu for a job at Nokia. How does Varhaiskasvatus "
                    "application timing work mid-semester in Linnanmaa, and which daycare centers "
                    "offer English support?"
                ),
                "persona_tag": "Nokia Engineer & Parent",
                "upvotes": 14,
                "status": "pending_review",
                "ai_draft": {
                    "headline_en": "Applying for English Daycare (Varhaiskasvatus) in Linnanmaa",
                    "kaleva_user_need": "Usefulness",
                    "draft_answer_script": (
                        "Welcome to Oulu! Municipal early childhood education, known in Finnish as "
                        "Varhaiskasvatus, should normally be applied for four months in advance, "
                        "or within two weeks if you suddenly start employment."
                    ),
                    "sources_used": ["api.ouka.fi", "Kaleva Municipal Desk"],
                    "journalist_verification_note": "Verify current Linnanmaa English group capacity."
                },
            },
        },
        {
            "title": "Winter cycling route maintenance (Talvikunnossapito) on Baana?",
            "meta": {
                "question_text": (
                    "Which main Baana cycling paths between Kaukovainio and the University of Oulu "
                    "get super-plowed first during heavy snowstorms?"
                ),
                "persona_tag": "University of Oulu Researcher",
                "upvotes": 9,
                "status": "pending_review",
                "ai_draft": {
                    "headline_en": "How Oulu Prioritizes Baana Winter Cycling Maintenance",
                    "kaleva_user_need": "Information",
                    "draft_answer_script": (
                        "Oulu's winter maintenance—Talvikunnossapito—prioritizes Class 1A Baana "
                        "corridors whenever snow exceeds two centimeters, ensuring the Linnanmaa "
                        "university route is cleared before morning rush hour."
                    ),
                    "sources_used": ["api.ouka.fi", "Kaleva Traffic Archive"],
                    "journalist_verification_note": "Confirm 2026 Class 1A contractor map."
                },
            },
        },
        {
            "title": "How to interpret school messages on Wilma without fluent Finnish?",
            "meta": {
                "question_text": (
                    "Teachers send daily announcements on Wilma using local school abbreviations. "
                    "What are the most important terms international parents in Oulu should watch for?"
                ),
                "persona_tag": "Oulu Tech Worker & Family",
                "upvotes": 2,  # 1 click away from 3rd-upvote auto-draft in the live demo!
                "status": "open_for_voting",
                "ai_draft": None,
            },
        },
    ]

    created_ids = []
    for item in demo_questions:
        row = Article(
            title=item["title"],
            content=json.dumps(item["meta"]),
            original_language="en",
            author="COMMUNITY_QUESTION",
        )
        db.add(row)
        db.flush()
        created_ids.append(row.id)

    db.commit()
    return {"seeded": len(created_ids), "question_ids": created_ids}