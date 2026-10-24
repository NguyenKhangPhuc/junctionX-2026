import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from app.models.article import Article
from app.schemas.article import (
    ArticleCreate,
    ArticleDetailResponse,
    ArticleResponse,
    ArticleSearchRequest,
    ArticleSearchResponse,
    ArticleSearchResultItem,
)
from app.services.embedding_service import embedding_service
from app.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post(
    "",
    response_model=ArticleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new article",
    description="Registers an article and automatically computes a 1024-dimensional dense text embedding.",
)
def create_article(
    payload: ArticleCreate,
    db: Session = Depends(get_db),
) -> ArticleResponse:
    """Create a new article record and generate its vector embedding.

    Args:
        payload: Article creation parameters.
        db: Database session.

    Returns:
        ArticleResponse: Created article metadata.
    """
    try:
        # Compute 1024-dimension embedding for title + content
        text_to_embed = f"{payload.title}\n{payload.content}"
        embedding_vector = embedding_service.generate_embedding(text_to_embed)

        article = Article(
            title=payload.title,
            content=payload.content,
            original_language=payload.original_language,
            author=payload.author,
            embedding=embedding_vector,
        )
        db.add(article)
        db.commit()
        db.refresh(article)

        logger.info(f"Article created successfully with ID {article.id}.")
        return ArticleResponse(
            id=article.id,
            title=article.title,
            content=article.content,
            original_language=article.original_language,
            author=article.author,
            has_embedding=article.embedding is not None,
            created_at=article.created_at,
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create article: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create article: {e}",
        ) from e


@router.get(
    "",
    response_model=List[ArticleResponse],
    summary="List all articles",
    description="Retrieve all articles ordered chronologically (newest first).",
)
def list_articles(
    db: Session = Depends(get_db),
) -> List[ArticleResponse]:
    """Retrieve all registered articles ordered by creation time descending."""
    articles = db.query(Article).order_by(Article.created_at.desc()).all()
    return [
        ArticleResponse(
            id=a.id,
            title=a.title,
            content=a.content,
            original_language=a.original_language,
            author=a.author,
            has_embedding=a.embedding is not None,
            created_at=a.created_at,
        )
        for a in articles
    ]


def _execute_similarity_search(
    query_text: str,
    limit: int,
    min_similarity: Optional[float],
    db: Session,
) -> ArticleSearchResponse:
    """Internal helper to execute vector similarity search across articles."""
    query_vec = embedding_service.generate_embedding(query_text, is_query=True)
    if not query_vec:
        logger.warning(f"Could not generate query embedding for query: '{query_text}'")
        return ArticleSearchResponse(query=query_text, total_results=0, results=[])

    items: List[ArticleSearchResultItem] = []

    # Check if active database engine supports native PostgreSQL pgvector
    if db.bind and db.bind.dialect.name == "postgresql":
        distance_col = Article.embedding.cosine_distance(query_vec).label("distance")
        sql_query = (
            db.query(Article, distance_col)
            .filter(Article.embedding.isnot(None))
            .order_by(distance_col.asc())
        )
        if min_similarity is not None:
            max_dist = 1.0 - min_similarity
            sql_query = sql_query.filter(distance_col <= max_dist)

        matches = sql_query.limit(limit).all()
        for article, dist in matches:
            dist_val = float(dist) if dist is not None else 1.0
            sim_score = max(0.0, min(1.0, 1.0 - dist_val))
            items.append(
                ArticleSearchResultItem(
                    id=article.id,
                    title=article.title,
                    content=article.content,
                    original_language=article.original_language,
                    author=article.author,
                    created_at=article.created_at,
                    similarity_score=round(sim_score, 4),
                    distance=round(dist_val, 4),
                )
            )
    else:
        # SQLite / in-memory cosine fallback for local automated testing
        import numpy as np

        articles = db.query(Article).filter(Article.embedding.isnot(None)).all()
        scored = []
        q_arr = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)

        for a in articles:
            if a.embedding is None:
                continue
            emb_arr = np.array(a.embedding, dtype=np.float32)
            emb_norm = np.linalg.norm(emb_arr)
            if q_norm == 0 or emb_norm == 0:
                sim = 0.0
            else:
                sim = float(np.dot(q_arr, emb_arr) / (q_norm * emb_norm))
            dist = 1.0 - sim
            if min_similarity is not None and sim < min_similarity:
                continue
            scored.append((a, sim, dist))

        scored.sort(key=lambda x: x[1], reverse=True)
        for article, sim, dist in scored[:limit]:
            items.append(
                ArticleSearchResultItem(
                    id=article.id,
                    title=article.title,
                    content=article.content,
                    original_language=article.original_language,
                    author=article.author,
                    created_at=article.created_at,
                    similarity_score=round(max(0.0, min(1.0, sim)), 4),
                    distance=round(max(0.0, dist), 4),
                )
            )

    return ArticleSearchResponse(
        query=query_text,
        total_results=len(items),
        results=items,
    )


@router.post(
    "/search",
    response_model=ArticleSearchResponse,
    summary="Semantic similarity search over articles (JSON body)",
    description="Vector search across article embeddings using cosine distance, returning relevance similarity points.",
)
def search_articles_post(
    payload: ArticleSearchRequest,
    db: Session = Depends(get_db),
) -> ArticleSearchResponse:
    """Search articles semantically by query text via POST request body."""
    return _execute_similarity_search(
        query_text=payload.query,
        limit=payload.limit,
        min_similarity=payload.min_similarity,
        db=db,
    )


@router.get(
    "/search",
    response_model=ArticleSearchResponse,
    summary="Semantic similarity search over articles (Query param)",
    description="Vector search across article embeddings via URL query parameter, returning relevance similarity points.",
)
def search_articles_get(
    q: str = Query(..., min_length=1, description="Search query string"),
    limit: int = Query(10, ge=1, le=100, description="Max results to return"),
    min_similarity: Optional[float] = Query(None, ge=0.0, le=1.0, description="Min similarity score threshold"),
    db: Session = Depends(get_db),
) -> ArticleSearchResponse:
    """Search articles semantically by query parameter string."""
    return _execute_similarity_search(
        query_text=q,
        limit=limit,
        min_similarity=min_similarity,
        db=db,
    )


@router.get(
    "/{article_id}",
    response_model=ArticleDetailResponse,
    summary="Get article details",
    description="Retrieve a single article along with all associated media videos.",
)
def get_article_by_id(
    article_id: int,
    db: Session = Depends(get_db),
) -> ArticleDetailResponse:
    """Retrieve article details by primary key ID."""
    article = (
        db.query(Article)
        .options(joinedload(Article.videos))
        .filter(Article.id == article_id)
        .first()
    )
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with ID {article_id} not found.",
        )

    return ArticleDetailResponse(
        id=article.id,
        title=article.title,
        content=article.content,
        original_language=article.original_language,
        author=article.author,
        has_embedding=article.embedding is not None,
        created_at=article.created_at,
        videos=article.videos,
    )


@router.delete(
    "/{article_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete article",
    description="Delete an article and cascade delete attached resources.",
)
def delete_article(
    article_id: int,
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Delete an article by primary key ID."""
    article = db.query(Article).filter(Article.id == article_id).first()
    if article is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with ID {article_id} not found.",
        )

    db.delete(article)
    db.commit()
    logger.info(f"Article ID {article_id} deleted successfully.")
    return {"message": f"Article with ID {article_id} deleted successfully."}
