"""Unit tests for Article CRUD, Vector Embedding, Video Upload, and Dubbed Management APIs."""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.utils.database import Base, get_db

# Use an in-memory SQLite database for fast isolated API testing
TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_article_create_list_and_delete(monkeypatch):
    from app.services.embedding_service import embedding_service
    monkeypatch.setattr(embedding_service, "generate_embedding", lambda text: [0.01] * 1024)

    # 1. Create Article
    payload = {
        "title": "Exploring Helsinki Technology",
        "content": "A short overview of cutting-edge tech in Finland.",
        "original_language": "en",
        "author": "Tech Analyst",
    }
    resp = client.post("/api/articles", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == payload["title"]
    assert data["original_language"] == "en"
    article_id = data["id"]

    # 2. List Articles (ordered newest first)
    list_resp = client.get("/api/articles")
    assert list_resp.status_code == 200
    articles = list_resp.json()
    assert len(articles) == 1
    assert articles[0]["id"] == article_id

    # 3. Get Article by ID
    get_resp = client.get(f"/api/articles/{article_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == article_id
    assert "videos" in get_resp.json()

    # 4. Delete Article
    del_resp = client.delete(f"/api/articles/{article_id}")
    assert del_resp.status_code == 200

    # 5. Verify 404 after deletion
    get_after_del = client.get(f"/api/articles/{article_id}")
    assert get_after_del.status_code == 404


def test_article_similarity_search(monkeypatch):
    from app.services.embedding_service import embedding_service

    # Vector 1: close to query vector
    vec_match = [1.0] + [0.0] * 1023
    # Vector 2: orthogonal to query vector
    vec_unrelated = [0.0, 1.0] + [0.0] * 1022
    # Query vector: close to vec_match
    vec_query = [0.99, 0.05] + [0.0] * 1022

    monkeypatch.setattr(
        embedding_service,
        "generate_embedding",
        lambda text, is_query=False: vec_query if is_query else vec_match,
    )

    db = TestingSessionLocal()
    art1 = Article(
        title="AI in Helsinki",
        content="Artificial intelligence and robotics in Finland.",
        original_language="en",
        embedding=vec_match,
    )
    art2 = Article(
        title="Ancient History of Rome",
        content="Roman empire architecture and history.",
        original_language="en",
        embedding=vec_unrelated,
    )
    db.add_all([art1, art2])
    db.commit()
    db.refresh(art1)
    db.refresh(art2)

    # 1. Test POST /api/articles/search
    search_payload = {
        "query": "artificial intelligence finland",
        "limit": 5,
    }
    resp = client.post("/api/articles/search", json=search_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "artificial intelligence finland"
    assert data["total_results"] == 2
    # Art1 must be ranked first with high similarity score
    assert data["results"][0]["id"] == art1.id
    assert data["results"][0]["similarity_score"] > 0.8
    assert "distance" in data["results"][0]

    # 2. Test GET /api/articles/search with min_similarity filter
    get_resp = client.get("/api/articles/search?q=artificial+intelligence&min_similarity=0.5")
    assert get_resp.status_code == 200
    filtered_data = get_resp.json()
    assert filtered_data["total_results"] == 1
    assert filtered_data["results"][0]["id"] == art1.id



def test_video_upload_endpoint(tmp_path: Path, monkeypatch):
    # Mock ffmpeg_service duration probe
    from app.services.ffmpeg_service import ffmpeg_service
    monkeypatch.setattr(ffmpeg_service, "get_media_duration", lambda path: 12.5)

    fake_video_content = b"fake-video-mp4-stream-data"
    files = {
        "file": ("test_upload_sample.mp4", io.BytesIO(fake_video_content), "video/mp4")
    }
    data = {
        "original_language": "fi",
    }

    resp = client.post("/api/videos", files=files, data=data)
    assert resp.status_code == 201
    video_data = resp.json()
    assert video_data["file_name"] == "test_upload_sample.mp4"
    assert video_data["file_path"] == "/data/videos/raw/test_upload_sample.mp4"
    assert video_data["duration"] == 12.5
    assert video_data["original_language"] == "fi"

    # Verify physical file existence in data/videos/raw
    project_root = Path(__file__).resolve().parent.parent
    saved_file = project_root / "data" / "videos" / "raw" / "test_upload_sample.mp4"
    assert saved_file.exists()
    assert saved_file.read_bytes() == fake_video_content

    # Clean up created file
    saved_file.unlink(missing_ok=True)


def test_dubbed_languages_and_delete(tmp_path: Path):
    db = TestingSessionLocal()
    # 1. Setup a test video
    video = Video(
        file_name="test_dubbed_vid.mp4",
        file_path="/data/videos/raw/test_dubbed_vid.mp4",
        duration=25.0,
        original_language="vi",
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # 2. Setup mock audio and video files on disk
    mock_dubbed_video = tmp_path / "dubbed_en.mp4"
    mock_dubbed_audio = tmp_path / "dubbed_en.wav"
    mock_dubbed_video.write_bytes(b"dummy-mp4")
    mock_dubbed_audio.write_bytes(b"dummy-wav")

    # 3. Create two dubbed versions (en and es)
    dubbed_en = VideoDubbedVersion(
        video_id=video.id,
        target_language="en",
        status="completed",
        dubbed_video_path=str(mock_dubbed_video),
        dubbed_audio_path=str(mock_dubbed_audio),
    )
    dubbed_es = VideoDubbedVersion(
        video_id=video.id,
        target_language="es",
        status="processing",
    )
    db.add_all([dubbed_en, dubbed_es])
    db.commit()

    # 4. GET /api/videos/{video_id}/dubbed
    resp = client.get(f"/api/videos/{video.id}/dubbed")
    assert resp.status_code == 200
    data = resp.json()
    assert data["video_id"] == video.id
    assert set(data["languages"]) == {"en", "es"}
    assert len(data["dubbed_versions"]) == 2

    # 5. DELETE /api/videos/{video_id}/dubbed/en
    del_resp = client.delete(f"/api/videos/{video.id}/dubbed/en")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["target_language"] == "en"
    assert str(mock_dubbed_video) in del_data["deleted_files"]
    assert str(mock_dubbed_audio) in del_data["deleted_files"]

    # Verify physical files deleted
    assert not mock_dubbed_video.exists()
    assert not mock_dubbed_audio.exists()

    # Verify database record deleted
    assert db.query(VideoDubbedVersion).filter(VideoDubbedVersion.video_id == video.id, VideoDubbedVersion.target_language == "en").first() is None
    # Spanish version should still exist
    assert db.query(VideoDubbedVersion).filter(VideoDubbedVersion.video_id == video.id, VideoDubbedVersion.target_language == "es").first() is not None

    # Deleting non-existent language returns 404
    del_again = client.delete(f"/api/videos/{video.id}/dubbed/en")
    assert del_again.status_code == 404
