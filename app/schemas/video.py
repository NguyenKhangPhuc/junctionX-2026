"""Pydantic schemas for Video and Dubbing requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """Schema for individual time-coded speech segments."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcript text for the segment")


class VideoProcessRequest(BaseModel):
    """Request payload to initiate or resume processing and dubbing for a video."""

    video_path: str = Field(..., description="Relative or absolute path to the local video file")
    target_language: str = Field("en", description="Target language code for dubbing (e.g., 'en', 'vi')")
    article_id: Optional[int] = Field(None, description="Optional article ID to link this video to")
    voice: Optional[str] = Field(None, description="Optional specific TTS voice identifier")
    tts_engine: str = Field(
        "xtts",
        description="TTS engine: 'xtts' for voice cloning (default), 'edge' for broadcast neural TTS",
    )


class DubbedVersionResponse(BaseModel):
    """Response schema for a translated and dubbed video version."""

    id: int
    video_id: int
    target_language: str
    transcript_text: Optional[str] = None
    transcript_json: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    dubbed_audio_path: Optional[str] = None
    dubbed_video_path: Optional[str] = None
    status: str
    is_default_file: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VideoProcessResponse(BaseModel):
    """Response returned upon running the end-to-end video dubbing pipeline."""

    video_id: int
    file_name: str
    file_path: str
    original_audio_path: Optional[str] = None
    duration: Optional[float] = None
    original_language: str
    original_transcript_text: Optional[str] = None
    original_transcript_json: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    is_default_file: bool = True
    dubbed_version: DubbedVersionResponse

    model_config = ConfigDict(from_attributes=True)


class VideoDetailResponse(BaseModel):
    """Detailed response containing a video and all of its associated dubbed versions."""

    id: int
    article_id: Optional[int] = None
    file_name: str
    file_path: str
    original_audio_path: Optional[str] = None
    duration: Optional[float] = None
    original_language: str
    original_transcript_text: Optional[str] = None
    original_transcript_json: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    is_default_file: bool = True
    created_at: datetime
    dubbed_versions: List[DubbedVersionResponse] = []

    model_config = ConfigDict(from_attributes=True)


class DubbedLanguagesResponse(BaseModel):
    """Response containing all available dubbed versions and supported language codes for a video."""

    video_id: int
    languages: List[str] = Field(default_factory=list, description="List of target language codes available")
    dubbed_versions: List[DubbedVersionResponse] = Field(
        default_factory=list, description="Full details of each dubbed version"
    )

    model_config = ConfigDict(from_attributes=True)


class DubbedDeleteResponse(BaseModel):
    """Response schema upon deleting a dubbed language version and its files."""

    video_id: int
    target_language: str
    message: str
    deleted_files: List[str] = Field(default_factory=list, description="List of deleted media file paths")