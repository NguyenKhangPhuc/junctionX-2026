"""Pydantic schemas for file and folder embedding operations."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.embedded_files import FileStatus
from app.models.embedded_folders import FolderKind, FolderStatus


class EmbedFileRequest(BaseModel):
    """Request payload for embedding an individual file."""

    path: str = Field(
        ..., description="Absolute or home-relative path of the target file"
    )
    kind: FolderKind = Field(..., description="Target category: 'image' or 'document'")
    embedded_folder_id: uuid.UUID | None = Field(
        default=None,
        description="Optional ID of parent embedded folder. If omitted, parent folder will be automatically registered.",
    )


class EmbedFileResponse(BaseModel):
    """Response payload returned upon successfully embedding a single file."""

    id: uuid.UUID
    embedded_folder_id: uuid.UUID
    kind: FolderKind
    path: str
    name: str
    extension: str
    mime_type: str
    size_bytes: int
    content_hash: str | None = None
    status: FileStatus
    error_message: str | None = None
    embedding_id: uuid.UUID | None = Field(
        default=None,
        description="ID of generated image embedding or first document chunk",
    )
    chunks_count: int | None = Field(
        default=None,
        description="Number of chunks created if kind is document",
    )
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EmbedFolderRequest(BaseModel):
    """Request payload for initiating folder-wide scanning and embedding."""

    path: str = Field(..., description="Absolute directory path to scan and embed")
    kind: FolderKind = Field(
        ..., description="Resource kind to process: 'image' or 'document'"
    )
    auto_embed_new: bool = Field(
        default=True,
        description="Flag indicating if future new files in this folder should be indexed automatically",
    )


class EmbedFolderResponse(BaseModel):
    """Response returned upon folder registration or embed job dispatch."""

    folder_id: uuid.UUID
    path: str
    kind: FolderKind
    status: FolderStatus
    files_total: int
    files_embedded: int
    files_failed: int
    files_skipped: int
    message: str

    model_config = ConfigDict(from_attributes=True)
