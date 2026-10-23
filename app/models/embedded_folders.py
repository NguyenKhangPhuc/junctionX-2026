import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.embedded_files import EmbeddedFiles


class FolderKind(str, enum.Enum):
    IMAGE = "image"
    DOCUMENT = "document"


class FolderStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    PARTIAL = "partial"
    FAILED = "failed"


class EmbeddedFolders(Base):
    __tablename__ = "embedded_folders"

    # 1. Định danh UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # 2. Đường dẫn thư mục (tuyệt đối)
    path: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # 3. Loại tài nguyên (image / document)
    kind: Mapped[FolderKind] = mapped_column(
        Enum(
            FolderKind,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
    )

    # 4. Trạng thái tiến trình
    status: Mapped[FolderStatus] = mapped_column(
        Enum(
            FolderStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=FolderStatus.QUEUED,
        index=True,
    )

    # 5. Metadata model embedding
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)

    # 6. Thống kê số lượng file phục vụ progress bar (mặc định = 0)
    files_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_embedded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    files_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 7. Trạng thái hoạt động
    auto_embed_new: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # 8. Mốc thời gian scan & embed (timestamptz)
    last_scanned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # 9. Audit timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # 10. Quan hệ danh sách files thuộc folder
    files: Mapped[list["EmbeddedFiles"]] = relationship(
        "EmbeddedFiles",
        back_populates="folder",
        cascade="all, delete-orphan",
    )

    # 11. Ràng buộc (Constraints)
    __table_args__ = (
        # Ràng buộc không đăng ký trùng cùng 1 path cho cùng 1 kind
        UniqueConstraint("path", "kind", name="uq_managed_folders_path_kind"),
        # Đảm bảo các chỉ số đếm không âm
        CheckConstraint("files_total >= 0", name="chk_files_total_non_negative"),
        CheckConstraint("files_embedded >= 0", name="chk_files_embedded_non_negative"),
        CheckConstraint("files_failed >= 0", name="chk_files_failed_non_negative"),
        CheckConstraint("files_skipped >= 0", name="chk_files_skipped_non_negative"),
        CheckConstraint("embed_dim > 0", name="chk_embed_dim_positive"),
    )
