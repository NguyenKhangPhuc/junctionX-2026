"""Document parsing and chunking service supporting PDF, DOCX, PPTX, text and HTML."""

import hashlib
import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class UnsupportedDocumentError(ValueError):
    """Raised when a file extension has no parser implementation."""


class DocumentTooLargeError(ValueError):
    """Raised when a file exceeds the configured ingestion size limit."""


class DocumentParseError(RuntimeError):
    """Raised when a parser fails to extract readable text from a file."""


@dataclass
class ParsedSection:
    """A contiguous block of extracted text tagged with its origin (page/slide/heading)."""

    text: str
    page_number: int | None = None
    section_label: str | None = None


@dataclass
class ParsedDocument:
    """Full extraction result for a document file."""

    path: str
    mime_type: str
    sections: list[ParsedSection] = field(default_factory=list)
    char_count: int = 0

    @property
    def full_text(self) -> str:
        """Concatenated plain text of every extracted section."""
        return "\n".join(section.text for section in self.sections if section.text.strip())


@dataclass
class ChunkRecord:
    """A retrieval-ready text chunk with provenance metadata."""

    text: str
    chunk_index: int
    char_count: int
    page_number: int | None = None
    section_label: str | None = None


class DocumentService:
    """Extracts plain text from heterogeneous document files and splits it into chunks."""

    DOCUMENT_EXTENSIONS: set[str] = {
        ".pdf",
        ".docx",
        ".pptx",
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".csv",
        ".tsv",
        ".log",
        ".html",
        ".htm",
    }
    IMAGE_EXTENSIONS: set[str] = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
    }

    MIME_TYPES: dict[str, str] = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".log": "text/plain",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    # ------------------------------------------------------------------ classification
    @classmethod
    def extension_for(cls, path: str | Path) -> str:
        """Normalized lowercase extension of a path."""
        return Path(path).suffix.lower()

    @classmethod
    def mime_type_for(cls, path: str | Path) -> str:
        """Best-effort MIME type derived from the file extension."""
        return cls.MIME_TYPES.get(cls.extension_for(path), "application/octet-stream")

    @classmethod
    def is_document(cls, path: str | Path) -> bool:
        """True when the file is a parseable text-bearing document."""
        return cls.extension_for(path) in cls.DOCUMENT_EXTENSIONS

    @classmethod
    def is_image(cls, path: str | Path) -> bool:
        """True when the file is a supported raster image."""
        return cls.extension_for(path) in cls.IMAGE_EXTENSIONS

    @classmethod
    def is_supported(cls, path: str | Path) -> bool:
        """True when the file can be indexed as either a document or an image."""
        return cls.is_document(path) or cls.is_image(path)

    # ------------------------------------------------------------------ file helpers
    @staticmethod
    def sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute the SHA-256 digest of file bytes for change detection.

        Args:
            path: Filesystem path of the file.
            chunk_size: Read buffer size in bytes.

        Returns:
            str: Hex-encoded digest.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(chunk_size), b""):
                digest.update(block)
        return digest.hexdigest()

    # ------------------------------------------------------------------ parsing
    def parse(self, path: str | Path, max_bytes: int | None = None) -> ParsedDocument:
        """Extract plain text sections from a document file.

        Args:
            path: Filesystem path to the document.
            max_bytes: Optional size guard; larger files raise :class:`DocumentTooLargeError`.

        Returns:
            ParsedDocument: Extracted sections plus MIME type and character count.

        Raises:
            UnsupportedDocumentError: No parser exists for the extension.
            FileNotFoundError: The path does not exist.
            DocumentTooLargeError: The file exceeds the size guard.
            DocumentParseError: Extraction produced no readable text.
        """
        file_path = Path(path).expanduser()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"Document not found: {file_path}")
        if not self.is_document(file_path):
            raise UnsupportedDocumentError(
                f"Unsupported document extension '{self.extension_for(file_path)}'. "
                f"Supported: {', '.join(sorted(self.DOCUMENT_EXTENSIONS))}"
            )
        size = file_path.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise DocumentTooLargeError(
                f"File {file_path.name} is {size} bytes, exceeding the {max_bytes} byte limit"
            )

        parser = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".pptx": self._parse_pptx,
            ".html": self._parse_html,
            ".htm": self._parse_html,
        }.get(self.extension_for(file_path), lambda p: self._parse_plain_text(p))

        try:
            sections = parser(file_path)
        except (UnsupportedDocumentError, DocumentTooLargeError):
            raise
        except Exception as exc:  # noqa: BLE001 - parser libraries raise heterogeneous errors
            raise DocumentParseError(f"Failed to parse {file_path.name}: {exc}") from exc

        sections = [section for section in sections if section.text.strip()]
        if not sections:
            raise DocumentParseError(f"No extractable text found in {file_path.name}")

        logger.debug(f"Parsed {file_path.name}: {len(sections)} sections, {size} bytes")
        char_count = sum(len(section.text) for section in sections)
        return ParsedDocument(
            path=str(file_path.resolve()),
            mime_type=self.mime_type_for(file_path),
            sections=sections,
            char_count=char_count,
        )

    @staticmethod
    def _parse_pdf(path: Path) -> list[ParsedSection]:
        """Extract one section per PDF page."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        sections: list[ParsedSection] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(ParsedSection(text=text, page_number=page_index))
        return sections

    @staticmethod
    def _parse_docx(path: Path) -> list[ParsedSection]:
        """Extract DOCX paragraphs grouped under the nearest heading label."""
        from docx import Document

        document = Document(str(path))
        sections: list[ParsedSection] = []
        buffer: list[str] = []
        heading: str | None = None

        def flush() -> None:
            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    sections.append(ParsedSection(text=text, section_label=heading))
                buffer.clear()

        for block in document.paragraphs:
            text = block.text.strip()
            if not text:
                continue
            style = (block.style.name or "").lower() if block.style else ""
            if style.startswith("heading") or style.startswith("title"):
                flush()
                heading = text[:200]
                continue
            buffer.append(text)
        flush()

        # Tables carry dense tabular facts; keep them as separate sections
        for table_index, table in enumerate(document.tables, start=1):
            rows: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(
                    ParsedSection(
                        text="\n".join(rows),
                        section_label=heading or f"Table {table_index}",
                    )
                )
        return sections

    @staticmethod
    def _parse_pptx(path: Path) -> list[ParsedSection]:
        """Extract PPTX slide text with slide numbers and speaker notes."""
        from pptx import Presentation

        presentation = Presentation(str(path))
        sections: list[ParsedSection] = []
        for slide_index, slide in enumerate(presentation.slides, start=1):
            parts: list[str] = []
            label: str | None = None
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                text = "\n".join(
                    paragraph.text for paragraph in shape.text_frame.paragraphs
                ).strip()
                if not text:
                    continue
                if label is None:
                    label = text.splitlines()[0][:200]
                parts.append(text)
            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"Notes: {notes}")
            if parts:
                sections.append(
                    ParsedSection(
                        text="\n".join(parts),
                        page_number=slide_index,
                        section_label=label or f"Slide {slide_index}",
                    )
                )
        return sections

    @classmethod
    def _parse_html(cls, path: Path) -> list[ParsedSection]:
        """Strip markup and return readable text blocks from an HTML file."""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = cls.strip_html(raw)
        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        return [ParsedSection(text=block) for block in blocks]

    @staticmethod
    def strip_html(raw: str) -> str:
        """Remove script/style tags and decode entities from an HTML string."""
        without_noise = re.sub(
            r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw or ""
        )
        without_tags = re.sub(r"(?s)<[^>]+>", "\n", without_noise)
        text = html.unescape(without_tags)
        text = "\n".join(line.strip() for line in text.splitlines())
        return re.sub(r"\n{2,}", "\n\n", text).strip()

    @staticmethod
    def _parse_plain_text(path: Path) -> list[ParsedSection]:
        """Read UTF-8 text files, keeping paragraph boundaries intact."""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        blocks = [block.strip() for block in re.split(r"\n{2,}", raw) if block.strip()]
        if not blocks:
            single = raw.strip()
            blocks = [single] if single else []
        return [ParsedSection(text=block) for block in blocks]

    # ------------------------------------------------------------------ chunking
    def chunk_document(
        self,
        document: ParsedDocument,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> list[ChunkRecord]:
        """Split extracted sections into overlapping retrieval chunks.

        Sentence/paragraph boundaries are preferred; oversized blocks are hard split on
        whitespace so that no chunk exceeds ``chunk_size`` characters.

        Args:
            document: Parsed document whose sections are to be split.
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Characters of trailing context carried into the next chunk.

        Returns:
            list[ChunkRecord]: Ordered chunks with provenance metadata.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")

        chunks: list[ChunkRecord] = []
        for section in document.sections:
            for piece in self._split_section(section.text, chunk_size):
                pending = piece
                # Carry trailing text of the previous chunk forward for continuity
                if chunks and chunk_overlap > 0:
                    tail = chunks[-1].text[-chunk_overlap:]
                    if tail and not tail.startswith(pending[: len(tail)]):
                        pending = f"{tail}\n{pending}"
                chunks.append(
                    ChunkRecord(
                        text=pending.strip(),
                        chunk_index=len(chunks),
                        char_count=len(pending.strip()),
                        page_number=section.page_number,
                        section_label=section.section_label,
                    )
                )
        return [chunk for chunk in chunks if chunk.text]

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
    ) -> list[ChunkRecord]:
        """Chunk a raw text blob without needing a parsed document first."""
        parsed = ParsedDocument(
            path="inline",
            mime_type="text/plain",
            sections=[ParsedSection(text=text)],
            char_count=len(text),
        )
        return self.chunk_document(parsed, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def _split_section(text: str, chunk_size: int) -> Iterable[str]:
        """Pack sentences/paragraphs into blocks of at most ``chunk_size`` characters."""
        sentences = DocumentService.split_sentences(text)
        blocks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if len(sentence) > chunk_size:
                if current:
                    blocks.append(" ".join(current).strip())
                    current, current_len = [], 0
                blocks.extend(DocumentService._hard_split(sentence, chunk_size))
                continue

            added = len(sentence) + (1 if current else 0)
            if current_len + added > chunk_size:
                blocks.append(" ".join(current).strip())
                current, current_len = [sentence], len(sentence)
            else:
                current.append(sentence)
                current_len += added

        if current:
            blocks.append(" ".join(current).strip())
        return [block for block in blocks if block]

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Split text into sentence-ish units, keeping whitespace layout sane."""
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?:;])\s+|\n", normalized)
        sentences = [part.strip() for part in parts if part and part.strip()]
        if not sentences:
            return [normalized]
        # Merge orphan fragments shorter than 3 chars into the previous sentence
        merged: list[str] = []
        for sentence in sentences:
            if merged and len(sentence) < 3:
                merged[-1] = f"{merged[-1]} {sentence}"
            else:
                merged.append(sentence)
        return merged

    @staticmethod
    def _hard_split(text: str, chunk_size: int) -> list[str]:
        """Break an oversized run of text on word boundaries."""
        words = text.split()
        if not words:
            return []
        blocks: list[str] = []
        current: list[str] = []
        length = 0
        for word in words:
            if length + len(word) + 1 > chunk_size and current:
                blocks.append(" ".join(current).strip())
                current, length = [word], len(word)
            else:
                current.append(word)
                length += len(word) + 1
        if current:
            blocks.append(" ".join(current).strip())
        return blocks


document_service = DocumentService()
