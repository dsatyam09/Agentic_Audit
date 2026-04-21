"""Document parser for PDF, DOCX, and TXT files."""

from __future__ import annotations

import logging
import pathlib

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
}


class DocumentParser:
    """Extract plain text from PDF, DOCX, and TXT documents."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, doc_path: str) -> str:
        """Return the full plain-text content of *doc_path*.

        Parameters
        ----------
        doc_path:
            Filesystem path to a PDF, DOCX, or TXT file.

        Returns
        -------
        str
            Extracted text.  Returns an empty string when the document
            contains no extractable text.

        Raises
        ------
        FileNotFoundError
            If *doc_path* does not exist.
        ValueError
            If the file extension is not supported.
        RuntimeError
            If text extraction fails for any other reason.
        """
        path = pathlib.Path(doc_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        file_type = _SUPPORTED_EXTENSIONS[ext]
        handler = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "txt": self._parse_txt,
        }[file_type]

        try:
            text = handler(path)
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract text from {doc_path}: {exc}"
            ) from exc

        logger.debug(
            "Parsed %s (%s) — %d characters extracted",
            path.name,
            file_type,
            len(text),
        )
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pdf(path: pathlib.Path) -> str:
        import fitz  # pymupdf

        pages: list[str] = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    pages.append(page_text)
        return "\n".join(pages)

    @staticmethod
    def _parse_docx(path: pathlib.Path) -> str:
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs)

    @staticmethod
    def _parse_txt(path: pathlib.Path) -> str:
        return path.read_text(encoding="utf-8")
