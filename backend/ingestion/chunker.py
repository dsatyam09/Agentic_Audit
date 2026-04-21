"""Token-aware text chunker for the ingestion pipeline."""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

_DEFAULT_ENCODING = "cl100k_base"
_DEFAULT_CHUNK_SIZE = 512  # tokens
_DEFAULT_OVERLAP = 50  # tokens


class DocumentChunker:
    """Split text into overlapping, token-counted chunks.

    Parameters
    ----------
    chunk_size:
        Maximum number of tokens per chunk (default 512).
    overlap:
        Number of overlapping tokens between consecutive chunks (default 50).
    encoding_name:
        tiktoken encoding name (default ``"cl100k_base"``).
    """

    def __init__(
        self,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
        encoding_name: str = _DEFAULT_ENCODING,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._enc = tiktoken.get_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[dict]:
        """Split *text* into overlapping token-bounded chunks.

        Parameters
        ----------
        text:
            Plain text to chunk.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``chunk_index`` (int): zero-based chunk ordinal.
            - ``chunk_text``  (str): the text of the chunk.
            - ``char_start``  (int): starting character offset in *text*.
            - ``char_end``    (int): ending character offset (exclusive).
        """
        if not text:
            return []

        tokens: list[int] = self._enc.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        step = self._chunk_size - self._overlap
        chunks: list[dict] = []
        chunk_index = 0

        for start_tok in range(0, total_tokens, step):
            end_tok = min(start_tok + self._chunk_size, total_tokens)

            chunk_tokens = tokens[start_tok:end_tok]
            chunk_text = self._enc.decode(chunk_tokens)

            # Compute character offsets by decoding the prefix up to each
            # boundary.  This is exact regardless of multi-byte characters.
            char_start = len(self._enc.decode(tokens[:start_tok]))
            char_end = len(self._enc.decode(tokens[:end_tok]))

            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )

            chunk_index += 1

            # If we've consumed all tokens, stop — avoids a trailing
            # duplicate when the last window aligns exactly.
            if end_tok >= total_tokens:
                break

        logger.debug(
            "Chunked %d tokens into %d chunks (size=%d, overlap=%d)",
            total_tokens,
            len(chunks),
            self._chunk_size,
            self._overlap,
        )
        return chunks
