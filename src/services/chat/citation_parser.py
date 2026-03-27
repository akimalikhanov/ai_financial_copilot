"""Incremental streaming parser for [S1], [S2] bracket citation markup.

The parser processes raw LLM output chunk-by-chunk and separates visible text
from hidden citation markers, producing structured AnswerCitationSpan objects
with character offsets relative to the clean (stripped) output.

Each bracket citation [S1] is stripped from visible text. The resulting span
covers from the end of the previous span to the position of the bracket, so
citation pills appear at the end of the cited passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from src.schemas.retrieval import AnswerCitationSpan, DisplayLabelMap, ParserOutput


class _State(Enum):
    TEXT = auto()  # normal text, no bracket in progress
    BRACKET = auto()  # saw '[', accumulating potential citation


# Matches [S1], [S10], [S1,S2], [S1, S2] etc. (only S-prefixed numeric IDs)
_BRACKET_RE = re.compile(r"^S(\d+)(?:,\s*S\d+)*$", re.IGNORECASE)
_MAX_BRACKET_LEN = 30  # safety: flush buffer if it grows too long without ']'


@dataclass
class BracketCitationParser:
    """Incremental streaming parser for ``[S1]`` bracket citation markup.

    Usage::

        parser = BracketCitationParser()
        for raw_chunk in llm_stream:
            result = parser.feed(raw_chunk)
            # result.visible_text  -> clean text to emit as delta
            # result.completed_spans -> spans that closed in this chunk
        final = parser.finalize()
    """

    _state: _State = field(default=_State.TEXT)
    _bracket_buffer: str = field(default="")
    _clean_offset: int = field(default=0)
    _last_span_end: int = field(default=0)
    # Pending refs: multiple consecutive citations like [S1][S2] at the same position
    _pending_refs: list[str] = field(default_factory=list)
    _pending_pos: int = field(default=0)
    _spans: list[AnswerCitationSpan] = field(default_factory=list)
    label_map: DisplayLabelMap = field(default_factory=DisplayLabelMap)
    _finalized: bool = field(default=False)

    def feed(self, chunk: str) -> ParserOutput:
        """Feed a raw LLM output chunk. Returns visible text and completed spans."""
        if self._finalized:
            raise RuntimeError("Parser already finalized")
        visible_parts: list[str] = []
        completed: list[AnswerCitationSpan] = []

        for ch in chunk:
            self._feed_char(ch, visible_parts, completed)

        visible_text = "".join(visible_parts)
        self._clean_offset += len(visible_text)
        for span in completed:
            self.label_map.get_labels_for_refs(span.ref_ids)
        self._spans.extend(completed)
        return ParserOutput(visible_text=visible_text, completed_spans=completed)

    def finalize(self) -> ParserOutput:
        """Flush pending state at stream end. Must be called exactly once."""
        if self._finalized:
            raise RuntimeError("Parser already finalized")
        self._finalized = True

        visible_parts: list[str] = []
        completed: list[AnswerCitationSpan] = []

        if self._state == _State.BRACKET:
            # Incomplete bracket — emit buffered chars as visible text
            visible_parts.append("[" + self._bracket_buffer)

        # Flush any pending consecutive citations that never got a non-ref following char
        if self._pending_refs:
            completed.append(self._flush_pending())

        visible_text = "".join(visible_parts)
        self._clean_offset += len(visible_text)
        for span in completed:
            self.label_map.get_labels_for_refs(span.ref_ids)
        self._spans.extend(completed)
        self._state = _State.TEXT
        self._bracket_buffer = ""
        return ParserOutput(visible_text=visible_text, completed_spans=completed)

    @property
    def all_spans(self) -> list[AnswerCitationSpan]:
        """All spans produced so far (including from finalize)."""
        return list(self._spans)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_pending(self) -> AnswerCitationSpan:
        """Flush accumulated pending refs into a single span."""
        span = AnswerCitationSpan(
            start=self._last_span_end,
            end=self._pending_pos,
            ref_ids=tuple(self._pending_refs),
        )
        self._last_span_end = self._pending_pos
        self._pending_refs = []
        return span

    def _feed_char(
        self,
        ch: str,
        visible: list[str],
        completed: list[AnswerCitationSpan],
    ) -> None:
        if self._state == _State.TEXT:
            self._on_text(ch, visible, completed)
        elif self._state == _State.BRACKET:
            self._on_bracket(ch, visible, completed)

    def _on_text(self, ch: str, visible: list[str], completed: list[AnswerCitationSpan]) -> None:
        if ch == "[":
            # Before starting a new bracket, flush pending refs if previous char was not '['
            # (pending refs are only flushed when we see non-citation content)
            self._state = _State.BRACKET
            self._bracket_buffer = ""
        else:
            # Non-bracket character — flush any pending consecutive citations first
            if self._pending_refs:
                completed.append(self._flush_pending())
            visible.append(ch)

    def _on_bracket(
        self,
        ch: str,
        visible: list[str],
        completed: list[AnswerCitationSpan],
    ) -> None:
        if ch == "]":
            # Check if buffer content matches a citation pattern
            content = self._bracket_buffer.strip()
            if self._is_citation(content):
                # Valid citation — extract ref IDs, record position
                refs = self._parse_refs(content)
                # Position is current clean_offset + len of currently accumulated visible text
                pos = self._clean_offset + len("".join(visible))
                if self._pending_refs:
                    # Another citation immediately follows — merge
                    self._pending_refs.extend(refs)
                    # pos stays at the same point (consecutive citations)
                else:
                    self._pending_refs = list(refs)
                    self._pending_pos = pos
            else:
                # Not a citation — flush as visible text
                if self._pending_refs:
                    completed.append(self._flush_pending())
                visible.append("[" + self._bracket_buffer + "]")
            self._bracket_buffer = ""
            self._state = _State.TEXT
        elif ch == "[":
            # Another '[' before closing — previous bracket is not a citation
            if self._pending_refs:
                completed.append(self._flush_pending())
            visible.append("[" + self._bracket_buffer)
            self._bracket_buffer = ""
            # Stay in BRACKET state for the new '['
        elif len(self._bracket_buffer) >= _MAX_BRACKET_LEN:
            # Safety flush
            if self._pending_refs:
                completed.append(self._flush_pending())
            visible.append("[" + self._bracket_buffer + ch)
            self._bracket_buffer = ""
            self._state = _State.TEXT
        else:
            self._bracket_buffer += ch

    @staticmethod
    def _is_citation(content: str) -> bool:
        return bool(_BRACKET_RE.match(content))

    @staticmethod
    def _parse_refs(content: str) -> list[str]:
        """Extract ref IDs like ['S1', 'S2'] from 'S1, S2' or 'S1,S2'."""
        return [part.strip().upper() for part in content.split(",") if part.strip()]
