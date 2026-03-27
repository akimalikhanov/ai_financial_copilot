import React, { useState, useRef, useCallback } from 'react';
import type { CitationSpan, ReferenceItem } from '../types';

interface CitedTextProps {
  text: string;
  spans: CitationSpan[];
  references: ReferenceItem[];
  onCitationClick: (ref: ReferenceItem) => void;
}

interface PopoverState {
  ref: ReferenceItem;
  anchorRect: DOMRect;
}

function CitationPopover({ state }: { state: PopoverState }) {
  const { ref, anchorRect } = state;

  const headingPath = ref.headingPath && ref.headingPath.length > 0
    ? ref.headingPath.join(' › ')
    : null;

  const snippet = ref.snippet
    ? ref.snippet.length > 120 ? ref.snippet.slice(0, 117) + '…' : ref.snippet
    : null;

  const pages = ref.pageNumbers && ref.pageNumbers.length > 0
    ? `p.\u00a0${ref.pageNumbers.join(', ')}`
    : null;

  // Position above the anchor, horizontally left-aligned but clamped
  const popoverWidth = 280;
  const left = Math.min(anchorRect.left + window.scrollX, window.innerWidth - popoverWidth - 12);
  const top = anchorRect.top + window.scrollY - 8; // will shift up via transform

  return (
    <div
      className="fixed z-50 pointer-events-none"
      style={{ left, top, transform: 'translateY(-100%)', width: popoverWidth }}
    >
      <div
        className="rounded-xl border border-[var(--border)] bg-[var(--surface-1)] shadow-xl text-left overflow-hidden"
        style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.18)' }}
      >
        {/* Header row */}
        <div className="flex items-center gap-2 px-3 pt-3 pb-2 border-b border-[var(--border)]">
          <span
            className="inline-flex items-center justify-center h-5 min-w-[1.25rem] px-1 rounded text-[10px] font-mono font-bold
              bg-[var(--accent)] text-white leading-none flex-shrink-0"
          >
            {state.ref.displayLabel}
          </span>
          <span className="text-xs font-semibold text-[var(--text)] leading-tight truncate">
            {ref.documentName || 'Unknown document'}
          </span>
        </div>

        {/* Meta row */}
        <div className="px-3 py-2 space-y-1">
          {headingPath && (
            <div className="text-[10px] text-[var(--accent)] font-medium truncate leading-snug">
              {headingPath}
            </div>
          )}
          {pages && (
            <div className="text-[10px] text-[var(--text-faint)] font-mono leading-snug">
              {pages}
            </div>
          )}
          {snippet && (
            <p className="text-[11px] text-[var(--text-muted,var(--text))] italic leading-snug mt-1 opacity-80">
              "{snippet}"
            </p>
          )}
          {!snippet && !headingPath && !pages && (
            <div className="text-[10px] text-[var(--text-faint)] italic">No preview available</div>
          )}
        </div>
      </div>
      {/* Arrow */}
      <div
        className="absolute left-3 bottom-0 translate-y-full"
        style={{
          width: 0, height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderTop: '6px solid var(--border)',
        }}
      />
    </div>
  );
}

/**
 * Renders message text with inline citation pills placed at the end of cited spans.
 * Pills show a rich hover popover with document title, heading path, page numbers, and snippet.
 */
export const CitedText: React.FC<CitedTextProps> = ({ text, spans, references, onCitationClick }) => {
  const [popover, setPopover] = useState<PopoverState | null>(null);
  const hoverTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showPopover = useCallback((ref: ReferenceItem, e: React.MouseEvent) => {
    if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setPopover({ ref, anchorRect: rect });
  }, []);

  const hidePopover = useCallback(() => {
    hoverTimeout.current = setTimeout(() => setPopover(null), 80);
  }, []);

  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts: React.ReactNode[] = [];
  let cursor = 0;

  for (let i = 0; i < sorted.length; i++) {
    const span = sorted[i];
    const start = Math.max(span.start, cursor);
    const end = Math.min(span.end, text.length);
    if (start >= end) continue;

    if (start > cursor) {
      parts.push(<span key={`t-${cursor}`}>{text.slice(cursor, start)}</span>);
    }

    const citedText = text.slice(start, end);

    const pills = span.displayLabels.map((label) => {
      const ref = references.find((r) => r.displayLabel === label);
      return (
        <button
          key={`pill-${i}-${label}`}
          onClick={(e) => {
            e.stopPropagation();
            if (ref) onCitationClick(ref);
          }}
          onMouseEnter={(e) => ref && showPopover(ref, e)}
          onMouseLeave={hidePopover}
          className="inline-flex items-center ml-0.5 px-1.5 py-0 text-[10px] font-mono font-bold
            bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent)]
            border-opacity-30 rounded-full cursor-pointer hover:bg-[var(--accent)]
            hover:text-white hover:border-opacity-100 transition-all align-baseline
            leading-tight"
        >
          {label}
        </button>
      );
    });

    parts.push(
      <span key={`c-${i}`}>
        {citedText}
        {pills}
      </span>
    );

    cursor = end;
  }

  if (cursor < text.length) {
    parts.push(<span key={`t-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return (
    <>
      <div className="whitespace-pre-wrap break-words">{parts}</div>
      {popover && <CitationPopover state={popover} />}
    </>
  );
};
