import React, { useState, useRef, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import type { Plugin } from 'unified';
import type { Root, Element, Text } from 'hast';
import { visit } from 'unist-util-visit';
import type { CitationSpan, ReferenceItem } from '../types';

// Sentinel wrapping a comma-separated list of display labels
const CITE_OPEN = '\u{E000}';
const CITE_CLOSE = '\u{E002}';
const CITE_RE = new RegExp(`${CITE_OPEN}([^${CITE_CLOSE}]*)${CITE_CLOSE}`, 'g');

function buildAnnotatedText(text: string, spans: CitationSpan[]): string {
  // Insert sentinels at span end positions (sorted descending so offsets stay valid)
  const insertions: Array<{ pos: number; sentinel: string }> = [];
  for (const span of spans) {
    const end = Math.min(span.end, text.length);
    if (end <= span.start) continue;
    const sentinel = `${CITE_OPEN}${span.displayLabels.join(',')}${CITE_CLOSE}`;
    insertions.push({ pos: end, sentinel });
  }
  insertions.sort((a, b) => b.pos - a.pos);

  let result = text;
  for (const { pos, sentinel } of insertions) {
    result = result.slice(0, pos) + sentinel + result.slice(pos);
  }
  return result;
}

// Rehype plugin: walks Text nodes, splits on sentinels, and replaces them
// with <span data-cite="labels"> elements so ReactMarkdown can render them.
const rehypeCitePills: Plugin<[], Root> = () => (tree) => {
  visit(tree, 'text', (node: Text, index, parent) => {
    if (!parent || index === undefined) return;
    if (!CITE_RE.test(node.value)) return;
    CITE_RE.lastIndex = 0;

    const children: Array<Text | Element> = [];
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = CITE_RE.exec(node.value)) !== null) {
      if (m.index > last) {
        children.push({ type: 'text', value: node.value.slice(last, m.index) });
      }
      children.push({
        type: 'element',
        tagName: 'cite-pill',
        properties: { 'data-cite': m[1] },
        children: [],
      } as Element);
      last = m.index + m[0].length;
    }
    if (last < node.value.length) {
      children.push({ type: 'text', value: node.value.slice(last) });
    }

    parent.children.splice(index, 1, ...children);
    return index + children.length;
  });
};

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
    ? `p. ${ref.pageNumbers.join(', ')}`
    : null;

  const popoverWidth = 280;
  const left = Math.min(anchorRect.left + window.scrollX, window.innerWidth - popoverWidth - 12);
  const top = anchorRect.top + window.scrollY - 8;

  return (
    <div
      className="fixed z-50 pointer-events-none"
      style={{ left, top, transform: 'translateY(-100%)', width: popoverWidth }}
    >
      <div
        className="rounded-xl border border-[var(--border)] bg-[var(--surface-1)] text-left overflow-hidden animate-fade-in"
        style={{ boxShadow: 'var(--shadow-md)' }}
      >
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

  const annotated = useMemo(
    () => (spans?.length ? buildAnnotatedText(text, spans) : text),
    [text, spans],
  );

  const components: Components = useMemo(() => ({
    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
    strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
    em: ({ children }) => <em>{children}</em>,
    code: ({ children }) => (
      <code className="px-1 py-0.5 rounded text-xs bg-[var(--surface-2)] font-mono">{children}</code>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-2">
        <table className="text-xs border-collapse w-full">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead>{children}</thead>,
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => <tr className="border-b border-[var(--border)]">{children}</tr>,
    th: ({ children }) => (
      <th className="px-2 py-1 text-left font-semibold text-[var(--text-faint)] border border-[var(--border)]">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="px-2 py-1 border border-[var(--border)]">{children}</td>
    ),
    // Custom element injected by rehypeCitePills
    'cite-pill': ({ node }: { node?: Element }) => {
      const labels = ((node?.properties?.['data-cite'] as string) ?? '').split(',').filter(Boolean);
      return (
        <>
          {labels.map((label) => {
            const ref = references.find((r) => r.displayLabel === label);
            return (
              <button
                key={label}
                aria-label={`View source ${label}${ref?.documentName ? `: ${ref.documentName}` : ''}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (ref) onCitationClick(ref);
                }}
                onMouseEnter={(e) => ref && showPopover(ref, e)}
                onMouseLeave={hidePopover}
                className="inline-flex items-center ml-0.5 px-1.5 py-0 text-[10px] font-mono font-bold
                  bg-[var(--accent-subtle)] text-[var(--accent)] border border-[var(--accent)]
                  border-opacity-30 rounded-[10px] cursor-pointer hover:bg-[var(--accent)]
                  hover:text-white hover:border-opacity-100 transition-all align-baseline
                  leading-tight press-scale"
              >
                {label}
              </button>
            );
          })}
        </>
      );
    },
  }), [references, onCitationClick, showPopover, hidePopover]);

  return (
    <>
      <div className="break-words">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeCitePills]}
          components={components}
        >
          {annotated}
        </ReactMarkdown>
      </div>
      {popover && <CitationPopover state={popover} />}
    </>
  );
};
