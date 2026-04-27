import React, { useState, useRef, useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Search } from 'lucide-react';
import { Document, Citation } from '../types';

interface EvidencePanelProps {
  isOpen: boolean;
  onClose: () => void;
  openDocs: { doc: Document; page: number }[];
  activeDocId: string | null;
  onSwitchDoc: (docId: string) => void;
  onCloseDoc: (docId: string) => void;
  highlight?: Citation['bboxHint'];
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  isOpen,
  onClose,
  openDocs,
  activeDocId,
  onSwitchDoc,
  onCloseDoc,
  highlight,
}) => {
  const [zoom, setZoom] = useState(100);

  // Sliding tab indicator
  const tabsRef = useRef<HTMLDivElement>(null);
  const tabBtnRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [tabIndicator, setTabIndicator] = useState<React.CSSProperties>({});

  const activeTabIdx = openDocs.findIndex(d => d.doc.id === activeDocId);

  useEffect(() => {
    const btn = tabBtnRefs.current[activeTabIdx];
    const container = tabsRef.current;
    if (!btn || !container) return;
    const containerRect = container.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    setTabIndicator({
      left: btnRect.left - containerRect.left,
      width: btnRect.width,
    });
  }, [activeTabIdx, activeDocId, openDocs.length]);

  if (!isOpen) return null;

  const activeDocData = openDocs.find(d => d.doc.id === activeDocId);
  const activeDoc = activeDocData?.doc;
  const currentPage = activeDocData?.page || 1;

  return (
    <div className={`
      fixed inset-0 z-40 flex flex-col bg-[var(--surface-1)] border-l border-[var(--border)]
      transition-transform duration-300 ease-in-out animate-slide-in-right
      md:relative md:inset-auto md:w-[600px] md:flex-shrink-0 md:transform-none md:animate-none
    `}>
      {/* Tab bar */}
      <div className="h-14 flex items-center border-b border-[var(--border)] bg-[var(--surface-2)] px-3 gap-2">
        <div
          ref={tabsRef}
          className="relative flex-1 flex items-center gap-1 overflow-x-auto no-scrollbar"
        >
          {/* Sliding tab indicator */}
          {openDocs.length > 0 && (
            <div
              className="absolute bottom-0 h-0.5 bg-[var(--accent)] rounded-full transition-all duration-200 ease-out"
              style={tabIndicator}
            />
          )}

          {openDocs.length === 0 && (
            <span className="text-xs text-[var(--text-faint)] font-medium px-1">No evidence selected</span>
          )}

          {openDocs.map(({ doc }, idx) => (
            <button
              key={doc.id}
              ref={el => { tabBtnRefs.current[idx] = el; }}
              onClick={() => onSwitchDoc(doc.id)}
              className={`
                flex items-center gap-1.5 px-3 py-2 text-xs font-medium whitespace-nowrap transition-colors rounded-t-md
                ${activeDocId === doc.id
                  ? 'text-[var(--text)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'}
              `}
            >
              <span className="truncate max-w-[120px]">{doc.company || doc.title}</span>
              <span
                className="opacity-50 hover:opacity-100 rounded p-0.5 hover:bg-[var(--surface-3)] transition-all"
                onClick={e => { e.stopPropagation(); onCloseDoc(doc.id); }}
              >
                <X size={10} />
              </span>
            </button>
          ))}
        </div>

        <button
          onClick={onClose}
          className="p-1.5 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors flex-shrink-0"
        >
          <X size={16} />
        </button>
      </div>

      {/* PDF Canvas area */}
      <div className="flex-1 overflow-auto p-6 bg-[var(--bg)] relative flex justify-center">
        {!activeDoc ? (
          <div className="flex flex-col items-center justify-center h-full text-[var(--text-faint)] space-y-4">
            <Search size={48} className="opacity-20" />
            <p className="text-sm text-center max-w-[200px]">
              Select a citation to view evidence or open a document from Library.
            </p>
          </div>
        ) : (
          <div
            className="relative bg-white shadow-[var(--shadow-md)] transition-all duration-300 rounded-sm"
            style={{ width: `${600 * (zoom / 100)}px`, height: `${850 * (zoom / 100)}px` }}
          >
            {/* Mock content */}
            <div className="absolute top-8 left-8 right-8 bottom-8 pointer-events-none">
              <div className="h-4 w-1/3 bg-gray-200 mb-8 rounded" />
              <div className="space-y-3">
                {Array.from({ length: 12 }).map((_, i) => (
                  <div key={i} className="h-2 bg-gray-100 rounded" style={{ width: `${60 + (i * 7) % 40}%` }} />
                ))}
              </div>
              <div className="mt-12 space-y-3">
                <div className="h-32 bg-gray-50 border border-gray-200 w-full flex items-center justify-center text-gray-300 text-xs font-mono rounded">
                  CHART / TABLE PLACEHOLDER
                </div>
              </div>
              <div className="mt-8 space-y-3">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="h-2 bg-gray-100 rounded" />
                ))}
              </div>
            </div>

            {/* Highlight overlay */}
            {highlight && (
              <div
                className="absolute bg-[var(--accent)]/20 border-2 border-[var(--accent)]/50 rounded animate-stream-reveal"
                style={{ left: `${highlight.x}%`, top: `${highlight.y}%`, width: `${highlight.w}%`, height: `${highlight.h}%` }}
              >
                <div className="absolute -top-6 left-0 bg-[var(--accent)] text-white text-[10px] px-2 py-0.5 font-semibold rounded">
                  CITATION MATCH
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Floating page controls pill */}
      {activeDoc && (
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 flex items-center gap-1 bg-[var(--surface-1)] border border-[var(--border)] rounded-full px-3 py-1.5 shadow-[var(--shadow-md)] backdrop-blur-sm">
          <button
            className="p-1 rounded-full text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
            onClick={() => {}}
          >
            <ChevronLeft size={14} />
          </button>
          <span className="text-xs font-mono text-[var(--text-muted)] px-1 min-w-[72px] text-center">
            {currentPage} / {activeDoc.pages}
          </span>
          <button
            className="p-1 rounded-full text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
            onClick={() => {}}
          >
            <ChevronRight size={14} />
          </button>
          <div className="w-px h-4 bg-[var(--border)] mx-1" />
          <button
            className="p-1 rounded-full text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
            onClick={() => setZoom(z => Math.max(50, z - 10))}
          >
            <ZoomOut size={13} />
          </button>
          <span className="text-[10px] font-mono text-[var(--text-faint)] w-9 text-center">{zoom}%</span>
          <button
            className="p-1 rounded-full text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
            onClick={() => setZoom(z => Math.min(200, z + 10))}
          >
            <ZoomIn size={13} />
          </button>
        </div>
      )}
    </div>
  );
};
