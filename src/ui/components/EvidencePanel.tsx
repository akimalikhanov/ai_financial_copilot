import React, { useState } from 'react';
import { X, ChevronLeft, ChevronRight, Search, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import { Document, Citation } from '../types';
import { Button } from './ui';

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
  highlight
}) => {
  const [zoom, setZoom] = useState(100);
  
  if (!isOpen) return null;

  const activeDocData = openDocs.find(d => d.doc.id === activeDocId);
  const activeDoc = activeDocData?.doc;
  const currentPage = activeDocData?.page || 1;

  return (
    <div className={`
      fixed inset-0 z-40 flex flex-col bg-[var(--surface-1)] border-l border-[var(--border)] 
      transition-transform duration-300 ease-in-out
      md:relative md:inset-auto md:w-[600px] md:flex-shrink-0 md:transform-none
      ${isOpen ? 'translate-x-0' : 'translate-x-full md:hidden'}
    `}>
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-[var(--border)] bg-[var(--surface-2)]">
        <div className="flex items-center gap-2 overflow-x-auto no-scrollbar max-w-[calc(100%-80px)]">
            {openDocs.length === 0 && <span className="text-sm text-[var(--text-faint)] font-medium">NO EVIDENCE SELECTED</span>}
            {openDocs.map(({ doc }) => (
            <button
                key={doc.id}
                onClick={() => onSwitchDoc(doc.id)}
                className={`
                  flex items-center gap-2 px-3 py-1.5 text-xs font-medium border rounded-md whitespace-nowrap transition-colors
                  ${activeDocId === doc.id 
                    ? 'bg-[var(--surface-3)] border-[var(--border)] text-[var(--text)]' 
                    : 'bg-[var(--bg)] border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--text)]'}
                `}
            >
                <span className="truncate max-w-[100px]">{doc.company}</span>
                <span 
                  className="ml-1 opacity-60 hover:opacity-100 p-0.5 rounded hover:bg-[var(--surface-3)]"
                  onClick={(e) => { e.stopPropagation(); onCloseDoc(doc.id); }}
                >
                  <X size={10} />
                </span>
            </button>
            ))}
        </div>
        <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" onClick={onClose}>
                <X size={18} />
            </Button>
        </div>
      </div>

      {/* Toolbar */}
      {activeDoc && (
        <div className="h-10 border-b border-[var(--border)] bg-[var(--surface-1)] flex items-center justify-between px-4">
            <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0"><ChevronLeft size={14} /></Button>
                <span className="text-xs font-mono text-[var(--text-faint)]">
                    Page {currentPage} / {activeDoc.pages}
                </span>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0"><ChevronRight size={14} /></Button>
            </div>
            <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.max(50, z - 10))}><ZoomOut size={14} /></Button>
                <span className="text-xs font-mono text-[var(--text-faint)] w-12 text-center">{zoom}%</span>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.min(200, z + 10))}><ZoomIn size={14} /></Button>
            </div>
        </div>
      )}

      {/* PDF Canvas Mock */}
      <div className="flex-1 overflow-auto p-8 bg-[var(--bg)] relative flex justify-center">
        {!activeDoc ? (
            <div className="flex flex-col items-center justify-center h-full text-[var(--text-faint)] space-y-4">
                <Search size={48} className="opacity-20" />
                <p className="text-sm text-center max-w-[200px]">Select a citation to view evidence or open a document from Library.</p>
            </div>
        ) : (
            <div 
                className="relative bg-white shadow-xl transition-all duration-300 rounded-sm"
                style={{ 
                    width: `${600 * (zoom / 100)}px`, 
                    height: `${850 * (zoom / 100)}px`,
                }}
            >
                {/* Mock Content */}
                <div className="absolute top-8 left-8 right-8 bottom-8 pointer-events-none">
                    <div className="h-4 w-1/3 bg-gray-200 mb-8 rounded" />
                    <div className="space-y-3">
                        {Array.from({ length: 12 }).map((_, i) => (
                            <div key={i} className="h-2 bg-gray-100 w-full rounded" style={{ width: `${Math.random() * 40 + 60}%`}} />
                        ))}
                    </div>
                    <div className="mt-12 space-y-3">
                        <div className="h-32 bg-gray-50 border border-gray-200 w-full flex items-center justify-center text-gray-300 text-xs font-mono rounded">
                            CHART / TABLE PLACEHOLDER
                        </div>
                    </div>
                    <div className="mt-8 space-y-3">
                         {Array.from({ length: 8 }).map((_, i) => (
                            <div key={i} className="h-2 bg-gray-100 w-full rounded" />
                        ))}
                    </div>
                </div>

                {/* Highlight Overlay */}
                {highlight && (
                    <div 
                        className="absolute bg-[var(--accent)]/20 border-2 border-[var(--accent)]/50 rounded animate-pulse"
                        style={{
                            left: `${highlight.x}%`,
                            top: `${highlight.y}%`,
                            width: `${highlight.w}%`,
                            height: `${highlight.h}%`
                        }}
                    >
                         <div className="absolute -top-6 left-0 bg-[var(--accent)] text-white text-[10px] px-2 py-0.5 font-semibold rounded">
                            CITATION MATCH
                         </div>
                    </div>
                )}
            </div>
        )}
      </div>
    </div>
  );
};
