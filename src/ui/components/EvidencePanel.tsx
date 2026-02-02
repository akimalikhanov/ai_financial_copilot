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
      fixed inset-0 z-40 flex flex-col bg-zinc-950 border-l border-zinc-800 
      transition-transform duration-300 ease-in-out
      md:relative md:inset-auto md:w-[600px] md:flex-shrink-0 md:transform-none
      ${isOpen ? 'translate-x-0' : 'translate-x-full md:hidden'}
    `}>
      {/* Header */}
      <div className="h-14 flex items-center justify-between px-4 border-b border-zinc-800 bg-zinc-900">
        <div className="flex items-center gap-2 overflow-x-auto no-scrollbar max-w-[calc(100%-80px)]">
            {openDocs.length === 0 && <span className="text-sm text-zinc-500 font-mono">NO EVIDENCE SELECTED</span>}
            {openDocs.map(({ doc }) => (
            <button
                key={doc.id}
                onClick={() => onSwitchDoc(doc.id)}
                className={`
                  flex items-center gap-2 px-3 py-1.5 text-xs font-mono border rounded-sm whitespace-nowrap transition-colors
                  ${activeDocId === doc.id 
                    ? 'bg-zinc-800 border-zinc-600 text-zinc-100' 
                    : 'bg-zinc-950 border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'}
                `}
            >
                <span className="truncate max-w-[100px]">{doc.company}</span>
                <span 
                  className="ml-1 opacity-60 hover:opacity-100 p-0.5 rounded hover:bg-zinc-700"
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
        <div className="h-10 border-b border-zinc-800 bg-zinc-900/50 flex items-center justify-between px-4">
            <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0"><ChevronLeft size={14} /></Button>
                <span className="text-xs font-mono text-zinc-400">
                    Page {currentPage} / {activeDoc.pages}
                </span>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0"><ChevronRight size={14} /></Button>
            </div>
            <div className="flex items-center gap-1">
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.max(50, z - 10))}><ZoomOut size={14} /></Button>
                <span className="text-xs font-mono text-zinc-500 w-12 text-center">{zoom}%</span>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setZoom(z => Math.min(200, z + 10))}><ZoomIn size={14} /></Button>
            </div>
        </div>
      )}

      {/* PDF Canvas Mock */}
      <div className="flex-1 overflow-auto p-8 bg-zinc-950 relative flex justify-center">
        {!activeDoc ? (
            <div className="flex flex-col items-center justify-center h-full text-zinc-600 space-y-4">
                <Search size={48} className="opacity-20" />
                <p className="text-sm font-mono text-center max-w-[200px]">Select a citation to view evidence or open a document from Library.</p>
            </div>
        ) : (
            <div 
                className="relative bg-white shadow-2xl transition-all duration-300"
                style={{ 
                    width: `${600 * (zoom / 100)}px`, 
                    height: `${850 * (zoom / 100)}px`,
                    opacity: 0.9 
                }}
            >
                {/* Mock Content */}
                <div className="absolute top-8 left-8 right-8 bottom-8 pointer-events-none">
                    <div className="h-4 w-1/3 bg-zinc-200 mb-8" />
                    <div className="space-y-3">
                        {Array.from({ length: 12 }).map((_, i) => (
                            <div key={i} className="h-2 bg-zinc-100 w-full" style={{ width: `${Math.random() * 40 + 60}%`}} />
                        ))}
                    </div>
                    <div className="mt-12 space-y-3">
                        <div className="h-32 bg-zinc-50 border border-zinc-200 w-full flex items-center justify-center text-zinc-300 text-xs font-mono">
                            CHART / TABLE PLACEHOLDER
                        </div>
                    </div>
                    <div className="mt-8 space-y-3">
                         {Array.from({ length: 8 }).map((_, i) => (
                            <div key={i} className="h-2 bg-zinc-100 w-full" />
                        ))}
                    </div>
                </div>

                {/* Highlight Overlay */}
                {highlight && (
                    <div 
                        className="absolute bg-yellow-400/30 border border-yellow-500/50 mix-blend-multiply animate-pulse"
                        style={{
                            left: `${highlight.x}%`,
                            top: `${highlight.y}%`,
                            width: `${highlight.w}%`,
                            height: `${highlight.h}%`
                        }}
                    >
                         <div className="absolute -top-6 left-0 bg-yellow-500 text-black text-[10px] px-1 font-bold">
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
