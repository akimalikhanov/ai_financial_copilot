import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Search } from 'lucide-react';
import { Viewer, Worker } from '@react-pdf-viewer/core';
import type { DocumentLoadEvent, PageChangeEvent, RenderPageProps } from '@react-pdf-viewer/core';
import { pageNavigationPlugin } from '@react-pdf-viewer/page-navigation';
import { zoomPlugin } from '@react-pdf-viewer/zoom';
import '@react-pdf-viewer/core/lib/styles/index.css';
import { Document, Citation } from '../types';
import { getPdfUrl, getAccessTokenValue } from '../services/api';

const WORKER_URL = new URL('pdfjs-dist/build/pdf.worker.min.js', import.meta.url).toString();

interface EvidencePanelProps {
  isOpen: boolean;
  onClose: () => void;
  openDocs: { doc: Document; page: number }[];
  activeDocId: string | null;
  onSwitchDoc: (docId: string) => void;
  onCloseDoc: (docId: string) => void;
  onPageChange: (docId: string, page: number) => void;
  highlight?: NonNullable<Citation['bboxHints']>;
  highlightLabel?: string;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  isOpen,
  onClose,
  openDocs,
  activeDocId,
  onSwitchDoc,
  onCloseDoc,
  onPageChange,
  highlight,
  highlightLabel,
}) => {
  const [zoom, setZoom] = useState(100);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [pageInput, setPageInput] = useState<string>('');
  const panelRef = useRef<HTMLDivElement>(null);
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    try { return parseInt(localStorage.getItem('evidence-panel-width') || '600', 10); } catch { return 600; }
  });
  const isDragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    dragStartX.current = e.clientX;
    dragStartWidth.current = panelRef.current?.offsetWidth ?? panelWidth;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = dragStartX.current - e.clientX;
      const next = Math.min(900, Math.max(320, dragStartWidth.current + delta));
      setPanelWidth(next);
      try { localStorage.setItem('evidence-panel-width', String(next)); } catch { /* noop */ }
    };
    const onUp = () => {
      isDragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [panelWidth]);

  const tabsRef = useRef<HTMLDivElement>(null);
  const tabBtnRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const [tabIndicator, setTabIndicator] = useState<React.CSSProperties>({});

  const activeTabIdx = openDocs.findIndex(d => d.doc.id === activeDocId);
  const activeDocData = openDocs.find(d => d.doc.id === activeDocId);
  const activeDoc = activeDocData?.doc;
  const currentPage = activeDocData?.page ?? 1;

  useEffect(() => {
    const btn = tabBtnRefs.current[activeTabIdx];
    const container = tabsRef.current;
    if (!btn || !container) return;
    const containerRect = container.getBoundingClientRect();
    const btnRect = btn.getBoundingClientRect();
    setTabIndicator({ left: btnRect.left - containerRect.left, width: btnRect.width });
  }, [activeTabIdx, activeDocId, openDocs.length]);

  // Plugin instances — must be called directly in the component body because
  // they internally call React.useMemo themselves.
  const pageNavPlugin = pageNavigationPlugin();
  const zoomPlug = zoomPlugin();
  const { jumpToPage } = pageNavPlugin;
  const { zoomTo: setViewerZoom } = zoomPlug;

  // Track whether the current doc has finished loading
  const docLoadedRef = useRef(false);
  // Track current page in a ref so onDocumentLoad can read the latest value
  const currentPageRef = useRef(currentPage);
  currentPageRef.current = currentPage;
  // Prevent echoing viewer scroll events back as commanded navigation
  const isJumpingRef = useRef(false);

  // Reset per-doc state when active doc switches
  useEffect(() => {
    docLoadedRef.current = false;
    setNumPages(null);
    setZoom(100);
  }, [activeDocId]);

  // Jump to page when externally commanded (citation click, prev/next, page input)
  // Only fires if doc is already loaded; the initial jump is handled in onDocumentLoad
  useEffect(() => {
    if (!docLoadedRef.current) return;
    isJumpingRef.current = true;
    jumpToPage(currentPage - 1);
    const t = setTimeout(() => { isJumpingRef.current = false; }, 800);
    return () => clearTimeout(t);
  }, [currentPage, jumpToPage]);

  // Sync our zoom state to the viewer after doc is loaded
  useEffect(() => {
    if (!docLoadedRef.current) return;
    setViewerZoom(zoom / 100);
  }, [zoom, setViewerZoom]);

  const handleDocLoad = useCallback((e: DocumentLoadEvent) => {
    setNumPages(e.doc.numPages);
    docLoadedRef.current = true;
    // Jump to the requested page now that the document is ready
    isJumpingRef.current = true;
    jumpToPage(currentPageRef.current - 1);
    setTimeout(() => { isJumpingRef.current = false; }, 800);
  }, [jumpToPage]);

  const handlePageChange = useCallback((e: PageChangeEvent) => {
    if (isJumpingRef.current || !activeDoc) return;
    onPageChange(activeDoc.id, e.currentPage + 1);
  }, [activeDoc, onPageChange]);

  // Track zoom changes originating inside the viewer (ctrl+scroll, pinch)
  const handleViewerZoom = useCallback((e: { scale: number }) => {
    setZoom(Math.round(e.scale * 100));
  }, []);

  // Citation highlight overlay rendered on each page.
  // Backend sends bbox in raw PDF points; we convert to CSS % using each page's
  // unscaled PDF point dimensions (props.width / props.scale).
  const renderPage = useCallback((props: RenderPageProps) => {
    const pageHighlight = highlight?.find(h => h.page - 1 === props.pageIndex);
    let style: React.CSSProperties | null = null;
    if (pageHighlight && props.scale > 0) {
      const pageW = props.width / props.scale;
      const pageH = props.height / props.scale;
      const bboxLeft = pageHighlight.left;
      const bboxRight = pageHighlight.right;
      let cssTop: number;
      let cssH: number;
      if ((pageHighlight.coord_origin || '').toUpperCase() === 'BOTTOMLEFT') {
        cssTop = pageH - pageHighlight.top;
        cssH = pageHighlight.top - pageHighlight.bottom;
      } else {
        cssTop = pageHighlight.top;
        cssH = pageHighlight.bottom - pageHighlight.top;
      }
      style = {
        left: `${(bboxLeft / pageW) * 100}%`,
        top: `${(cssTop / pageH) * 100}%`,
        width: `${((bboxRight - bboxLeft) / pageW) * 100}%`,
        height: `${(cssH / pageH) * 100}%`,
      };
    }
    return (
      <>
        {props.canvasLayer.children}
        {props.annotationLayer.children}
        {props.textLayer.children}
        {style && (
          <div
            className="absolute rounded animate-stream-reveal pointer-events-none"
            style={{
              ...style,
              background: 'transparent',
              border: '2px solid rgba(16, 163, 127, 0.85)',
            }}
          >
            {highlightLabel && (
              <div
                className="absolute -top-5 right-0 text-white text-[10px] px-1.5 py-px font-semibold rounded whitespace-nowrap tracking-wide"
                style={{ background: 'rgba(16, 163, 127, 0.85)' }}
              >
                {highlightLabel}
              </div>
            )}
          </div>
        )}
      </>
    );
  }, [highlight, highlightLabel]);

  const httpHeaders = useMemo(() => {
    const token = getAccessTokenValue();
    return token ? { Authorization: `Bearer ${token}` } : {};
  }, [activeDoc?.id]);

  // Keyboard shortcuts
  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === 'INPUT') return;
      if (!activeDoc) return;
      const total = numPages ?? activeDoc.pages;
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        if (currentPage < total) onPageChange(activeDoc.id, currentPage + 1);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        if (currentPage > 1) onPageChange(activeDoc.id, currentPage - 1);
      } else if (e.key === 'Escape') {
        onClose();
      } else if ((e.metaKey || e.ctrlKey) && (e.key === '=' || e.key === '+')) {
        e.preventDefault();
        setZoom(z => Math.min(300, z + 10));
      } else if ((e.metaKey || e.ctrlKey) && e.key === '-') {
        e.preventDefault();
        setZoom(z => Math.max(50, z - 10));
      } else if ((e.metaKey || e.ctrlKey) && e.key === '0') {
        e.preventDefault();
        setZoom(100);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isOpen, activeDoc, currentPage, numPages, onPageChange, onClose]);

  return (
    <div
      ref={panelRef}
      style={{ width: isOpen ? panelWidth : 0 }}
      className={`fixed inset-y-0 right-0 z-40 flex flex-col bg-[var(--surface-1)] border-l border-[var(--border)] transition-[transform,width] duration-300 ease-out md:relative md:inset-auto md:flex-shrink-0 ${
        isOpen ? 'translate-x-0' : 'translate-x-full md:translate-x-0 md:border-l-0 md:overflow-hidden'
      }`}
    >
      {/* Drag-to-resize handle */}
      <div
        onMouseDown={onDragStart}
        className="hidden md:block absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-[var(--accent)] opacity-0 hover:opacity-40 transition-opacity z-10"
      />

      {/* Tab bar */}
      <div className="h-14 flex items-center border-b border-[var(--border)] bg-[var(--surface-2)] px-3 gap-2">
        <div ref={tabsRef} className="relative flex-1 flex items-center gap-1 overflow-x-auto no-scrollbar">
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
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium whitespace-nowrap transition-colors rounded-t-md ${
                activeDocId === doc.id ? 'text-[var(--text)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
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

      {/* PDF area */}
      <div className="flex-1 overflow-hidden bg-[var(--bg)] relative">
        {!activeDoc ? (
          <div className="flex flex-col items-center justify-center h-full text-[var(--text-faint)] space-y-4">
            <Search size={48} className="opacity-20" />
            <p className="text-sm text-center max-w-[200px]">
              Select a citation to view evidence or open a document from Library.
            </p>
          </div>
        ) : (
          <Worker workerUrl={WORKER_URL}>
            <div style={{ height: '100%' }}>
              <Viewer
                key={activeDocId}
                fileUrl={getPdfUrl(activeDoc.id)}
                httpHeaders={httpHeaders}
                withCredentials={true}
                defaultScale={1}
                plugins={[pageNavPlugin, zoomPlug]}
                renderPage={renderPage}
                onDocumentLoad={handleDocLoad}
                onPageChange={handlePageChange}
                onZoom={handleViewerZoom}
                renderLoader={(pct) => (
                  <div className="flex items-center justify-center h-full">
                    <span className="text-xs text-[var(--text-faint)]">Loading PDF… {Math.round(pct)}%</span>
                  </div>
                )}
                renderError={() => (
                  <div className="flex items-center justify-center h-40">
                    <span className="text-xs text-red-400">Failed to load PDF.</span>
                  </div>
                )}
              />
            </div>
          </Worker>
        )}
      </div>

      {/* Page controls bar */}
      {activeDoc && (
        <div className="flex-shrink-0 flex items-center justify-center gap-1 border-t border-[var(--border)] bg-[var(--surface-2)] px-3 py-2">
          <button
            className="p-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors disabled:opacity-30"
            disabled={currentPage <= 1}
            onClick={() => onPageChange(activeDoc.id, currentPage - 1)}
          >
            <ChevronLeft size={14} />
          </button>
          <div className="flex items-center gap-1 px-1">
            <input
              className="w-9 text-center text-xs font-mono bg-[var(--input-bg)] border border-[var(--input-border)] rounded px-1 py-0.5 text-[var(--text)] focus:outline-none focus:border-[var(--input-border-focus)]"
              value={pageInput !== '' ? pageInput : String(currentPage)}
              onChange={e => setPageInput(e.target.value)}
              onFocus={e => { setPageInput(String(currentPage)); e.target.select(); }}
              onBlur={() => {
                const n = parseInt(pageInput, 10);
                const total = numPages ?? activeDoc.pages;
                if (!isNaN(n) && n >= 1 && n <= total) onPageChange(activeDoc.id, n);
                setPageInput('');
              }}
              onKeyDown={e => {
                if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                if (e.key === 'Escape') { setPageInput(''); (e.target as HTMLInputElement).blur(); }
              }}
            />
            <span className="text-xs font-mono text-[var(--text-faint)]">/ {numPages ?? activeDoc.pages}</span>
          </div>
          <button
            className="p-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors disabled:opacity-30"
            disabled={currentPage >= (numPages ?? activeDoc.pages)}
            onClick={() => onPageChange(activeDoc.id, currentPage + 1)}
          >
            <ChevronRight size={14} />
          </button>
          <div className="w-px h-4 bg-[var(--border)] mx-1" />
          <button
            className="p-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors disabled:opacity-30"
            disabled={zoom <= 50}
            onClick={() => setZoom(z => Math.max(50, z - 10))}
          >
            <ZoomOut size={13} />
          </button>
          <button
            className="text-[10px] font-mono text-[var(--text-faint)] w-9 text-center hover:text-[var(--accent)] transition-colors rounded px-1"
            title="Reset to fit width"
            onClick={() => setZoom(100)}
          >
            {zoom}%
          </button>
          <button
            className="p-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-3)] transition-colors disabled:opacity-30"
            disabled={zoom >= 300}
            onClick={() => setZoom(z => Math.min(300, z + 10))}
          >
            <ZoomIn size={13} />
          </button>
        </div>
      )}
    </div>
  );
};
