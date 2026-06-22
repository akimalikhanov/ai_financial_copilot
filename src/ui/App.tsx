import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import {
  MessageSquare, BookOpen, Plus, Send, Search,
  LogOut, User, FileText, Trash2, Layers, AlertTriangle,
  ChevronDown, ChevronLeft, ChevronRight, Bot, Sun, Moon, Monitor, SlidersHorizontal,
  Pencil, Loader2, PanelLeft, Sparkles,
} from 'lucide-react';
import { Document, Chat, Message, MessageMetadata, Scope, ViewMode, MobileTab, Citation, ReferenceItem } from './types';
import { CitedText } from './components/CitedText';
import {
  chatEnqueue,
  chatStreamSubscribe,
  fetchChatStats,
  fetchModels,
  getMe,
  listDocuments,
  fetchFilterOptions,
  createConversation,
  fetchMessages,
  updateConversation,
  fetchConversations,
  deleteConversation,
  deleteDocument,
  subscribeIngestionStream,
  ApiError,
  ModelInfo,
  type RequestStatsItem,
  type UserInfo,
  type DocumentListItemResponse,
  type StageEvent,
} from './services/api';
import { useAuth } from './context/AuthContext';
import { LoginPage } from './components/LoginPage';
import { Button, Input, Badge, Card, ChatBubble, Toggle } from './components/ui';
import { ScopeBar } from './components/ScopeBar';
import { EvidencePanel } from './components/EvidencePanel';
import { UploadModal } from './components/UploadModal';
import { DocPickerModal } from './components/DocPickerModal';
import { ControlPane, ModelParams, RequestStats, ModelCapabilities } from './components/ControlPane';
import { MessageActions } from './components/MessageActions';

type ThemeMode = 'light' | 'dark' | 'system';

const FALLBACK_MODELS: ModelInfo[] = [
  { id: 'gpt-4o-mini', name: 'GPT-4o-mini' },
];

const generateId = () => Math.random().toString(36).slice(2, 11);
const mapDocStatus = (s: string): Document['status'] =>
  s === 'ready' ? 'Ready' : s === 'failed' ? 'Error' : 'Processing';

const toYear = (v: unknown): number => {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v);
  if (typeof v === 'string' && v.trim().length > 0) {
    const n = Number(v);
    if (Number.isFinite(n)) return Math.trunc(n);
  }
  return 0;
};

function groupChatsByDate(chats: Chat[]): { label: string; items: Chat[] }[] {
  const now = new Date();
  const today = new Date(now); today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
  const last7 = new Date(today); last7.setDate(today.getDate() - 7);
  const last30 = new Date(today); last30.setDate(today.getDate() - 30);

  const groups = [
    { label: 'Today', items: [] as Chat[] },
    { label: 'Yesterday', items: [] as Chat[] },
    { label: 'Last 7 days', items: [] as Chat[] },
    { label: 'Last 30 days', items: [] as Chat[] },
    { label: 'Older', items: [] as Chat[] },
  ];

  for (const chat of chats) {
    const d = new Date(chat.createdAt);
    if (d >= today) groups[0].items.push(chat);
    else if (d >= yesterday) groups[1].items.push(chat);
    else if (d >= last7) groups[2].items.push(chat);
    else if (d >= last30) groups[3].items.push(chat);
    else groups[4].items.push(chat);
  }

  return groups.filter(g => g.items.length > 0);
}

const toUiMessage = (msg: { id: string; role: string; content: string; created_at: string; metadata?: Record<string, unknown>; feedback?: { rating: 'up' | 'down'; comment?: string | null } | null }): Message => {
  const meta = msg.metadata || {};
  const spans = meta.citation_spans as Array<{ start: number; end: number; ref_ids: string[]; display_labels: string[] }> | undefined;
  const refs = meta.references as Array<Record<string, unknown>> | undefined;
  return {
    id: msg.id,
    role: msg.role as 'user' | 'assistant',
    content: msg.content,
    timestamp: new Date(msg.created_at).getTime(),
    citations: meta.citations as Citation[] | undefined,
    citationSpans: spans?.map((s) => ({ start: s.start, end: s.end, refIds: s.ref_ids, displayLabels: s.display_labels })),
    references: refs?.map((r: Record<string, unknown>) => ({
      refId: r.ref_id as string,
      displayLabel: r.display_label as string,
      chunkId: r.chunk_id as string,
      documentId: r.document_id as string,
      documentName: r.document_name as string,
      filename: (r.filename as string | null),
      pageNumbers: r.page_numbers as number[],
      headingPath: r.heading_path as string[],
      snippet: (r.snippet as string | null),
      bboxHint: (r.bbox_hint as Citation['bboxHint'] | undefined) ?? undefined,
    })),
    feedback: msg.feedback ? { rating: msg.feedback.rating, comment: msg.feedback.comment ?? null } : null,
  };
};

const toUiDoc = (d: DocumentListItemResponse): Document => ({
  id: d.id,
  title: d.extracted_title ?? d.original_filename,
  company: (d.metadata?.company as string | undefined) ?? '',
  year: toYear(d.metadata?.year),
  type: (d.metadata?.type as string | undefined) ?? '',
  pages: d.page_count ?? 0,
  status: mapDocStatus(d.status),
  tags: [],
});

function EvidenceList({
  references,
  onRefClick,
}: {
  references: ReferenceItem[];
  onRefClick: (ref: ReferenceItem) => void;
}) {
  const sorted = [...references].sort((a, b) =>
    a.displayLabel.localeCompare(b.displayLabel, undefined, { numeric: true })
  );
  const [open, setOpen] = useState(false);
  const toggle = () => setOpen((o) => !o);

  return (
    <div className="mt-3">
      <button
        onClick={toggle}
        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg cursor-pointer
          hover:bg-[var(--surface-1)] transition-colors group mb-1"
      >
        <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--text-faint)] group-hover:text-[var(--accent)] transition-colors flex-shrink-0">
          Evidence
        </span>
        {!open && (() => {
          const PILL_LIMIT = 2;
          const visible = sorted.slice(0, PILL_LIMIT);
          const overflow = sorted.length - PILL_LIMIT;
          return (
            <div className="flex items-center gap-1 flex-1 overflow-hidden">
              {visible.map((ref) => (
                <span
                  key={ref.refId}
                  className="inline-flex items-center justify-center h-4 min-w-[1.25rem] px-1 rounded
                    text-[9px] font-mono font-bold bg-[var(--accent-subtle)] text-[var(--accent)]
                    border border-[var(--accent)] border-opacity-30 leading-none flex-shrink-0"
                >
                  {ref.displayLabel}
                </span>
              ))}
              {overflow > 0 && (
                <span className="text-[9px] text-[var(--text-faint)] font-mono flex-shrink-0">
                  +{overflow} more
                </span>
              )}
            </div>
          );
        })()}
        {open && <div className="flex-1 h-px bg-[var(--border)]" />}
        <ChevronDown
          size={12}
          className={`text-[var(--text-faint)] group-hover:text-[var(--accent)] transition-all flex-shrink-0 ${open ? 'rotate-180' : ''}`}
        />
      </button>

      <div
        className="grid transition-[grid-template-rows] duration-300 ease-in-out"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="space-y-1 pb-0.5">
            {sorted.map((ref) => {
              const headingPath =
                ref.headingPath && ref.headingPath.length > 0 ? ref.headingPath.join(' › ') : null;
              const pages =
                ref.pageNumbers && ref.pageNumbers.length > 0
                  ? `p. ${ref.pageNumbers.join(', ')}`
                  : null;
              const snippet =
                ref.snippet
                  ? ref.snippet.length > 140
                    ? ref.snippet.slice(0, 137) + '…'
                    : ref.snippet
                  : null;

              return (
                <button
                  key={ref.refId}
                  onClick={() => onRefClick(ref)}
                  className="w-full flex items-stretch gap-3 bg-[var(--surface-1)] border border-[var(--border)]
                    hover:border-[var(--accent)] hover:bg-[var(--surface-2)] rounded-xl px-3 py-2.5
                    text-left transition-all group"
                >
                  <div className="flex-shrink-0 flex items-start pt-0.5">
                    <span
                      className="inline-flex items-center justify-center h-5 min-w-[1.5rem] px-1 rounded
                        text-[10px] font-mono font-bold bg-[var(--accent-subtle)] text-[var(--accent)]
                        border border-[var(--accent)] border-opacity-30
                        group-hover:bg-[var(--accent)] group-hover:text-white group-hover:border-opacity-100
                        transition-all leading-none"
                    >
                      {ref.displayLabel}
                    </span>
                  </div>
                  <div className="min-w-0 flex-1 space-y-0.5">
                    <div className="text-xs font-semibold text-[var(--text)] truncate leading-snug">
                      {ref.documentName || 'Unknown document'}
                    </div>
                    {headingPath && (
                      <div className="text-[10px] text-[var(--accent)] truncate leading-snug opacity-80">
                        {headingPath}
                      </div>
                    )}
                    {pages && (
                      <div className="text-[10px] text-[var(--text-faint)] font-mono leading-snug">
                        {pages}
                      </div>
                    )}
                    {snippet && (
                      <p className="text-[11px] text-[var(--text-faint)] italic leading-snug mt-1 line-clamp-2">
                        "{snippet}"
                      </p>
                    )}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  load_and_validate_request: 'Loading request',
  build_conversation_context: 'Building context',
  scan_user_input: 'Scanning input',
  route_query: 'Routing query',
  agent_loop: 'Searching documents',
  // classic (non-agent) fallback stages — shown if agent loop is disabled
  transform_query: 'Transforming query',
  build_rag_context: 'Retrieving sources',
  render_prompt: 'Rendering prompt',
  stream_llm_response: 'Generating answer',
  persist_and_emit: 'Finalizing',
};

// Ordered stages as seen by the agent path (used for the stepper)
const AGENT_STAGES = [
  'load_and_validate_request',
  'build_conversation_context',
  'scan_user_input',
  'route_query',
  'agent_loop',
  'render_prompt',
  'stream_llm_response',
  'persist_and_emit',
];

// Ordered stages for classic (non-agent) path
const CLASSIC_STAGES = [
  'load_and_validate_request',
  'build_conversation_context',
  'scan_user_input',
  'route_query',
  'transform_query',
  'build_rag_context',
  'render_prompt',
  'stream_llm_response',
  'persist_and_emit',
];

const MD_COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
  strong: ({ children }) => <strong className="font-semibold text-[var(--text)]">{children}</strong>,
  em: ({ children }) => <em>{children}</em>,
  code: ({ children }) => (
    <code className="px-1 py-0.5 rounded text-xs bg-[var(--surface-2)] font-mono">{children}</code>
  ),
  pre: ({ children }) => (
    <pre className="p-3 rounded-lg bg-[var(--surface-2)] overflow-x-auto text-xs font-mono mb-3">{children}</pre>
  ),
  ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  h1: ({ children }) => <h1 className="text-lg font-semibold mb-2 mt-4 first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="text-base font-semibold mb-2 mt-3 first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="text-sm font-semibold mb-1.5 mt-3 first:mt-0">{children}</h3>,
  table: ({ children }) => (
    <div className="overflow-x-auto mb-3">
      <table className="text-xs border-collapse w-full">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="px-3 py-1.5 text-left border border-[var(--border)] bg-[var(--surface-2)] font-medium">{children}</th>
  ),
  td: ({ children }) => <td className="px-3 py-1.5 border border-[var(--border)]">{children}</td>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-[var(--accent)] pl-3 italic text-[var(--text-muted)] mb-3">{children}</blockquote>
  ),
  a: ({ href, children }) => (
    <a href={href} className="text-[var(--accent)] underline" target="_blank" rel="noopener noreferrer">{children}</a>
  ),
};

const INGEST_STAGE_LABELS: Record<string, string> = {
  fetch_document_record: 'Starting',
  download_pdf: 'Downloading',
  parse_pdf_docling: 'Parsing',
  export_docling_artifacts: 'Exporting',
  save_metadata_and_upload_artifacts: 'Saving',
  chunk_document: 'Chunking',
  summarize_table_chunks: 'Summarizing',
  persist_chunks_postgres: 'Persisting',
  embed_chunks: 'Embedding',
  ensure_vector_and_search_indexes: 'Indexing',
  index_and_backup_chunks: 'Indexing',
  finalize_ready: 'Finalizing',
};
interface StageRecord { stage: string; startedAt: number; endedAt?: number; }

interface ToolCallRecord { entity: string; startedAt: number; endedAt?: number; chunksReturned?: number; }

interface StageSnapshot {
  current: StageEvent;
  records: StageRecord[];
  toolCalls: ToolCallRecord[];
  done: boolean;
}

function AgentTimeline({ snapshot }: { snapshot: StageSnapshot }) {
  const [expanded, setExpanded] = React.useState(false);
  const [tick, setTick] = React.useState(0);
  const prevStageRef = React.useRef(snapshot.current.stage);
  const [labelKey, setLabelKey] = React.useState(0);

  React.useEffect(() => {
    if (snapshot.current.stage !== prevStageRef.current) {
      prevStageRef.current = snapshot.current.stage;
      setLabelKey(k => k + 1);
    }
  }, [snapshot.current.stage]);

  React.useEffect(() => {
    if (snapshot.done) return;
    const id = setInterval(() => setTick(t => t + 1), 100);
    return () => clearInterval(id);
  }, [snapshot.done]);

  void tick;

  // Default to agent path; only switch to classic if transform_query is seen
  const isClassicPath = snapshot.records.some(r => r.stage === 'transform_query') ||
    snapshot.current.stage === 'transform_query';
  const stages = isClassicPath ? CLASSIC_STAGES : AGENT_STAGES;

  const first = snapshot.records[0]?.startedAt;
  const last = snapshot.records[snapshot.records.length - 1]?.endedAt;
  const totalElapsed = snapshot.done
    ? (first && last ? ((last - first) / 1000).toFixed(1) : null)
    : (first ? ((Date.now() - first) / 1000).toFixed(1) : null);

  const label = snapshot.done
    ? 'Done'
    : STAGE_LABELS[snapshot.current.stage] ?? snapshot.current.stage.replace(/_/g, ' ');

  return (
    <div className="pl-4 py-1">
      {/* Collapsed pill */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-[var(--border)] bg-[var(--surface-1)] hover:bg-[var(--surface-2)] text-xs"
        style={{ transition: 'background 0.2s, border-color 0.2s' }}
      >
        {snapshot.done ? (
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)] opacity-60 flex-shrink-0" />
        ) : (
          <span className="relative flex h-2 w-2 flex-shrink-0">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--accent)] opacity-40" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[var(--accent)]" />
          </span>
        )}

        <span
          key={labelKey}
          className="stage-label-change font-medium text-[var(--text)]"
          style={{ minWidth: '7rem', textAlign: 'left' }}
        >
          {label}
        </span>

        {!snapshot.done && (
          <span className="text-[var(--text-faint)]">{snapshot.current.index}/{snapshot.current.total}</span>
        )}

        {totalElapsed !== null && (
          <span className="text-[var(--text-faint)]">· {totalElapsed}s</span>
        )}

        <ChevronDown
          size={11}
          className="text-[var(--text-faint)] ml-0.5"
          style={{ transition: 'transform 0.25s ease', transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
        />
      </button>

      {/* Expanded stepper */}
      <div
        className="grid"
        style={{
          gridTemplateRows: expanded ? '1fr' : '0fr',
          opacity: expanded ? 1 : 0,
          transition: 'grid-template-rows 0.3s ease, opacity 0.25s ease',
        }}
      >
        <div className="overflow-hidden">
          <div className="mt-2.5 ml-1 flex flex-col">
            {stages.map((s, i) => {
              const rec = snapshot.records.find(r => r.stage === s);
              const isDone = !!rec?.endedAt;
              const isActive = !snapshot.done && s === snapshot.current.stage;
              const isPending = !rec;

              const dur = rec?.endedAt
                ? ((rec.endedAt - rec.startedAt) / 1000).toFixed(2) + 's'
                : isActive
                  ? ((Date.now() - rec!.startedAt) / 1000).toFixed(1) + 's'
                  : null;

              // Skip optional scan_user_input if it never ran
              if (s === 'scan_user_input' && isPending) return null;

              return (
                <React.Fragment key={s}>
                  <div className="flex items-stretch gap-3">
                    {/* Track column */}
                    <div className="flex flex-col items-center flex-shrink-0" style={{ width: 14 }}>
                      <div
                        className="w-px flex-1"
                        style={{
                          minHeight: i === 0 ? 0 : 6,
                          background: isDone || isActive ? 'var(--accent)' : 'var(--border)',
                          transition: 'background 0.4s ease',
                          visibility: i === 0 ? 'hidden' : 'visible',
                        }}
                      />
                      <div
                        style={{
                          width: 12, height: 12,
                          borderRadius: '50%',
                          flexShrink: 0,
                          border: `2px solid ${isDone || isActive ? 'var(--accent)' : 'var(--border)'}`,
                          background: isDone ? 'var(--accent)' : 'transparent',
                          transition: 'border-color 0.4s ease, background 0.4s ease',
                          animation: isActive ? 'pulse 1.4s ease-in-out infinite' : undefined,
                        }}
                      />
                      <div
                        className="w-px flex-1"
                        style={{
                          minHeight: i === stages.length - 1 ? 0 : 6,
                          background: isDone ? 'var(--accent)' : 'var(--border)',
                          transition: 'background 0.4s ease',
                          visibility: i === stages.length - 1 ? 'hidden' : 'visible',
                        }}
                      />
                    </div>

                    <div className="flex items-center gap-2 py-1" style={{ minHeight: 28 }}>
                      <span
                        className={`text-xs leading-tight ${isActive ? 'stage-label-change' : ''}`}
                        style={{
                          color: isActive ? 'var(--text)' : 'var(--text-faint)',
                          fontWeight: isActive ? 500 : 400,
                          opacity: isPending ? 0.4 : 1,
                          transition: 'color 0.3s, opacity 0.3s',
                        }}
                      >
                        {STAGE_LABELS[s]}
                      </span>
                      {dur && (
                        <span
                          className="text-[10px] font-mono flex-shrink-0 ml-1"
                          style={{
                            color: 'var(--text-faint)',
                            opacity: isActive ? 0.65 : 0.45,
                            transition: 'opacity 0.3s',
                          }}
                        >
                          {dur}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Tool call sub-items under agent_loop */}
                  {s === 'agent_loop' && snapshot.toolCalls.length > 0 && (
                    <div className="ml-8 mb-1 flex flex-col gap-1">
                      {snapshot.toolCalls.map((tc, ti) => {
                        const tcDone = !!tc.endedAt;
                        const tcDur = tc.endedAt
                          ? ((tc.endedAt - tc.startedAt) / 1000).toFixed(2) + 's'
                          : ((Date.now() - tc.startedAt) / 1000).toFixed(1) + 's';
                        return (
                          <div
                            key={`${tc.entity}-${ti}`}
                            className="tc-row-enter flex items-center gap-2"
                            style={{ animationDelay: `${ti * 40}ms` }}
                          >
                            {/* Status dot */}
                            <span
                              className={tcDone ? 'tc-dot-pop' : ''}
                              style={{
                                display: 'inline-block',
                                width: 6, height: 6,
                                borderRadius: '50%',
                                flexShrink: 0,
                                background: tcDone ? 'var(--accent)' : 'var(--text-faint)',
                                opacity: tcDone ? 0.8 : 0.35,
                                transition: 'background 0.3s ease, opacity 0.3s ease',
                              }}
                            />

                            {/* Entity name — shimmer while running, solid when done */}
                            <span
                              className={`text-[10px] leading-none ${!tcDone ? 'tc-shimmer' : ''}`}
                              style={tcDone ? { color: 'var(--text-faint)' } : {}}
                            >
                              {tc.entity}
                            </span>

                            {/* Chunk count badge — fades in on completion */}
                            {tc.chunksReturned !== undefined && (
                              <span
                                className="text-[9px] px-1 py-px rounded-full"
                                style={{
                                  background: 'var(--surface-2)',
                                  color: 'var(--text-faint)',
                                  border: '1px solid var(--border)',
                                  opacity: tcDone ? 0.75 : 0,
                                  transition: 'opacity 0.4s ease',
                                }}
                              >
                                {tc.chunksReturned} chunks
                              </span>
                            )}

                            {/* Elapsed time */}
                            <span
                              className="text-[9px] font-mono ml-1"
                              style={{
                                color: 'var(--text-faint)',
                                opacity: tcDone ? 0.45 : 0.3,
                                transition: 'opacity 0.3s ease',
                              }}
                            >
                              {tcDur}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { authChecked, isAuthenticated, setAccessToken, logout } = useAuth();

  // --- Global State ---
  const [view, setView] = useState<ViewMode>('ASK');
  const [mobileTab, setMobileTab] = useState<MobileTab>('CONVERSATION');
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    try {
      const saved = localStorage.getItem('theme') as ThemeMode | null;
      return saved && ['light', 'dark', 'system'].includes(saved) ? saved : 'system';
    } catch { return 'system'; }
  });

  // Data
  const [docs, setDocs] = useState<Document[]>([]);
  const [chats, setChats] = useState<Chat[]>([]);
  const [filterOptions, setFilterOptions] = useState<{ companies: string[]; years: number[] }>({ companies: [], years: [] });

  // Messages (fetched from backend, keyed by conversationId)
  const [messagesByConversation, setMessagesByConversation] = useState<Record<string, Message[]>>({});
  const [messagesLoading, setMessagesLoading] = useState<Record<string, boolean>>({});
  const [messagesHasMore, setMessagesHasMore] = useState<Record<string, boolean>>({});
  const [messagesMinSeq, setMessagesMinSeq] = useState<Record<string, number>>({});

  // Models
  const [models, setModels] = useState<ModelInfo[]>(FALLBACK_MODELS);
  const [modelsLoading, setModelsLoading] = useState(true);

  // Active Context
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState(FALLBACK_MODELS[0].id);
  const [scope, setScope] = useState<Scope>({
    mode: 'allDocs',
    docIds: [],
    filters: {}
  });

  // Evidence Panel State
  const [isEvidenceOpen, setIsEvidenceOpen] = useState(false);
  const [openPdfTabs, setOpenPdfTabs] = useState<{ doc: Document, page: number }[]>([]);
  const [activePdfDocId, setActivePdfDocId] = useState<string | null>(null);
  const [activeHighlight, setActiveHighlight] = useState<{ bbox: Citation['bboxHint']; label: string } | undefined>(undefined);

  // UI State
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem('sidebar-collapsed') === 'true'; } catch { return false; }
  });
  const [isComposerFocused, setIsComposerFocused] = useState(false);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isAwaitingResponse, setIsAwaitingResponse] = useState(false);
  // keyed by assistant message id — persists the snapshot after streaming ends
  const [stageSnapshots, setStageSnapshots] = useState<Record<string, StageSnapshot>>({});
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isDocPickerOpen, setIsDocPickerOpen] = useState(false);

  // Control Pane State
  const [isControlPaneOpen, setIsControlPaneOpen] = useState(false);
  const [modelParams, setModelParams] = useState<ModelParams>({
    temperature: 0.2,
    maxTokens: 2000,
    reasoningEffort: null,
    verbosity: null,
  });
  const [lastRequestStats, setLastRequestStats] = useState<RequestStats | null>(null);
  const [statsHistory, setStatsHistory] = useState<RequestStats[]>([]);

  // Current user
  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);

  // Chat search
  const [chatSearch, setChatSearch] = useState('');

  // Delete / Rename state
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>('');

  // Library state
  const [libSearch, setLibSearch] = useState('');
  const [libCompany, setLibCompany] = useState('');
  const [libYear, setLibYear] = useState('');
  const [docToDelete, setDocToDelete] = useState<string | null>(null);
  const [docDeleteLoading, setDocDeleteLoading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // --- Theme ---
  useEffect(() => {
    try { localStorage.setItem('theme', themeMode); } catch { /* noop */ }

    const apply = (prefersDark: boolean) => {
      const effective = themeMode === 'system' ? (prefersDark ? 'dark' : 'light') : themeMode;
      document.documentElement.classList.toggle('dark', effective === 'dark');
      document.documentElement.classList.toggle('light', effective === 'light');
    };

    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    apply(mq.matches);

    if (themeMode === 'system') {
      const listener = (e: MediaQueryListEvent) => apply(e.matches);
      mq.addEventListener('change', listener);
      return () => mq.removeEventListener('change', listener);
    }
  }, [themeMode]);

  useEffect(() => {
    try { localStorage.setItem('sidebar-collapsed', String(isSidebarCollapsed)); } catch { /* noop */ }
  }, [isSidebarCollapsed]);

  // Close mobile sidebar on Escape
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isSidebarOpen) setIsSidebarOpen(false);
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [isSidebarOpen]);

  const cycleTheme = () => {
    setThemeMode(m => m === 'light' ? 'dark' : m === 'dark' ? 'system' : 'light');
  };
  const themeIcon = themeMode === 'light' ? <Sun size={18} /> : themeMode === 'dark' ? <Moon size={18} /> : <Monitor size={18} />;
  const themeLabel = themeMode === 'light' ? 'Light mode' : themeMode === 'dark' ? 'Dark mode' : 'System mode';

  // --- Derived State ---
  const activeChat = chats.find(c => c.id === activeChatId);
  const activeMessages = activeChat?.conversationId
    ? messagesByConversation[activeChat.conversationId] || []
    : [];
  const activeDocsCount = React.useMemo(() => {
    if (scope.mode === 'allDocs') return docs.length;
    if (scope.mode === 'filteredByMetadata') {
       return docs.filter(d => {
         const f = scope.filters;
         const matchCompany = !f.company?.length || f.company.includes(d.company);
         const matchYear = !f.year?.length || f.year.includes(d.year);
         const matchType = !f.type?.length || f.type.includes(d.type);
         return matchCompany && matchYear && matchType;
       }).length;
    }
    return scope.docIds.length;
  }, [docs, scope]);

  // --- Effects ---
  useEffect(() => {
    if (!isAuthenticated) { setCurrentUser(null); return; }
    let cancelled = false;
    getMe().then(u => { if (!cancelled) setCurrentUser(u); }).catch(console.error);
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  const refreshDocs = useCallback(async () => {
    const [docsRes, opts] = await Promise.all([listDocuments(), fetchFilterOptions()]);
    setDocs(docsRes.documents.map(toUiDoc));
    setFilterOptions(opts);
  }, []);

  useEffect(() => {
    if (!isAuthenticated) { setDocs([]); setFilterOptions({ companies: [], years: [] }); return; }
    let cancelled = false;
    const unsubs: (() => void)[] = [];
    (async () => {
      try {
        const [docsRes, opts] = await Promise.all([listDocuments(), fetchFilterOptions()]);
        if (cancelled) return;
        const mapped = docsRes.documents.map(toUiDoc);
        setDocs(mapped);
        setFilterOptions(opts);
        for (const doc of mapped) {
          if (doc.status !== 'Processing') continue;
          const unsub = subscribeIngestionStream(
            doc.id,
            (ev) => {
              setDocs(prev => prev.map(d =>
                d.id === doc.id ? { ...d, ingestionStage: ev.stage, ingestionStageIndex: ev.stage_index, ingestionStageTotal: ev.stage_total } : d
              ));
            },
            () => { unsub(); void refreshDocs(); },
            () => {
              unsub();
              setDocs(prev => prev.map(d =>
                d.id === doc.id ? { ...d, status: 'Error', ingestionStage: undefined } : d
              ));
            },
          );
          unsubs.push(unsub);
        }
      } catch (err) { console.error('Failed to load documents:', err); }
    })();
    return () => { cancelled = true; unsubs.forEach(u => u()); };
  }, [isAuthenticated, refreshDocs]);

  const handleUpload = useCallback((uploaded: { id: string; original_filename: string; status: string; metadata: Record<string, unknown> }) => {
    const newDoc: Document = {
      id: uploaded.id,
      title: (uploaded.metadata?.company as string | undefined) ?? uploaded.original_filename,
      company: (uploaded.metadata?.company as string | undefined) ?? '',
      year: toYear(uploaded.metadata?.year),
      type: (uploaded.metadata?.type as string | undefined) ?? '',
      pages: 0,
      status: 'Processing',
      tags: [],
      ingestionStage: undefined,
    };
    setDocs(prev => [newDoc, ...prev]);

    const unsub = subscribeIngestionStream(
      uploaded.id,
      (ev) => {
        setDocs(prev => prev.map(d =>
          d.id === uploaded.id ? { ...d, ingestionStage: ev.stage, ingestionStageIndex: ev.stage_index, ingestionStageTotal: ev.stage_total } : d
        ));
      },
      () => {
        unsub();
        void refreshDocs();
      },
      () => {
        unsub();
        setDocs(prev => prev.map(d =>
          d.id === uploaded.id ? { ...d, status: 'Error', ingestionStage: undefined } : d
        ));
      },
    );
  }, [refreshDocs]);

  useEffect(() => {
    if (!isAuthenticated) return;
    let cancelled = false;
    fetchConversations()
      .then((res) => {
        if (cancelled) return;
        const mapped: Chat[] = res.conversations.map((c) => ({
          id: c.id,
          title: c.title ?? 'Untitled',
          createdAt: new Date(c.created_at).getTime(),
          conversationId: c.id,
        }));
        setChats(mapped);
      })
      .catch(err => console.error('Failed to load conversations:', err));
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  useEffect(() => {
    if (!isAuthenticated) { setModels(FALLBACK_MODELS); setModelsLoading(false); return; }
    let cancelled = false;
    setModelsLoading(true);
    (async () => {
      try {
        const fetched = await fetchModels();
        if (!cancelled && fetched.length > 0) {
          setModels(fetched);
          setActiveModel(prev => fetched.some(m => m.id === prev) ? prev : fetched[0].id);
        }
      } catch (err) { console.error('Failed to fetch models:', err); }
      finally { if (!cancelled) setModelsLoading(false); }
    })();
    return () => { cancelled = true; };
  }, [isAuthenticated]);

  useEffect(() => {
    if (!activeChat?.conversationId) return;
    const conversationId = activeChat.conversationId;
    if (messagesByConversation[conversationId]) return;
    let cancelled = false;
    setMessagesLoading(prev => ({ ...prev, [conversationId]: true }));
    (async () => {
      try {
        const response = await fetchMessages(conversationId, { limit: 50 });
        if (!cancelled) {
          setMessagesByConversation(prev => ({ ...prev, [conversationId]: response.messages.map(toUiMessage) }));
          setMessagesHasMore(prev => ({ ...prev, [conversationId]: response.has_more }));
          if (response.messages.length > 0) {
            const minSeq = Math.min(...response.messages.map(m => m.seq));
            setMessagesMinSeq(prev => ({ ...prev, [conversationId]: minSeq }));
          }
        }
      } catch (error) {
        console.error('Failed to fetch messages:', error);
        if (!cancelled) setMessagesByConversation(prev => ({ ...prev, [conversationId]: [] }));
      } finally {
        if (!cancelled) setMessagesLoading(prev => { const next = { ...prev }; delete next[conversationId]; return next; });
      }
    })();
    return () => { cancelled = true; };
  }, [activeChat?.conversationId, messagesByConversation]);

  useEffect(() => {
    if (messagesEndRef.current) messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
  }, [activeMessages, isTyping]);

  useEffect(() => {
    if (!activeChat?.conversationId || !messagesContainerRef.current) return;
    const conversationId = activeChat.conversationId;
    const container = messagesContainerRef.current;
    let scrollTimeout: ReturnType<typeof setTimeout> | null = null;
    const handleScroll = () => {
      if (scrollTimeout) clearTimeout(scrollTimeout);
      scrollTimeout = setTimeout(() => {
        if (container.scrollTop < 100 && messagesHasMore[conversationId] && !messagesLoading[conversationId]) {
          const minSeq = messagesMinSeq[conversationId];
          if (minSeq !== undefined && minSeq > 0) {
            setMessagesLoading(prev => ({ ...prev, [conversationId]: true }));
            fetchMessages(conversationId, { limit: 50, before_seq: minSeq })
              .then(response => {
                const uiMessages: Message[] = response.messages.map(toUiMessage);
                setMessagesByConversation(prev => {
                  const existing = prev[conversationId] || [];
                  const allMessages = [...uiMessages, ...existing];
                  const uniqueMessages = Array.from(new Map(allMessages.map(m => [m.id, m])).values());
                  uniqueMessages.sort((a, b) => a.timestamp - b.timestamp);
                  return { ...prev, [conversationId]: uniqueMessages };
                });
                setMessagesHasMore(prev => ({ ...prev, [conversationId]: response.has_more }));
                if (response.messages.length > 0) {
                  const newMinSeq = Math.min(...response.messages.map(m => m.seq));
                  setMessagesMinSeq(prev => ({ ...prev, [conversationId]: Math.min(prev[conversationId] ?? Infinity, newMinSeq) }));
                }
              })
              .catch(err => console.error('Failed to load older messages:', err))
              .finally(() => setMessagesLoading(prev => { const next = { ...prev }; delete next[conversationId]; return next; }));
          }
        }
      }, 100);
    };
    container.addEventListener('scroll', handleScroll);
    return () => { container.removeEventListener('scroll', handleScroll); if (scrollTimeout) clearTimeout(scrollTimeout); };
  }, [activeChat?.conversationId, messagesHasMore, messagesLoading, messagesMinSeq]);

  const modelCapabilities: ModelCapabilities = React.useMemo(() => {
    const supportsAdvanced = activeModel.startsWith('gpt-5') || activeModel.includes('o1') || activeModel.includes('o3');
    return {
      supportsTemperature: !supportsAdvanced,
      supportsReasoningEffort: supportsAdvanced,
      supportsVerbosity: supportsAdvanced,
    };
  }, [activeModel]);

  const fetchStats = useCallback(async (conversationId: string | undefined) => {
    if (!conversationId) { setLastRequestStats(null); setStatsHistory([]); return; }
    try {
      const { requests } = await fetchChatStats(conversationId, 50);
      const mapItem = (r: RequestStatsItem): RequestStats => {
        const input = r.input_tokens ?? 0;
        const output = r.output_tokens ?? 0;
        const reasoning = r.reasoning_tokens ?? 0;
        const chatTotal = r.total_tokens ?? input + output + reasoning;
        return {
          inputTokens: input, outputTokens: output, reasoningTokens: reasoning,
          totalTokens: chatTotal,
          pipelineTotalTokens: r.pipeline_total_tokens ?? chatTotal,
          cost: r.pipeline_cost_usd ?? r.cost_usd ?? 0,
          latencyMs: r.latency_ms ?? 0,
          ttftMs: r.ttft_ms ?? null,
          tps: r.tps ?? null,
          model: r.model,
          timestamp: new Date(r.created_at).getTime(),
        };
      };
      const history = requests.map(mapItem);
      setLastRequestStats(history[0] ?? null);
      setStatsHistory(history);
    } catch (err) { console.error('Failed to load stats:', err); }
  }, []);

  useEffect(() => {
    if (!isControlPaneOpen || !isAuthenticated) return;
    if (!activeChat?.conversationId) { setLastRequestStats(null); setStatsHistory([]); return; }
    fetchStats(activeChat.conversationId);
  }, [isControlPaneOpen, isAuthenticated, activeChatId, activeChat?.conversationId, fetchStats]);

  // --- Handlers ---
  const updateMessage = useCallback(
    (conversationId: string, msgId: string, updater: (msg: Message) => Message) => {
      setMessagesByConversation(prev => {
        const messages = prev[conversationId] || [];
        return { ...prev, [conversationId]: messages.map(m => m.id === msgId ? updater(m) : m) };
      });
    }, []
  );

  const appendMessage = useCallback(
    (conversationId: string, msg: Message) => {
      setMessagesByConversation(prev => ({
        ...prev,
        [conversationId]: [...(prev[conversationId] || []), msg],
      }));
    }, []
  );

  const stopTypingForText = useCallback((text?: string) => {
    if (!text || text.trim().length === 0) return;
    setIsTyping(false);
  }, []);

  const handleSendMessage = async (text: string = inputMessage) => {
    if (!text.trim() || !activeChatId || !activeChat?.conversationId) return;
    const conversationId = activeChat.conversationId;
    setInputMessage('');
    setIsTyping(true);
    setIsAwaitingResponse(true);
    const clientMsgId = crypto.randomUUID();
    const clientRequestId = crypto.randomUUID();
    const extraParams: Record<string, unknown> = {};
    if (modelCapabilities.supportsReasoningEffort && modelParams.reasoningEffort)
      extraParams.reasoning_effort = modelParams.reasoningEffort;
    if (modelCapabilities.supportsVerbosity && modelParams.verbosity)
      extraParams.verbosity = modelParams.verbosity;

    try {
      const enqueueRes = await chatEnqueue({
        conversation_id: conversationId,
        content: text,
        client_msg_id: clientMsgId,
        client_request_id: clientRequestId,
        model: activeModel,
        params: { temperature: modelParams.temperature, max_tokens: modelParams.maxTokens, ...extraParams },
        metadata: { scope: { mode: scope.mode, docIds: scope.docIds, filters: scope.filters } },
      });
      appendMessage(conversationId, { id: enqueueRes.user_message_id, role: 'user', content: text, timestamp: Date.now() });
      const placeholderId = enqueueRes.assistant_message_id;
      appendMessage(conversationId, { id: placeholderId, role: 'assistant', content: '', timestamp: Date.now() });
      await chatStreamSubscribe(
        enqueueRes.request_id,
        (chunk) => {
          updateMessage(conversationId, placeholderId, m => ({ ...m, content: m.content + chunk.text }));
          stopTypingForText(chunk.text);
        },
        (span) => {
          updateMessage(conversationId, placeholderId, m => ({
            ...m,
            citationSpans: [...(m.citationSpans || []), { start: span.start, end: span.end, refIds: span.ref_ids, displayLabels: span.display_labels }],
          }));
        },
        (refs) => {
          updateMessage(conversationId, placeholderId, m => ({
            ...m,
            references: refs.items.map(r => ({
              refId: r.ref_id, displayLabel: r.display_label, chunkId: r.chunk_id,
              documentId: r.document_id, documentName: r.document_name, filename: r.filename,
              pageNumbers: r.page_numbers, headingPath: r.heading_path, snippet: r.snippet,
              bboxHint: r.bbox_hint ?? undefined,
            })),
          }));
        },
        (chunk) => {
          if (chunk.text) updateMessage(conversationId, placeholderId, m => ({ ...m, content: m.content + chunk.text }));
          if (chunk.stats) fetchStats(conversationId);
          setIsTyping(false);
          setIsAwaitingResponse(false);
          // Mark snapshot done and clear live state
          setStageSnapshots(snaps => {
            const prev = snaps[placeholderId];
            if (!prev) return snaps;
            const finalRecs = prev.records.map(r => r.endedAt ? r : { ...r, endedAt: Date.now() });
            return { ...snaps, [placeholderId]: { ...prev, records: finalRecs, done: true } };
          });
        },
        (error) => {
          updateMessage(conversationId, placeholderId, m => ({ ...m, content: m.content || `Error: ${error.message}` }));
          setIsTyping(false);
          setIsAwaitingResponse(false);
        },
        undefined,
        (meta) => {
          updateMessage(conversationId, placeholderId, m => ({
            ...m,
            metadata: { confidence: meta.confidence, ungrounded_claims: meta.ungrounded_claims, route: meta.route },
          }));
        },
        (stage) => {
          const now = Date.now();
          setStageSnapshots(snaps => {
            const prev = snaps[placeholderId];
            const records: StageRecord[] = prev?.records ?? [];
            const closed = records.map((r: StageRecord) => r.endedAt ? r : { ...r, endedAt: now });
            const exists = closed.find((r: StageRecord) => r.stage === stage.stage);
            const next = exists ? closed : [...closed, { stage: stage.stage, startedAt: now }];
            return { ...snaps, [placeholderId]: { current: stage, records: next, toolCalls: prev?.toolCalls ?? [], done: false } };
          });
        },
        (ev) => {
          // tool_call_started: add a new in-flight entry
          setStageSnapshots(snaps => {
            const prev = snaps[placeholderId];
            if (!prev) return snaps;
            const tc: ToolCallRecord = { entity: ev.entity, startedAt: Date.now() };
            return { ...snaps, [placeholderId]: { ...prev, toolCalls: [...prev.toolCalls, tc] } };
          });
        },
        (ev) => {
          // tool_call_completed: close the most recent entry for this entity
          setStageSnapshots(snaps => {
            const prev = snaps[placeholderId];
            if (!prev) return snaps;
            let found = false;
            const updated = [...prev.toolCalls].reverse().map(tc => {
              if (!found && tc.entity === ev.entity && !tc.endedAt) {
                found = true;
                return { ...tc, endedAt: Date.now(), chunksReturned: ev.chunks_returned };
              }
              return tc;
            }).reverse();
            return { ...snaps, [placeholderId]: { ...prev, toolCalls: updated } };
          });
        },
        (ev) => {
          setChats(prev => prev.map(c =>
            c.conversationId === ev.conversation_id ? { ...c, title: ev.title } : c
          ));
        },
      );
    } catch (err) {
      const error = err as ApiError;
      appendMessage(conversationId, { id: `temp-${clientMsgId}`, role: 'user', content: text, timestamp: Date.now() });
      appendMessage(conversationId, { id: `error-${Date.now()}`, role: 'assistant', content: `Error: ${error.message}`, timestamp: Date.now() });
      setIsTyping(false);
      setIsAwaitingResponse(false);
    }
  };

  const handleCitationClick = (citation: Citation) => {
    const doc = docs.find(d => d.id === citation.docId);
    if (!doc) return;
    setIsEvidenceOpen(true); setMobileTab('EVIDENCE');
    setOpenPdfTabs(prev => {
      const exists = prev.find(p => p.doc.id === doc.id);
      if (exists) return prev.map(p => p.doc.id === doc.id ? { ...p, page: citation.page } : p);
      return [...prev, { doc, page: citation.page }];
    });
    setActivePdfDocId(doc.id);
    setActiveHighlight(citation.bboxHint ? { bbox: citation.bboxHint, label: '' } : undefined);
  };

  const handleReferenceClick = (ref: ReferenceItem) => {
    const doc = docs.find(d => d.id === ref.documentId) ?? {
      id: ref.documentId,
      title: ref.documentName || ref.filename || 'Unknown document',
      company: '',
      year: 0,
      type: '',
      pages: 0,
      status: 'Ready' as const,
      tags: [],
    };
    setIsEvidenceOpen(true); setMobileTab('EVIDENCE');
    const page = ref.pageNumbers[0] || 1;
    setOpenPdfTabs(prev => {
      const exists = prev.find(p => p.doc.id === doc.id);
      if (exists) return prev.map(p => p.doc.id === doc.id ? { ...p, page } : p);
      return [...prev, { doc, page }];
    });
    setActivePdfDocId(doc.id);
    setActiveHighlight(ref.bboxHint ? { bbox: ref.bboxHint, label: ref.displayLabel } : undefined);
  };

  const handleNewChat = async () => {
    try {
      const response = await createConversation({ title: 'New Analysis', settings: {} });
      const newChat: Chat = {
        id: Date.now().toString(),
        title: 'New Analysis',
        createdAt: Date.now(),
        conversationId: response.conversation_id,
      };
      setChats([newChat, ...chats]);
      setActiveChatId(newChat.id);
      setMessagesByConversation(prev => ({ ...prev, [response.conversation_id]: [] }));
      if (window.innerWidth < 768) setIsSidebarOpen(false);
    } catch (error) {
      console.error('Failed to create conversation:', error);
      alert('Failed to create conversation. Please try again.');
    }
  };

  const handleDeleteRequest = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setChatToDelete(id);
  };

  const confirmDeleteChat = async () => {
    if (!chatToDelete) return;
    const chat = chats.find(c => c.id === chatToDelete);
    if (!chat?.conversationId) { setChatToDelete(null); return; }
    try {
      await deleteConversation(chat.conversationId);
      setChats(prev => prev.filter(c => c.id !== chatToDelete));
      if (activeChatId === chatToDelete) setActiveChatId(null);
      setMessagesByConversation(prev => { const next = { ...prev }; delete next[chat.conversationId]; return next; });
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      alert('Failed to delete conversation. Please try again.');
    } finally { setChatToDelete(null); }
  };

  const handleStartRename = (e: React.MouseEvent, chat: Chat) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditingTitle(chat.title);
  };

  const handleCancelRename = () => { setEditingChatId(null); setEditingTitle(''); };

  const handleSaveRename = async (chat: Chat) => {
    if (!chat.conversationId || !editingTitle.trim()) { handleCancelRename(); return; }
    const newTitle = editingTitle.trim();
    if (newTitle === chat.title) { handleCancelRename(); return; }
    try {
      await updateConversation(chat.conversationId, { title: newTitle });
      setChats(prev => prev.map(c => c.id === chat.id ? { ...c, title: newTitle } : c));
      setEditingChatId(null); setEditingTitle('');
    } catch (error) {
      console.error('Failed to rename conversation:', error);
      alert('Failed to rename conversation. Please try again.');
      setEditingTitle(chat.title);
    }
  };

  // --- Renderers ---

  const renderSidebar = () => {
    const q = chatSearch.toLowerCase();
    const filteredChats = q ? chats.filter(c => c.title.toLowerCase().includes(q)) : chats;
    const chatGroups = groupChatsByDate(filteredChats);
    const collapsed = isSidebarCollapsed;

    const renderChatItem = (chat: Chat) => (
      <div
        key={chat.id}
        onClick={() => { if (editingChatId !== chat.id) { setActiveChatId(chat.id); if (window.innerWidth < 768) setIsSidebarOpen(false); } }}
        className={`
          group flex items-center justify-between p-2.5 rounded-lg mb-0.5 cursor-pointer transition-colors
          ${activeChatId === chat.id
            ? 'bg-[var(--surface-3)] text-[var(--text)] border border-[var(--border)]'
            : 'text-[var(--text-muted)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] border border-transparent'}
        `}
      >
        {editingChatId === chat.id ? (
          <input
            type="text"
            value={editingTitle}
            onChange={e => setEditingTitle(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSaveRename(chat); else if (e.key === 'Escape') handleCancelRename(); }}
            onBlur={() => handleSaveRename(chat)}
            autoFocus
            className="flex-1 bg-[var(--input-bg)] border border-[var(--input-border)] rounded px-2 py-1 text-sm text-[var(--text)] focus:outline-none focus:border-[var(--input-border-focus)]"
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <div className="truncate text-sm font-medium pr-1 flex-1 min-w-0">{chat.title}</div>
        )}
        {editingChatId !== chat.id && (
          <div className="flex items-center gap-0.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <button onClick={e => handleStartRename(e, chat)} className="p-1 rounded hover:text-[var(--accent)]">
              <Pencil size={11} />
            </button>
            <button onClick={e => handleDeleteRequest(e, chat.id)} className="p-1 rounded hover:text-[var(--danger)]">
              <Trash2 size={11} />
            </button>
          </div>
        )}
      </div>
    );

    return (
      <div className={`
        fixed inset-y-0 left-0 z-30 bg-[var(--surface-1)] border-r border-[var(--border)]
        transform transition-all duration-200 ease-in-out flex flex-col
        md:relative md:translate-x-0
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        ${collapsed ? 'w-16' : 'w-[280px]'}
      `}>
        {/* New Chat button */}
        <div className={`shrink-0 ${collapsed ? 'p-2' : 'p-3'}`}>
          {collapsed ? (
            <>
              <button
                onClick={handleNewChat}
                className="w-full flex items-center justify-center p-2 rounded-lg bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] transition-colors"
                title="New Chat"
              >
                <Plus size={16} />
              </button>
              <button
                onClick={() => setIsSidebarCollapsed(false)}
                className="w-full flex items-center justify-center p-2 mt-1 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors hidden md:flex"
                title="Expand sidebar"
              >
                <ChevronRight size={16} />
              </button>
            </>
          ) : (
            <Button className="w-full justify-start gap-2" onClick={handleNewChat}>
              <Plus size={16} /> New Chat
            </Button>
          )}
        </div>

        {!collapsed && (
          <div className="px-3 pb-2 shrink-0 flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-2.5 text-[var(--text-faint)]" size={13} />
              <Input placeholder="Search chats..." className="pl-8 h-9 text-xs" value={chatSearch} onChange={e => setChatSearch(e.target.value)} />
            </div>
            <button
              onClick={() => setIsSidebarCollapsed(true)}
              className="p-1.5 rounded-md text-[var(--text-faint)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors hidden md:flex shrink-0"
              title="Collapse sidebar"
            >
              <PanelLeft size={16} />
            </button>
          </div>
        )}

        {/* Chat list */}
        {!collapsed && (
          <div className="flex-1 overflow-y-auto px-2 min-h-0 pb-2">
            {chatGroups.length === 0 && (
              <div className="text-xs text-[var(--text-faint)] text-center py-6">
                {chatSearch ? 'No matching chats' : 'No conversations yet'}
              </div>
            )}
            {chatGroups.map(group => (
              <div key={group.label}>
                <div className="px-2 py-2 text-[10px] font-semibold text-[var(--text-faint)] uppercase tracking-widest select-none">
                  {group.label}
                </div>
                {group.items.map(renderChatItem)}
              </div>
            ))}
          </div>
        )}

        {/* Collapsed: keep only quick actions */}
        {collapsed && (
          <div className="flex-1 overflow-y-auto flex flex-col items-center gap-1 px-2 py-1 min-h-0">
            <button
              onClick={() => setIsSidebarCollapsed(false)}
              title="Search chats"
              className="w-9 h-9 rounded-lg flex items-center justify-center text-[var(--text-faint)] hover:bg-[var(--surface-2)] hover:text-[var(--text)] transition-colors"
            >
              <Search size={14} />
            </button>
          </div>
        )}

        {/* User profile */}
        <div className={`border-t border-[var(--border)] mt-auto shrink-0 bg-[var(--surface-1)] ${collapsed ? 'p-2' : 'p-3'}`}>
          <div className={`flex items-center ${collapsed ? 'justify-center' : 'gap-3'}`}>
            <div className="h-8 w-8 rounded-full bg-[var(--surface-2)] flex items-center justify-center border border-[var(--border)] flex-shrink-0">
              <User size={14} className="text-[var(--text-muted)]" />
            </div>
            {!collapsed && (
              <>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-[var(--text)] truncate">
                    {currentUser?.display_name ?? currentUser?.email ?? 'User'}
                  </div>
                  <div className="text-xs text-[var(--text-faint)] truncate">
                    {currentUser?.display_name ? currentUser.email : (currentUser ? 'Pro Plan' : null)}
                  </div>
                </div>
                <button
                  onClick={() => logout()}
                  className="p-1.5 text-[var(--icon)] hover:text-[var(--accent)] hover:bg-[var(--surface-2)] rounded-md transition-colors"
                  title="Log out"
                >
                  <LogOut size={15} />
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderChatArea = () => {
    if (!activeChatId) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center text-center p-8 animate-fade-in">
          <div className="w-16 h-16 bg-[var(--surface-2)] rounded-2xl flex items-center justify-center mb-6 border border-[var(--border)]">
            <Bot className="text-[var(--text-faint)]" size={32} />
          </div>
          <h2 className="text-xl font-semibold text-[var(--text)] mb-2">AI Financial Copilot</h2>
          <p className="text-[var(--text-muted)] max-w-md mb-6 leading-relaxed">
            Select a conversation from the sidebar or start a new chat.
          </p>
          <Button onClick={handleNewChat} className="mb-6">
            <Plus size={16} className="mr-2" /> Start New Chat
          </Button>
        </div>
      );
    }

    const conversationId = activeChat?.conversationId;
    const isChatLoading = conversationId !== undefined &&
      messagesByConversation[conversationId] === undefined;
    const isChatEmpty = !isChatLoading && activeMessages.length === 0;

    return (
      <div className="flex-1 flex flex-col min-w-0 h-full pb-14 md:pb-0">
        <ScopeBar
          scope={scope}
          docCount={activeDocsCount}
          onModeChange={m => setScope(s => ({ ...s, mode: m }))}
          onFilterChange={f => setScope(s => ({ ...s, filters: { ...s.filters, ...f } }))}
          onAddFiles={() => setIsDocPickerOpen(true)}
          filterOptions={filterOptions}
        />

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto min-h-0 p-4 md:p-8 pb-32 md:pb-8">
          {isChatLoading ? (
            <div className="h-full flex flex-col items-center justify-center gap-4">
              <Loader2 className="w-6 h-6 animate-spin text-[var(--accent)]" />
            </div>
          ) : isChatEmpty ? (
            <div className="h-full flex flex-col items-center justify-center text-center animate-fade-in">
              <div className="w-16 h-16 bg-[var(--surface-2)] rounded-2xl flex items-center justify-center mb-5 border border-[var(--border)]">
                <Bot className="text-[var(--text-faint)]" size={32} />
              </div>
              <h2 className="text-xl font-semibold text-[var(--text)] mb-2">Ask anything</h2>
              <p className="text-[var(--text-muted)] max-w-md mb-6 leading-relaxed text-sm">
                I can help you analyze financial documents, extract key metrics, and compare company performance.
              </p>
            </div>
          ) : (
            <div className="space-y-8 max-w-[var(--content-max)] mx-auto">
              {messagesHasMore[activeChat.conversationId] && (
                <div className="text-center text-sm text-[var(--text-faint)] py-2">
                  {messagesLoading[activeChat.conversationId] ? 'Loading older messages…' : 'Scroll up to load more'}
                </div>
              )}
              {activeMessages.map((msg, msgIdx) => {
                if (msg.role === 'assistant' && !msg.content) return null;

                const prevMsg = msgIdx > 0 ? activeMessages[msgIdx - 1] : null;
                const isFirstInTurn = msg.role === 'assistant' && (prevMsg === null || prevMsg.role === 'user');
                const isLastMsg = msgIdx === activeMessages.length - 1;
                const isStreaming = isLastMsg && msg.role === 'assistant' && isTyping && msg.content.length > 0;

                return (
                  <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'items-start'}`}>
                    {/* Assistant avatar — only on first message of a turn */}
                    {msg.role === 'assistant' && (
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 transition-opacity ${
                        isFirstInTurn
                          ? 'bg-[var(--accent-subtle)] border border-[var(--accent)] border-opacity-20 opacity-100'
                          : 'opacity-0 pointer-events-none'
                      }`}>
                        <Bot size={16} className="text-[var(--accent)]" />
                      </div>
                    )}

                    <div className={`space-y-2 ${msg.role === 'user' ? 'flex flex-col items-end ml-[15%] max-w-[85%]' : 'flex-1 min-w-0'}`}>
                      {msg.role === 'assistant' && stageSnapshots[msg.id] && (
                        <AgentTimeline snapshot={stageSnapshots[msg.id]} />
                      )}
                      {msg.role === 'assistant' && (msg.metadata as MessageMetadata)?.route === 'retrieve' && (msg.metadata as MessageMetadata)?.confidence && ['low', 'none'].includes((msg.metadata as MessageMetadata).confidence!) && (
                        <div className="pl-4 mb-1">
                          <span className="inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                            Limited source coverage — answer may be incomplete
                          </span>
                        </div>
                      )}
                      {msg.role === 'assistant' ? (
                        <div className="pl-4 text-sm leading-relaxed text-[var(--text)]">
                          {msg.citationSpans && msg.citationSpans.length > 0 ? (
                            <CitedText
                              text={msg.content}
                              spans={msg.citationSpans}
                              references={msg.references || []}
                              onCitationClick={handleReferenceClick}
                            />
                          ) : (
                            <>
                              <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                                {msg.content}
                              </ReactMarkdown>
                              {isStreaming && (
                                <span className="animate-caret-blink font-mono text-[var(--accent)] opacity-80">▊</span>
                              )}
                            </>
                          )}
                        </div>
                      ) : (
                        /* User bubble */
                        <div className="p-4 rounded-2xl text-sm leading-relaxed bg-[var(--bubble-user-bg)] text-[var(--text)] shadow-[var(--shadow-xs)]"
                          style={{ width: 'fit-content' }}>
                          {msg.citationSpans && msg.citationSpans.length > 0 ? (
                            <CitedText
                              text={msg.content}
                              spans={msg.citationSpans}
                              references={msg.references || []}
                              onCitationClick={handleReferenceClick}
                            />
                          ) : (
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD_COMPONENTS}>
                              {msg.content}
                            </ReactMarkdown>
                          )}
                        </div>
                      )}

                      {/* Action row (Copy + Thumbs Up/Down) for completed assistant messages */}
                      {msg.role === 'assistant' && !isStreaming && msg.content && activeChat?.conversationId && (
                        <MessageActions
                          messageId={msg.id}
                          content={msg.content}
                          feedback={msg.feedback ?? null}
                          onFeedbackChange={(fb) => updateMessage(activeChat.conversationId, msg.id, (m) => ({ ...m, feedback: fb }))}
                        />
                      )}

                      {/* Evidence section */}
                      {msg.references && msg.references.length > 0 && (
                        <div className={msg.role === 'assistant' ? 'pl-2' : ''}>
                          <EvidenceList references={msg.references} onRefClick={handleReferenceClick} />
                        </div>
                      )}
                      {/* Legacy citations */}
                      {!msg.references?.length && msg.citations && msg.citations.length > 0 && (
                        <div className={`flex flex-wrap gap-2 mt-2 ${msg.role === 'assistant' ? 'pl-2' : ''} ${msg.role === 'user' ? 'justify-end' : ''}`}>
                          {msg.citations.map((c, i) => {
                            const doc = docs.find(d => d.id === c.docId);
                            return (
                              <button
                                key={i}
                                onClick={() => handleCitationClick(c)}
                                className="flex items-center gap-2 bg-[var(--surface-1)] border border-[var(--border)] hover:border-[var(--accent)] hover:bg-[var(--surface-2)] px-3 py-2 rounded-lg text-left transition-all group max-w-xs"
                              >
                                <div className="h-8 w-8 bg-[var(--bg)] flex items-center justify-center rounded-md text-[var(--text-faint)] group-hover:text-[var(--accent)] border border-[var(--border)]">
                                  <span className="font-mono text-xs font-bold">{i + 1}</span>
                                </div>
                                <div className="min-w-0">
                                  <div className="text-xs font-medium text-[var(--text)] truncate">{doc?.company}</div>
                                  <div className="text-[10px] text-[var(--text-faint)] font-mono truncate">Page {c.page} · {doc?.type}</div>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Typing indicator — shown while placeholder message has no content yet */}
              {isTyping && activeMessages[activeMessages.length - 1]?.content === '' && (() => {
                const placeholderSnap = stageSnapshots[activeMessages[activeMessages.length - 1]?.id ?? ''];
                return (
                  <div className="flex gap-3 items-start max-w-[var(--content-max)] mx-auto">
                    <div className="w-8 h-8 rounded-lg bg-[var(--accent-subtle)] border border-[var(--accent)] border-opacity-20 flex items-center justify-center flex-shrink-0">
                      <Bot size={16} className="text-[var(--accent)]" />
                    </div>
                    {placeholderSnap && <AgentTimeline snapshot={placeholderSnap} />}
                  </div>
                );
              })()}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="p-4 bg-[var(--bg)]">
          <div className="max-w-3xl mx-auto">
            <div className={`rounded-2xl border-[1.5px] bg-[var(--input-bg)] transition-all duration-200 ${
              isComposerFocused
                ? 'border-[var(--accent)] shadow-[0_0_0_3px_var(--focus-ring)]'
                : 'border-[var(--input-border)] shadow-[var(--shadow-sm)]'
            }`}>
              {/* Model pill */}
              <div className="flex items-center gap-2 px-4 pt-3 pb-1">
                <div className="relative">
                  <select
                    value={activeModel}
                    onChange={e => setActiveModel(e.target.value)}
                    className="appearance-none text-xs font-semibold text-[var(--accent)] bg-[var(--input-bg)]
                      border border-[var(--accent)] border-opacity-30 rounded-full px-3 py-1 pr-6
                      outline-none cursor-pointer hover:border-opacity-100 transition-colors"
                  >
                    {models.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                  <ChevronDown className="absolute right-1.5 top-[5px] text-[var(--accent)] pointer-events-none" size={10} />
                </div>
              </div>

              {/* Textarea */}
              <textarea
                value={inputMessage}
                onChange={e => setInputMessage(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                onFocus={() => setIsComposerFocused(true)}
                onBlur={() => setIsComposerFocused(false)}
                placeholder="Ask me anything…"
                className="w-full bg-transparent px-4 py-2 text-sm text-[var(--text)] placeholder:text-[var(--placeholder)] focus:outline-none min-h-[72px] resize-none font-sans"
              />

              {/* Bottom bar */}
              <div className="flex items-center justify-between px-4 pb-3 pt-1">
                <div className="hidden sm:flex items-center gap-1.5 text-[10px] text-[var(--text-faint)]">
                  <kbd className="px-1 py-0.5 bg-[var(--surface-2)] rounded border border-[var(--border)] font-mono text-[9px]">↵</kbd>
                  <span>Send</span>
                  <span className="opacity-40 mx-1">·</span>
                  <kbd className="px-1 py-0.5 bg-[var(--surface-2)] rounded border border-[var(--border)] font-mono text-[9px]">⇧↵</kbd>
                  <span>New line</span>
                </div>
                <button
                  onClick={() => handleSendMessage()}
                  disabled={!inputMessage.trim() || isAwaitingResponse}
                  className={`w-9 h-9 rounded-full flex items-center justify-center transition-all press-scale ${
                    inputMessage.trim() && !isAwaitingResponse
                      ? 'bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] shadow-[var(--shadow-sm)]'
                      : 'bg-[var(--surface-2)] text-[var(--text-faint)] cursor-not-allowed'
                  }`}
                >
                  <Send size={15} />
                </button>
              </div>
            </div>
            <div className="text-center mt-2 text-[10px] text-[var(--text-faint)]">
              AI can make mistakes. Verify with citations.
            </div>
          </div>
        </div>
      </div>
    );
  };

  const confirmDeleteDoc = async () => {
    if (!docToDelete) return;
    setDocDeleteLoading(true);
    try {
      await deleteDocument(docToDelete);
      setDocs(prev => prev.filter(d => d.id !== docToDelete));
    } catch (err) {
      console.error('Failed to delete document:', err);
      alert('Failed to delete document. Please try again.');
    } finally {
      setDocDeleteLoading(false);
      setDocToDelete(null);
    }
  };

  const renderLibrary = () => {
    const companies = Array.from(new Set(docs.map(d => d.company).filter(Boolean))).sort();
    const years = Array.from(new Set(docs.map(d => d.year).filter(y => y > 0))).sort((a, b) => Number(b) - Number(a));

    const q = libSearch.toLowerCase();
    const filteredDocs = docs.filter(doc => {
      if (q && !doc.title.toLowerCase().includes(q) && !doc.company.toLowerCase().includes(q)) return false;
      if (libCompany && doc.company !== libCompany) return false;
      if (libYear && String(doc.year) !== libYear) return false;
      return true;
    });

    return (
      <div className="flex-1 overflow-auto bg-[var(--bg)] p-6 md:p-10 animate-fade-in pb-20 md:pb-10">
        {docToDelete && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
            <Card className="w-full max-w-sm p-6 shadow-[var(--shadow-lg)]" variant="elevated">
              <div className="flex flex-col items-center text-center gap-4">
                <div className="w-12 h-12 bg-[var(--danger-bg)] rounded-full flex items-center justify-center text-[var(--danger)]">
                  <AlertTriangle size={24} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-[var(--text)]">Delete Document?</h3>
                  <p className="text-sm text-[var(--text-muted)] mt-1">This will permanently remove the document and all its data.</p>
                </div>
                <div className="flex w-full gap-2 mt-2">
                  <Button variant="secondary" className="flex-1" onClick={() => setDocToDelete(null)} disabled={docDeleteLoading}>Cancel</Button>
                  <Button variant="danger" className="flex-1" onClick={confirmDeleteDoc} disabled={docDeleteLoading}>
                    {docDeleteLoading ? <Loader2 size={15} className="animate-spin" /> : 'Delete'}
                  </Button>
                </div>
              </div>
            </Card>
          </div>
        )}

        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-2xl font-semibold text-[var(--text)] mb-2">Document Library</h1>
              <p className="text-[var(--text-muted)] text-sm">
                {docs.length} document{docs.length !== 1 ? 's' : ''} · {docs.filter(d => d.status === 'Ready').length} ready
              </p>
            </div>
            <Button onClick={() => setIsUploadOpen(true)} className="gap-2">
              <Plus size={16} /> Upload PDFs
            </Button>
          </div>

          <div className="flex gap-4 mb-6 flex-wrap">
            <div className="relative flex-1 min-w-[200px] max-w-md">
              <Search className="absolute left-3 top-2.5 text-[var(--text-faint)]" size={14} />
              <Input
                placeholder="Search documents..."
                className="pl-9"
                value={libSearch}
                onChange={e => setLibSearch(e.target.value)}
              />
            </div>
            <select
              value={libCompany}
              onChange={e => setLibCompany(e.target.value)}
              className="appearance-none bg-[var(--input-bg)] border border-[var(--input-border)] rounded-md px-3 pr-8 text-sm text-[var(--text)] focus:border-[var(--input-border-focus)] outline-none cursor-pointer"
            >
              <option value="">All Companies</option>
              {companies.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <select
              value={libYear}
              onChange={e => setLibYear(e.target.value)}
              className="appearance-none bg-[var(--input-bg)] border border-[var(--input-border)] rounded-md px-3 pr-8 text-sm text-[var(--text)] focus:border-[var(--input-border-focus)] outline-none cursor-pointer"
            >
              <option value="">All Years</option>
              {years.map(y => <option key={y} value={String(y)}>{y}</option>)}
            </select>
          </div>

          {docs.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center py-24 border border-dashed border-[var(--border)] rounded-xl">
              <FileText size={40} className="text-[var(--text-faint)] mb-4" />
              <h3 className="text-base font-semibold text-[var(--text)] mb-1">No documents yet</h3>
              <p className="text-sm text-[var(--text-muted)] mb-4">Upload PDFs to start analyzing your financial documents.</p>
              <Button onClick={() => setIsUploadOpen(true)} className="gap-2"><Plus size={16} /> Upload PDFs</Button>
            </div>
          ) : filteredDocs.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center py-24 border border-dashed border-[var(--border)] rounded-xl">
              <Search size={40} className="text-[var(--text-faint)] mb-4" />
              <h3 className="text-base font-semibold text-[var(--text)] mb-1">No results</h3>
              <p className="text-sm text-[var(--text-muted)]">Try adjusting your search or filters.</p>
            </div>
          ) : (
            <div className="border border-[var(--border)] rounded-xl overflow-hidden bg-[var(--surface-1)] shadow-[var(--shadow-sm)]">
              <table className="w-full text-left text-sm">
                <thead className="bg-[var(--surface-2)] border-b border-[var(--border)] text-[var(--text-muted)] text-xs uppercase tracking-wider">
                  <tr>
                    <th className="px-6 py-4 font-medium w-48">Status</th>
                    <th className="px-6 py-4 font-medium">Document Name</th>
                    <th className="px-6 py-4 font-medium">Company</th>
                    <th className="px-6 py-4 font-medium">Year</th>
                    <th className="px-6 py-4 font-medium">Pages</th>
                    <th className="px-6 py-4 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {filteredDocs.map(doc => (
                    <tr
                      key={doc.id}
                      className="hover:bg-[var(--surface-2)] transition-colors group cursor-pointer"
                      onClick={() => {
                        setView('ASK');
                        handleCitationClick({ docId: doc.id, page: 1, excerpt: '' });
                        handleNewChat();
                        setScope({ mode: 'thisDoc', docIds: [doc.id], filters: {} });
                      }}
                    >
                      <td className="px-6 py-4 w-40">
                        <Badge variant={doc.status === 'Ready' ? 'success' : doc.status === 'Error' ? 'danger' : 'warning'}>
                          {doc.status === 'Processing' && doc.ingestionStage
                            ? `${INGEST_STAGE_LABELS[doc.ingestionStage] ?? doc.ingestionStage.replace(/_/g, ' ')}${doc.ingestionStageIndex != null && doc.ingestionStageTotal != null ? ` ${doc.ingestionStageIndex}/${doc.ingestionStageTotal}` : ''}`
                            : doc.status}
                        </Badge>
                      </td>
                      <td className="px-6 py-4 font-medium text-[var(--text)]">
                        <div className="flex items-center gap-2">
                          <FileText size={16} className="text-[var(--text-faint)] shrink-0" />
                          <span className="truncate max-w-[280px]">{doc.title}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-[var(--text-muted)]">{doc.company || '—'}</td>
                      <td className="px-6 py-4 font-mono text-[var(--text-faint)]">{doc.year || '—'}</td>
                      <td className="px-6 py-4 font-mono text-[var(--text-faint)]">{doc.pages || '—'}</td>
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={e => { e.stopPropagation(); setDocToDelete(doc.id); }}
                          className="p-1.5 rounded-md text-[var(--text-faint)] hover:text-[var(--danger)] hover:bg-[var(--danger-bg)] transition-colors opacity-0 group-hover:opacity-100"
                          title="Delete document"
                        >
                          <Trash2 size={15} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  };

  if (!authChecked) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[var(--bg)]">
        <Loader2 className="w-8 h-8 animate-spin text-[var(--accent)]" />
      </div>
    );
  }
  if (!isAuthenticated) {
    return <LoginPage onSuccess={token => setAccessToken(token)} />;
  }

  return (
    <div className="flex flex-col h-screen bg-[var(--bg)] text-[var(--text)] overflow-hidden font-sans">
      <UploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} onUpload={handleUpload} />
      <DocPickerModal
        isOpen={isDocPickerOpen}
        onClose={() => setIsDocPickerOpen(false)}
        docs={docs}
        selectedIds={scope.docIds}
        onConfirm={ids => { setScope(prev => ({ ...prev, mode: 'selectedDocs', docIds: ids })); setIsDocPickerOpen(false); }}
      />

      {/* Delete Confirmation Modal */}
      {chatToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 modal-backdrop-enter">
          <Card className="w-full max-w-sm p-6 shadow-[var(--shadow-lg)] modal-card-enter" variant="elevated">
            <div className="flex flex-col items-center text-center gap-4">
              <div className="w-12 h-12 bg-[var(--danger-bg)] rounded-full flex items-center justify-center text-[var(--danger)]">
                <AlertTriangle size={24} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-[var(--text)]">Delete Chat?</h3>
                <p className="text-sm text-[var(--text-muted)] mt-1">This action cannot be undone.</p>
              </div>
              <div className="flex w-full gap-2 mt-2">
                <Button variant="secondary" className="flex-1" onClick={() => setChatToDelete(null)}>Cancel</Button>
                <Button variant="danger" className="flex-1" onClick={confirmDeleteChat}>Delete</Button>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Global Header — spans full width regardless of view */}
      <header className="h-14 border-b border-[var(--border)] flex items-end justify-between bg-[var(--surface-1)] z-10 shrink-0">
        <div className="flex items-end">
          {/* Logo */}
          <div className="flex items-center gap-2.5 self-center px-4">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
              <Layers size={18} />
            </Button>
            <div className="h-7 w-7 bg-[var(--accent)] rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="font-bold text-white font-mono text-xs">AI</span>
            </div>
            <span className="font-semibold text-[var(--text)] tracking-tight text-sm hidden md:block">Financial Copilot</span>
          </div>

          {/* Divider */}
          <div className="w-px h-6 bg-[var(--border)] self-center mx-1" />

          {/* View switcher — tab style with underline */}
          {([
            { id: 'ASK' as ViewMode, label: 'Ask', Icon: MessageSquare },
            { id: 'LIBRARY' as ViewMode, label: 'Library', Icon: BookOpen },
          ]).map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setView(id)}
              className={`flex items-center gap-2 px-4 h-14 text-sm font-semibold transition-colors relative ${
                view === id
                  ? 'text-[var(--accent)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text)]'
              }`}
            >
              <Icon size={16} />
              {label}
              {view === id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--accent)] rounded-t-full" />
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 self-center">
          <button
            onClick={() => setIsControlPaneOpen(!isControlPaneOpen)}
            className={`p-2 rounded-lg transition-colors ${
              isControlPaneOpen
                ? 'text-[var(--accent)] bg-[var(--accent-subtle)]'
                : 'text-[var(--icon)] hover:text-[var(--text)] hover:bg-[var(--surface-2)]'
            }`}
            title="Control Pane"
          >
            <SlidersHorizontal size={18} />
          </button>
          <button
            onClick={cycleTheme}
            className="p-2 rounded-lg text-[var(--icon)] hover:text-[var(--text)] hover:bg-[var(--surface-2)] transition-colors"
            title={themeLabel}
            aria-label={themeLabel}
          >
            {themeIcon}
          </button>
          <Badge variant="outline" className="h-7 px-3">v2.4.0</Badge>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden relative">
        {view === 'ASK' && renderSidebar()}

        {/* Mobile overlay */}
        {isSidebarOpen && window.innerWidth < 768 && (
          <div className="fixed inset-0 z-20 md:hidden sidebar-scrim" onClick={() => setIsSidebarOpen(false)} />
        )}

        {/* Center Workspace */}
        <div className="flex-1 flex flex-col min-w-0 bg-[var(--bg)] relative">
          {/* Main Views */}
          {view === 'LIBRARY' ? renderLibrary() : (
            <div className="flex-1 flex overflow-hidden">
              <div className={`flex-1 flex flex-col ${window.innerWidth < 768 && mobileTab === 'EVIDENCE' ? 'hidden' : 'flex'}`}>
                {renderChatArea()}
              </div>
              <EvidencePanel
                isOpen={isEvidenceOpen || (window.innerWidth < 768 && mobileTab === 'EVIDENCE')}
                onClose={() => { setIsEvidenceOpen(false); setMobileTab('CONVERSATION'); }}
                openDocs={openPdfTabs}
                activeDocId={activePdfDocId}
                onSwitchDoc={setActivePdfDocId}
                onCloseDoc={id => {
                  const newTabs = openPdfTabs.filter(t => t.doc.id !== id);
                  setOpenPdfTabs(newTabs);
                  if (newTabs.length === 0) { setIsEvidenceOpen(false); setMobileTab('CONVERSATION'); }
                  else if (activePdfDocId === id) setActivePdfDocId(newTabs[0].doc.id);
                }}
                onPageChange={(docId, page) =>
                  setOpenPdfTabs(prev => prev.map(t => t.doc.id === docId ? { ...t, page } : t))
                }
                highlight={activeHighlight?.bbox}
                highlightLabel={activeHighlight?.label}
              />
              <ControlPane
                isOpen={isControlPaneOpen}
                onToggle={() => setIsControlPaneOpen(!isControlPaneOpen)}
                params={modelParams}
                onParamsChange={p => setModelParams(prev => ({ ...prev, ...p }))}
                capabilities={modelCapabilities}
                stats={lastRequestStats}
                statsHistory={statsHistory}
              />
            </div>
          )}
        </div>
      </div>

      {/* Mobile Bottom Nav */}
      {view === 'ASK' && (
        <div className="md:hidden h-14 border-t border-[var(--border)] bg-[var(--surface-1)] grid grid-cols-2 fixed bottom-0 left-0 right-0 z-50 shadow-lg">
          <button
            onClick={() => { setMobileTab('CONVERSATION'); setIsEvidenceOpen(false); }}
            className={`flex flex-col items-center justify-center gap-1 transition-colors ${
              mobileTab === 'CONVERSATION' ? 'text-[var(--accent)] bg-[var(--accent-subtle)]' : 'text-[var(--text-muted)]'
            }`}
          >
            <MessageSquare size={18} />
            <span className="text-[10px] font-semibold">CHAT</span>
          </button>
          <button
            onClick={() => { setMobileTab('EVIDENCE'); setIsEvidenceOpen(true); }}
            className={`flex flex-col items-center justify-center gap-1 transition-colors ${
              mobileTab === 'EVIDENCE' ? 'text-[var(--accent)] bg-[var(--accent-subtle)]' : 'text-[var(--text-muted)]'
            }`}
          >
            <BookOpen size={18} />
            <span className="text-[10px] font-semibold">EVIDENCE</span>
          </button>
        </div>
      )}
    </div>
  );
}
